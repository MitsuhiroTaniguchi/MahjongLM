from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest
import pymahjong  # noqa: F401

from tenhou_tokenizer import TenhouTokenizer
from tenhou_tokenizer import engine
from tests.dataset_sample import DATASET_2023, get_dataset_2023_sample_zip
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event
from tests.validation_helpers import (
    trace_round_token_slices,
    validate_event_token_slice,
    validate_round_stepwise,
    validate_score_rotation,
    validate_token_stream,
)

ROOT = Path(__file__).resolve().parents[1]
SANMA_RAW = ROOT / "tests" / "fixtures" / "tenhou" / "2014091101gm-00b9-0000-5ca6b487.txt"
CONVERT = ROOT / "scripts" / "paifu_scraping" / "convert.pl"
KAN_MULTI_URA_GAME_ID = "2023/2023112600gm-00a9-0000-1bc26cca.txt"


def _convert_sanma_sample() -> dict:
    proc = subprocess.run(
        ["perl", "-T", str(CONVERT), str(SANMA_RAW)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_validation_helpers_accept_minimal_game() -> None:
    game = minimal_game([qipai_event(), pingju_event()])
    tokens = TenhouTokenizer().tokenize_game(game)

    validate_token_stream(tokens)
    validate_score_rotation(game)
    validate_round_stepwise(game["log"][0])


def test_validation_helpers_accept_converted_sanma_game() -> None:
    game = _convert_sanma_sample()
    tokens = TenhouTokenizer().tokenize_game(game)

    validate_token_stream(tokens)
    validate_score_rotation(game)


def test_converted_sanma_game_emits_hule_detail_tokens() -> None:
    game = _convert_sanma_sample()

    tokens = TenhouTokenizer().tokenize_game(game)

    assert "yaku_riichi" in tokens
    assert "yaku_menzen_tsumo" in tokens
    assert "han_6" in tokens
    assert "fu_30" in tokens


def test_dataset_game_with_kan_emits_multiple_ura_dora_reveals() -> None:
    if not DATASET_2023.exists():
        pytest.skip(f"missing dataset: {DATASET_2023}")

    with zipfile.ZipFile(get_dataset_2023_sample_zip()) as zf:
        game = json.load(zf.open(KAN_MULTI_URA_GAME_ID))

    assert any(
        isinstance(event, dict) and ("gang" in event or "kaigang" in event)
        for event in game["log"][4]
    )

    tokens = TenhouTokenizer().tokenize_game(game)

    ura_idx = tokens.index("ura_dora")
    assert tokens[ura_idx : ura_idx + 4] == ["ura_dora", "p2", "s2", "m4"]
    assert tokens.count("yaku_ura_dora") == 7


def test_stepwise_validation_accepts_call_then_discard_round() -> None:
    game = minimal_game(
        [
            qipai_event(hands=["m1456789p1234s11", "m23p123456s123z11", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 0, "p": "s2"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"fulou": {"l": 1, "m": "m1-23"}},
            {"dapai": {"l": 1, "p": "z1"}},
            pingju_event(),
        ]
    )

    validate_round_stepwise(game["log"][0])


def test_stepwise_validation_accepts_ankan_and_rinshan_round() -> None:
    game = minimal_game(
        [
            qipai_event(hands=["m111p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 0, "p": "m1"}},
            {"gang": {"l": 0, "m": "m1111"}},
            {"gangzimo": {"l": 0, "p": "p5"}},
            {"dapai": {"l": 0, "p": "p5_"}},
            pingju_event(),
        ]
    )

    validate_round_stepwise(game["log"][0])


def test_event_token_slice_rejects_fulou_detail_after_pass() -> None:
    with pytest.raises(AssertionError):
        validate_event_token_slice(
            "fulou",
            [
                "take_react_1_chi",
                "pass_react_2_pon_voluntary",
                "chi_pos_mid",
            ],
        )


def test_event_token_slice_rejects_discard_options_before_discard() -> None:
    with pytest.raises(AssertionError):
        validate_event_token_slice(
            "dapai",
            [
                "opt_react_1_ron",
                "discard_0_m1",
            ],
        )


def test_event_token_slice_rejects_bare_tile_after_draw() -> None:
    with pytest.raises(AssertionError):
        validate_event_token_slice(
            "zimo",
            [
                "draw_0_m1",
                "m2",
            ],
        )


def test_event_token_slice_rejects_ron_result_without_take() -> None:
    with pytest.raises(AssertionError):
        validate_event_token_slice(
            "hule",
            [
                "pass_react_2_ron_voluntary",
                "score_delta_0",
                "TENBO_ZERO",
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
            ],
        )


def test_event_token_slice_rejects_tsumo_result_without_take() -> None:
    with pytest.raises(AssertionError):
        validate_event_token_slice(
            "hule",
            [
                "pass_self_0_ankan",
                "m1",
                "score_delta_0",
                "TENBO_ZERO",
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
            ],
        )


def test_trace_round_token_slices_emits_event_ordered_blocks() -> None:
    round_data = [
        qipai_event(),
        {"zimo": {"l": 0, "p": "m1"}},
        {"dapai": {"l": 0, "p": "m1"}},
        pingju_event(),
    ]

    tokenizer, traces = trace_round_token_slices(round_data)

    assert len(traces) == 4
    assert [trace["event_key"] for trace in traces] == ["qipai", "zimo", "dapai", "pingju"]
    assert traces[0]["tokens"][0] == "round_start"
    assert any(token.startswith("draw_0_") for token in traces[1]["tokens"])
    assert any(token.startswith("discard_0_") for token in traces[2]["tokens"])
    assert any(token.startswith("pingju_") for token in traces[3]["tokens"])
    assert any(token == "score_delta_3" for token in traces[3]["tokens"])
    assert tokenizer.tokens[-1] == "rank_3_4"


def test_trace_round_token_slices_matches_multi_ron_declined_pass_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}, 2: {"ron"}, 3: {"ron"}},
            trigger="discard",
        ),
    )
    round_data = [
        qipai_event(),
        {"zimo": {"l": 0, "p": "m1"}},
        {"dapai": {"l": 0, "p": "m1"}},
        {"hule": {"l": 1, "baojia": 0, "fenpei": [0, -1000, 1000, 0]}},
        {"hule": {"l": 2, "baojia": 0, "fenpei": [0, 0, 1000, -1000]}},
    ]

    tokenizer, traces = trace_round_token_slices(round_data)

    first_hule_tokens = traces[3]["tokens"]
    assert "pass_react_3_ron_voluntary" in first_hule_tokens
    assert first_hule_tokens.index("take_react_1_ron") < first_hule_tokens.index("pass_react_3_ron_voluntary")
    validate_token_stream(["game_start", *tokenizer.tokens, "game_end"])


def test_trace_round_token_slices_rejects_event_after_round_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}},
            trigger="discard",
        ),
    )
    round_data = [
        qipai_event(),
        {"zimo": {"l": 0, "p": "m1"}},
        {"dapai": {"l": 0, "p": "m1"}},
        {"hule": {"l": 1, "baojia": 0, "fenpei": [0, -1000, 1000, 0]}},
        pingju_event(),
    ]

    with pytest.raises(AssertionError, match="round already ended"):
        trace_round_token_slices(round_data)


@pytest.mark.slow
def test_dataset_validation_first_500_games() -> None:
    if not DATASET_2023.exists():
        pytest.skip(f"missing dataset: {DATASET_2023}")

    with zipfile.ZipFile(get_dataset_2023_sample_zip()) as zf:
        for name in zf.namelist():
            game = json.load(zf.open(name))
            tokens = TenhouTokenizer().tokenize_game(game)
            validate_token_stream(tokens)
            validate_score_rotation(game)
            for round_data in game.get("log", []):
                validate_round_stepwise(round_data)
