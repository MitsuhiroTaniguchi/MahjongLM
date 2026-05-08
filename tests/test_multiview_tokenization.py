from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tenhou_tokenizer import TOKEN_VIEW_COMPLETE, TOKEN_VIEW_OMNISCIENT, imperfect_view_token, tokenize_game_views
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event


ROOT = Path(__file__).resolve().parents[1]
CONVERT = ROOT / "scripts" / "paifu_scraping" / "convert.pl"
SANMA_RAW = ROOT / "tests" / "fixtures" / "tenhou" / "2014091101gm-00b9-0000-5ca6b487.txt"


def _convert_sanma_sample(*, include_shuffle_seed: bool = False) -> dict:
    if shutil.which("perl") is None:
        pytest.skip("perl is required for sanma conversion sample tests")
    raw_text = SANMA_RAW.read_text(encoding="utf-8")
    proc = subprocess.run(
        ["perl", "-T", str(CONVERT), str(SANMA_RAW)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    game = json.loads(proc.stdout.decode("utf-8"))
    if include_shuffle_seed:
        match = re.search(r'<SHUFFLE\s+[^>]*seed="mt19937ar-sha512-n288-base64,([^"]+)"', raw_text)
        assert match is not None
        game["_shuffle_seed"] = match.group(1)
    return game


def test_tokenize_game_views_emits_complete_plus_per_player_views() -> None:
    game = minimal_game([qipai_event(), pingju_event()])

    views = tokenize_game_views(game)

    assert len(views) == 5
    assert views[0].view_type == "complete"
    assert views[0].viewer_seat is None
    assert views[0].tokens[:4] == ["rule_player_4", TOKEN_VIEW_COMPLETE, "game_start", "round_start"]
    assert [view.viewer_seat for view in views[1:]] == [0, 1, 2, 3]
    assert [view.tokens[1] for view in views[1:]] == [
        imperfect_view_token(0),
        imperfect_view_token(1),
        imperfect_view_token(2),
        imperfect_view_token(3),
    ]


def test_imperfect_view_masks_other_players_haipai_tiles() -> None:
    game = minimal_game([qipai_event(), pingju_event()])

    view = tokenize_game_views(game)[1]

    assert "haipai_0" in view.tokens
    assert "m1" in view.tokens
    assert "haipai_1" not in view.tokens
    assert "haipai_2" not in view.tokens
    assert "haipai_3" not in view.tokens
    assert view.tokens.count("hidden_haipai_1") == 1
    assert view.tokens.count("hidden_haipai_2") == 1
    assert view.tokens.count("hidden_haipai_3") == 1


def test_imperfect_view_masks_other_players_draws_and_hides_their_self_options() -> None:
    game = minimal_game(
        [
            qipai_event(),
            {"zimo": {"l": 1, "p": "m1"}},
            {"dapai": {"l": 1, "p": "m1"}},
            pingju_event(),
        ]
    )

    view = tokenize_game_views(game)[1]

    assert "draw_1_hidden" in view.tokens
    assert "draw_1_m1" not in view.tokens
    assert all(not token.startswith("opt_self_1_") for token in view.tokens)
    assert all(not token.startswith("pass_self_1_") for token in view.tokens)


def test_imperfect_view_keeps_viewers_reaction_options_only() -> None:
    game = minimal_game(
        [
            qipai_event(
                hands=[
                    "m123456789p1234",
                    "m123456789p1234",
                    "m123456789p1234",
                    "m112345678p1234",
                ]
            ),
            {"zimo": {"l": 0, "p": "z1"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"fulou": {"l": 3, "m": "m1-11"}},
            {"dapai": {"l": 3, "p": "p1"}},
            pingju_event(),
        ]
    )

    view = tokenize_game_views(game)[4]

    assert any(token.startswith("opt_react_3_pon") for token in view.tokens)
    assert all(not token.startswith("opt_react_1_") for token in view.tokens)
    assert all(not token.startswith("opt_react_2_") for token in view.tokens)


def test_imperfect_view_hides_other_players_pass_tokens() -> None:
    game = minimal_game(
        [
            qipai_event(
                hands=[
                    "m123456789p1234",
                    "m123456789p1234",
                    "m123456789p1234",
                    "m123456789p1234",
                ]
            ),
            {"zimo": {"l": 0, "p": "z1"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"zimo": {"l": 1, "p": "z2"}},
            pingju_event(),
        ]
    )

    view = tokenize_game_views(game)[1]

    assert all(not token.startswith("pass_react_1_") for token in view.tokens)
    assert all(not token.startswith("pass_react_2_") for token in view.tokens)
    assert all(not token.startswith("pass_react_3_") for token in view.tokens)


def test_tokenize_game_views_emits_three_player_view_count() -> None:
    views = tokenize_game_views(_convert_sanma_sample())

    assert len(views) == 4
    assert views[0].view_type == "complete"
    assert [view.viewer_seat for view in views[1:]] == [0, 1, 2]


def test_omniscient_view_adds_wall_blocks_without_changing_existing_views() -> None:
    seeded_game = _convert_sanma_sample(include_shuffle_seed=True)
    seeded_views = tokenize_game_views(seeded_game)
    unseeded_views = tokenize_game_views(_convert_sanma_sample())

    assert [view.view_type for view in unseeded_views] == ["complete", "imperfect", "imperfect", "imperfect"]
    assert [view.view_type for view in seeded_views] == [
        "complete",
        "omniscient",
        "imperfect",
        "imperfect",
        "imperfect",
    ]
    assert seeded_views[0].tokens == unseeded_views[0].tokens
    assert seeded_views[2].tokens == unseeded_views[1].tokens

    omniscient = seeded_views[1]
    assert omniscient.tokens[1] == TOKEN_VIEW_OMNISCIENT
    round_start_index = omniscient.tokens.index("round_start")
    assert omniscient.tokens[round_start_index + 1] == "wall"
    first_wall = omniscient.tokens[round_start_index + 2 : round_start_index + 138]
    assert len(first_wall) == 136
    assert first_wall.count("m0") == 1
    assert first_wall.count("p0") == 1
    assert first_wall.count("s0") == 1


def test_imperfect_view_uses_qijia_relative_player_token() -> None:
    game = minimal_game([qipai_event(), pingju_event()])
    game["qijia"] = 2

    views = tokenize_game_views(game)

    assert [view.tokens[1] for view in views[1:]] == [
        imperfect_view_token(0),
        imperfect_view_token(1),
        imperfect_view_token(2),
        imperfect_view_token(3),
    ]


def test_imperfect_view_tracks_same_player_across_round_seat_rotation() -> None:
    game = {
        "qijia": 2,
        "log": [
            [
                qipai_event(jushu=0),
                pingju_event(),
            ],
            [
                qipai_event(jushu=1),
                {"zimo": {"l": 3, "p": "p1"}},
                {"dapai": {"l": 3, "p": "p1"}},
                pingju_event(),
            ],
        ],
    }

    views = tokenize_game_views(game)
    player0_view = views[1]

    assert player0_view.tokens[1] == imperfect_view_token(0)
    assert "draw_3_p1" in player0_view.tokens
    assert "draw_3_hidden" not in player0_view.tokens
