from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERT = ROOT / "tests" / "fixtures" / "tenhou" / "convert.pl"
SANMA_RAW = ROOT / "tests" / "fixtures" / "tenhou" / "2014091101gm-00b9-0000-5ca6b487.txt"


def _convert(path: Path) -> dict:
    proc = subprocess.run(
        ["perl", "-T", str(CONVERT), str(path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def _convert_inline(raw: str) -> dict:
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write(raw)
        path = Path(f.name)
    return _convert(path)


def test_convert_go_type_emits_tonpu_title() -> None:
    game = _convert_inline(
        '<mjloggm ver="2.3"><GO type="129" lobby="0"/>'
        '<UN n0="A" n1="B" n2="C" n3="D" dan="10,10,10,10" '
        'rate="1500,1500,1500,1500" sx="M,M,M,M"/>'
        '<TAIKYOKU oya="0"/></mjloggm>'
    )
    assert game["title"].startswith("四上東")


def test_convert_go_type_emits_hanchan_title() -> None:
    game = _convert_inline(
        '<mjloggm ver="2.3"><GO type="137" lobby="0"/>'
        '<UN n0="A" n1="B" n2="C" n3="D" dan="10,10,10,10" '
        'rate="1500,1500,1500,1500" sx="M,M,M,M"/>'
        '<TAIKYOKU oya="0"/></mjloggm>'
    )
    assert game["title"].startswith("四上南")


def test_convert_sanma_normalizes_to_three_players() -> None:
    game = _convert(SANMA_RAW)

    assert game["title"].startswith("三")
    assert len(game["player"]) == 3
    assert len(game["defen"]) == 3
    assert len(game["point"]) == 3
    assert len(game["rank"]) == 3

    for round_data in game["log"]:
        qipai = round_data[0]["qipai"]
        assert len(qipai["defen"]) == 3
        assert len(qipai["shoupai"]) == 3
        for event in round_data:
            if "hule" in event:
                assert len(event["hule"]["fenpei"]) == 3
            if "pingju" in event:
                assert len(event["pingju"]["fenpei"]) == 3
                assert len(event["pingju"]["shoupai"]) == 3


def test_convert_preserves_penuki_state_in_pingju_shoupai() -> None:
    game = _convert(SANMA_RAW)

    saw_penuki = False
    saw_pingju_with_penuki = False
    for round_data in game["log"]:
        for event in round_data:
            if "penuki" in event:
                saw_penuki = True
            if saw_penuki and "pingju" in event:
                assert any("b" in (shoupai or "") for shoupai in event["pingju"]["shoupai"])
                saw_pingju_with_penuki = True
                break
        if saw_pingju_with_penuki:
            break

    assert saw_penuki
    assert saw_pingju_with_penuki
