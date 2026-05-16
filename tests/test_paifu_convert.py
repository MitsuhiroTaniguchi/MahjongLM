from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from scripts.build_hf_from_raw_archives import resolve_perl_executable


ROOT = Path(__file__).resolve().parents[1]
CONVERT = ROOT / "scripts" / "paifu_scraping" / "convert.pl"
SANMA_RAW = ROOT / "tests" / "fixtures" / "tenhou" / "2014091101gm-00b9-0000-5ca6b487.txt"


def _convert(path: Path) -> dict:
    proc = subprocess.run(
        [resolve_perl_executable(None), "-T", str(CONVERT), str(path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return json.loads(proc.stdout.decode("utf-8"))


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


def test_convert_preserves_consecutive_dora_reveals_after_gangzimo() -> None:
    game = _convert_inline(
        '<mjloggm ver="2.3"><GO type="1" lobby="0"/>'
        '<UN n0="A" n1="B" n2="C" n3="D" dan="10,10,10,10" '
        'rate="1500,1500,1500,1500" sx="M,M,M,M"/>'
        '<TAIKYOKU oya="0"/>'
        '<INIT seed="0,0,0,0,0,16" ten="250,250,250,250" '
        'oya="0" hai0="0,1,2,3,4,5,6,7,8,9,10,11,12" '
        'hai1="13,14,15,16,17,18,19,20,21,22,23,24,25" '
        'hai2="26,27,28,29,30,31,32,33,34,35,36,37,38" '
        'hai3="39,40,41,42,43,44,45,46,47,48,49,50,51"/>'
        '<N who="0" m="11776"/>'
        '<T44/>'
        '<DORA hai="62"/>'
        '<DORA hai="51"/>'
        '<RYUUKYOKU type="yao9" ba="0,0" sc="250,0,250,0,250,0,250,0"/>'
        '</mjloggm>'
    )

    round_data = game["log"][0]
    gangzimo_idx = next(idx for idx, event in enumerate(round_data) if "gangzimo" in event)
    assert round_data[gangzimo_idx + 1 : gangzimo_idx + 3] == [
        {"kaigang": {"baopai": "p7"}},
        {"kaigang": {"baopai": "p4"}},
    ]


def test_convert_resets_pending_gang_between_rounds() -> None:
    game = _convert_inline(
        '<mjloggm ver="2.3"><GO type="1" lobby="0"/>'
        '<UN n0="A" n1="B" n2="C" n3="D" dan="10,10,10,10" '
        'rate="1500,1500,1500,1500" sx="M,M,M,M"/>'
        '<TAIKYOKU oya="0"/>'
        '<INIT seed="0,0,0,0,0,16" ten="250,250,250,250" '
        'oya="0" hai0="0,1,2,3,4,5,6,7,8,9,10,11,12" '
        'hai1="13,14,15,16,17,18,19,20,21,22,23,24,25" '
        'hai2="26,27,28,29,30,31,32,33,34,35,36,37,38" '
        'hai3="39,40,41,42,43,44,45,46,47,48,49,50,51"/>'
        '<N who="0" m="11776"/>'
        '<RYUUKYOKU type="yao9" ba="0,0" sc="250,0,250,0,250,0,250,0"/>'
        '<INIT seed="1,0,0,0,0,20" ten="250,250,250,250" '
        'oya="0" hai0="0,1,2,3,4,5,6,7,8,9,10,11,12" '
        'hai1="13,14,15,16,17,18,19,20,21,22,23,24,25" '
        'hai2="26,27,28,29,30,31,32,33,34,35,36,37,38" '
        'hai3="39,40,41,42,43,44,45,46,47,48,49,50,51"/>'
        '<T52/>'
        '<RYUUKYOKU type="yao9" ba="0,0" sc="250,0,250,0,250,0,250,0"/>'
        '</mjloggm>'
    )

    assert game["log"][1][1] == {"zimo": {"l": 0, "p": "p0"}}
