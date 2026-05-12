from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pymahjong as pm  # type: ignore

PM_XIANGTING = pm.Xiangting()
PM_FASTAPI_AVAILABLE = all(
    hasattr(pm, name) for name in ("wait_mask", "has_riichi_discard", "has_hupai")
)
PM_MULTI_HUPAI_AVAILABLE = hasattr(pm, "has_hupai_multi")
PM_EVALUATE_DRAW_AVAILABLE = hasattr(pm, "evaluate_draw")
PM_THREE_PLAYER_API_AVAILABLE = hasattr(pm, "SELF_OPT_PENUKI")
PM_STATELESS_SIMULATION_API_AVAILABLE = all(
    hasattr(pm, name)
    for name in ("compute_self_option_mask", "compute_reaction_option_masks", "compute_rob_kan_option_masks")
)
PM_SHOUPAI_SIMULATION_API_AVAILABLE = all(
    hasattr(pm, name)
    for name in (
        "compute_self_option_mask_shoupai",
        "compute_reaction_option_masks_shoupai",
        "compute_rob_kan_option_masks_shoupai",
    )
)
PM_SIMULATION_API_AVAILABLE = (
    PM_STATELESS_SIMULATION_API_AVAILABLE or PM_SHOUPAI_SIMULATION_API_AVAILABLE
)

SUIT_BASE = {"m": 0, "p": 9, "s": 18, "z": 27}
INDEX_TO_TILE = [
    *(f"m{i}" for i in range(1, 10)),
    *(f"p{i}" for i in range(1, 10)),
    *(f"s{i}" for i in range(1, 10)),
    *(f"z{i}" for i in range(1, 8)),
]
YAOCHU_INDICES: Set[int] = {0, 8, 9, 17, 18, 26, *range(27, 34)}
MELD_TYPE_TO_PM_CODE = {
    "chi": 0,
    "pon": 1,
    "minkan": 2,
    "ankan": 3,
}
TENBO_UNITS: Tuple[Tuple[int, str], ...] = (
    (10000, "TENBO_10000"),
    (9000, "TENBO_9000"),
    (8000, "TENBO_8000"),
    (7000, "TENBO_7000"),
    (6000, "TENBO_6000"),
    (5000, "TENBO_5000"),
    (4000, "TENBO_4000"),
    (3000, "TENBO_3000"),
    (2000, "TENBO_2000"),
    (1000, "TENBO_1000"),
    (900, "TENBO_900"),
    (800, "TENBO_800"),
    (700, "TENBO_700"),
    (600, "TENBO_600"),
    (500, "TENBO_500"),
    (400, "TENBO_400"),
    (300, "TENBO_300"),
    (200, "TENBO_200"),
    (100, "TENBO_100"),
)
HUPAI_TOKEN_BY_NAME = {
    "門前清自摸和": "yaku_menzen_tsumo",
    "立直": "yaku_riichi",
    "一発": "yaku_ippatsu",
    "槍槓": "yaku_chankan",
    "嶺上開花": "yaku_rinshan",
    "海底摸月": "yaku_haitei",
    "河底撈魚": "yaku_houtei",
    "平和": "yaku_pinfu",
    "断幺九": "yaku_tanyao",
    "一盃口": "yaku_iipeikou",
    "自風 東": "yaku_jikaze_ton",
    "自風 南": "yaku_jikaze_nan",
    "自風 西": "yaku_jikaze_shaa",
    "自風 北": "yaku_jikaze_pei",
    "場風 東": "yaku_bakaze_ton",
    "場風 南": "yaku_bakaze_nan",
    "場風 西": "yaku_bakaze_shaa",
    "場風 北": "yaku_bakaze_pei",
    "役牌 白": "yaku_haku",
    "役牌 發": "yaku_hatsu",
    "役牌 中": "yaku_chun",
    "両立直": "yaku_double_riichi",
    "七対子": "yaku_chiitoitsu",
    "混全帯幺九": "yaku_chanta",
    "一気通貫": "yaku_ittsu",
    "三色同順": "yaku_sanshoku_doujun",
    "三色同刻": "yaku_sanshoku_doukou",
    "三槓子": "yaku_sankantsu",
    "対々和": "yaku_toitoi",
    "三暗刻": "yaku_sanankou",
    "小三元": "yaku_shousangen",
    "混老頭": "yaku_honroutou",
    "二盃口": "yaku_ryanpeikou",
    "純全帯幺九": "yaku_junchan",
    "混一色": "yaku_honitsu",
    "清一色": "yaku_chinitsu",
    "天和": "yaku_tenhou",
    "地和": "yaku_chiihou",
    "大三元": "yaku_daisangen",
    "四暗刻": "yaku_suuankou",
    "四暗刻単騎": "yaku_suuankou_tanki",
    "字一色": "yaku_tsuuiisou",
    "緑一色": "yaku_ryuuiisou",
    "清老頭": "yaku_chinroutou",
    "九蓮宝燈": "yaku_chuuren_poutou",
    "純正九蓮宝燈": "yaku_junsei_chuuren_poutou",
    "国士無双": "yaku_kokushi_musou",
    "国士無双１３面": "yaku_kokushi_musou_13_wait",
    "大四喜": "yaku_daisuushi",
    "小四喜": "yaku_shousuushi",
    "四槓子": "yaku_suukantsu",
    "ドラ": "yaku_dora",
    "裏ドラ": "yaku_ura_dora",
    "赤ドラ": "yaku_aka_dora",
}
DORA_HUPAI_NAMES = frozenset({"ドラ", "裏ドラ", "赤ドラ"})
RIICHI_HUPAI_NAMES = frozenset({"立直", "両立直"})
SELF_OPT_TSUMO = int(getattr(pm, "SELF_OPT_TSUMO", 1 << 0))
SELF_OPT_RIICHI = int(getattr(pm, "SELF_OPT_RIICHI", 1 << 1))
SELF_OPT_ANKAN = int(getattr(pm, "SELF_OPT_ANKAN", 1 << 2))
SELF_OPT_KAKAN = int(getattr(pm, "SELF_OPT_KAKAN", 1 << 3))
SELF_OPT_KYUSHUKYUHAI = int(getattr(pm, "SELF_OPT_KYUSHUKYUHAI", 1 << 4))
SELF_OPT_PENUKI = int(getattr(pm, "SELF_OPT_PENUKI", 1 << 5))
REACT_OPT_RON = int(getattr(pm, "REACT_OPT_RON", 1 << 0))
REACT_OPT_CHI = int(getattr(pm, "REACT_OPT_CHI", 1 << 1))
REACT_OPT_PON = int(getattr(pm, "REACT_OPT_PON", 1 << 2))
REACT_OPT_MINKAN = int(getattr(pm, "REACT_OPT_MINKAN", 1 << 3))


class TokenizeError(RuntimeError):
    pass


def _strip_tile_suffix(tile: str) -> str:
    return tile.replace("*", "").replace("_", "")


def _norm_digit(digit: str) -> str:
    return "5" if digit == "0" else digit


def tile_to_index(tile: str) -> int:
    if len(tile) != 2:
        raise TokenizeError(f"invalid tile format: {tile}")
    suit, digit = tile[0], tile[1]
    if suit not in SUIT_BASE or not digit.isdigit():
        raise TokenizeError(f"invalid tile format: {tile}")
    if suit == "z":
        number = int(digit)
        if number < 1 or number > 7:
            raise TokenizeError(f"invalid honor tile: {tile}")
    else:
        number = int(_norm_digit(digit))
        if number < 1 or number > 9:
            raise TokenizeError(f"invalid suited tile: {tile}")
    return SUIT_BASE[suit] + number - 1


def index_to_tile(index: int) -> str:
    return INDEX_TO_TILE[index]


def token_tile(tile: str) -> str:
    if len(tile) != 2:
        raise TokenizeError(f"invalid tile format: {tile}")
    suit, digit = tile[0], tile[1]
    if suit not in SUIT_BASE or not digit.isdigit():
        raise TokenizeError(f"invalid tile format: {tile}")
    if suit == "z":
        number = int(digit)
        if number < 1 or number > 7:
            raise TokenizeError(f"invalid honor tile: {tile}")
        return f"{suit}{number}"
    number = int(digit)
    if number == 0:
        return f"{suit}0"
    if number < 1 or number > 9:
        raise TokenizeError(f"invalid suited tile: {tile}")
    return f"{suit}{number}"


def _parse_tiles(text: str, *, stop_at_comma: bool, context: str) -> List[str]:
    tiles: List[str] = []
    suit: Optional[str] = None
    for ch in text:
        if ch in SUIT_BASE:
            suit = ch
            continue
        if stop_at_comma and ch == ",":
            break
        if ch in "-=+,":
            continue
        if ch.isdigit():
            if suit is None:
                raise TokenizeError(f"digit without suit in {context}: {text}")
            tiles.append(token_tile(f"{suit}{ch}"))
            continue
        if ch.isspace():
            continue
        raise TokenizeError(f"unexpected character in {context}: {text}")
    return tiles


def parse_hand_counts(hand: str) -> List[int]:
    counts = [0] * 34
    for tile in _parse_tiles(hand, stop_at_comma=True, context="hand"):
        counts[tile_to_index(tile)] += 1
    return counts


def parse_hand_red_fives(hand: str) -> Dict[str, int]:
    red_fives = {"m": 0, "p": 0, "s": 0}
    for tile in _parse_tiles(hand, stop_at_comma=True, context="hand"):
        if tile[1] == "0" and tile[0] in red_fives:
            red_fives[tile[0]] += 1
    return red_fives


def parse_meld_tiles(meld: str) -> List[int]:
    tiles = [tile_to_index(tile) for tile in _parse_tiles(meld, stop_at_comma=False, context="meld")]
    if not tiles:
        raise TokenizeError(f"empty meld parse: {meld}")
    return tiles


def parse_meld_token_tiles_and_called(meld: str) -> Tuple[List[str], Optional[int]]:
    tiles: List[str] = []
    called_index: Optional[int] = None
    suit: Optional[str] = None
    mark_next = False

    for ch in meld:
        if ch in SUIT_BASE:
            suit = ch
            continue
        if ch == ",":
            continue
        if ch in "-=+":
            mark_next = True
            continue
        if ch.isdigit():
            if suit is None:
                raise TokenizeError(f"digit without suit in meld: {meld}")
            if mark_next:
                called_index = len(tiles)
                mark_next = False
            tiles.append(token_tile(f"{suit}{ch}"))
            continue

    if not tiles:
        raise TokenizeError(f"empty meld parse: {meld}")
    # Some converter variants may place the marker after the called digit.
    if mark_next and called_index is None:
        called_index = len(tiles) - 1
    return tiles, called_index


def token_tile_sort_key(tile: str) -> Tuple[int, int]:
    return (tile_to_index(tile), 0 if tile[1] == "0" else 1)


def encode_tenbo_tokens(value: int) -> List[str]:
    if value == 0:
        return ["TENBO_ZERO"]
    sign = "TENBO_PLUS" if value > 0 else "TENBO_MINUS"
    remaining = abs(value)
    if remaining % 100 != 0:
        raise TokenizeError(f"score value must be a multiple of 100: {value}")
    tokens: List[str] = [sign]
    for unit_value, unit_token in TENBO_UNITS:
        count, remaining = divmod(remaining, unit_value)
        if count:
            tokens.extend([unit_token] * count)
    if remaining != 0:
        raise TokenizeError(f"score value cannot be represented by tenbo units: {value}")
    return tokens


def _opened_hand_tokens(seat: int, hand_text: str) -> List[str]:
    tiles = sorted(_parse_tiles(hand_text, stop_at_comma=True, context="opened_hand"), key=token_tile_sort_key)
    if not tiles:
        return []
    return [f"opened_hand_{seat}", *tiles]


def _hule_winning_tile_token(hand_text: str) -> Optional[str]:
    tiles = _parse_tiles(hand_text, stop_at_comma=True, context="hule_opened_hand")
    if not tiles:
        return None
    # Tenhou hule.shoupai includes the winning tile as the last concealed tile.
    return tiles[-1]


def _opened_hule_hand_tokens(seat: int, hand_text: str) -> List[str]:
    tiles = _parse_tiles(hand_text, stop_at_comma=True, context="hule_opened_hand")
    if not tiles:
        return []
    # Tenhou hule.shoupai includes the winning tile as the last concealed tile.
    concealed_tiles = sorted(tiles[:-1], key=token_tile_sort_key)
    if not concealed_tiles:
        return []
    return [f"opened_hand_{seat}", *concealed_tiles]


def classify_fulou(meld_tiles: List[int]) -> str:
    if meld_tiles[0] != meld_tiles[-1]:
        return "chi"
    if len(meld_tiles) == 3:
        return "pon"
    return "minkan"


def classify_gang(meld_text: str) -> str:
    if any(c in meld_text for c in "-=+"):
        return "kakan"
    return "ankan"


def _remove_tiles(concealed: List[int], tile: int, n: int) -> None:
    if concealed[tile] < n:
        raise TokenizeError(f"cannot remove tile {index_to_tile(tile)} x{n}")
    concealed[tile] -= n


def _has_legal_post_call_discard(concealed: List[int], forbidden_tiles: Set[int]) -> bool:
    return any(count > 0 and tile_idx not in forbidden_tiles for tile_idx, count in enumerate(concealed))


def _make_pm_shoupai(counts: List[int], melds: List[Tuple[str, int]]):
    fulu = []
    for meld_type, pai_34 in melds:
        if meld_type == "chi":
            fulu.append(pm.Mianzi(pm.FuluType.chi, pai_34))
        elif meld_type == "pon":
            fulu.append(pm.Mianzi(pm.FuluType.peng, pai_34))
        elif meld_type == "minkan":
            fulu.append(pm.Mianzi(pm.FuluType.minggang, pai_34))
        elif meld_type == "ankan":
            fulu.append(pm.Mianzi(pm.FuluType.angang, pai_34))
        else:
            raise TokenizeError(f"unsupported meld type for pymahjong: {meld_type}")
    if fulu:
        return pm.Shoupai(tuple(counts), fulu)
    return pm.Shoupai(tuple(counts))


def _encode_pm_melds(melds: List[Tuple[str, int]]) -> List[Tuple[int, int]]:
    return [(MELD_TYPE_TO_PM_CODE[meld_type], pai_34) for meld_type, pai_34 in melds]


def _pm_xiangting(counts: List[int], meld_count: int, three_player: bool = False) -> int:
    x, _mode, _disc, _wait = PM_XIANGTING.calculate(tuple(counts), 4 - meld_count, 7, False, three_player)
    return int(x)


def _pm_wait_mask(counts: List[int], meld_count: int, three_player: bool = False) -> int:
    if PM_FASTAPI_AVAILABLE:
        if PM_THREE_PLAYER_API_AVAILABLE and three_player:
            return int(pm.wait_mask(tuple(counts), int(meld_count), True))
        return int(pm.wait_mask(tuple(counts), int(meld_count)))
    x, _mode, _disc, wait = PM_XIANGTING.calculate(tuple(counts), 4 - meld_count, 7, False, three_player)
    if int(x) != 0:
        return 0
    return int(wait)


def _pm_wait_tiles(counts: List[int], meld_count: int, three_player: bool = False) -> Set[int]:
    wait_mask = _pm_wait_mask(counts, meld_count, three_player)
    if wait_mask == 0:
        return set()
    return {i for i in range(34) if (wait_mask >> i) & 1}


def _furiten_mask(tiles: Set[int]) -> int:
    mask = 0
    for tile in tiles:
        mask |= 1 << tile
    return mask


def _is_kokushi_agari_shape(counts: List[int], meld_count: int) -> bool:
    if meld_count != 0:
        return False
    if sum(counts) != 14:
        return False
    pair_found = False
    for i, c in enumerate(counts):
        if i in YAOCHU_INDICES:
            if c == 0:
                return False
            if c == 1:
                continue
            if c == 2 and not pair_found:
                pair_found = True
                continue
            return False
        if c != 0:
            return False
    return pair_found


def _pm_has_riichi_discard(counts: List[int], meld_count: int, three_player: bool = False) -> bool:
    if PM_FASTAPI_AVAILABLE:
        if PM_THREE_PLAYER_API_AVAILABLE and three_player:
            return bool(pm.has_riichi_discard(tuple(counts), int(meld_count), True))
        return bool(pm.has_riichi_discard(tuple(counts), int(meld_count)))

    base = list(counts)
    for i in range(34):
        if three_player and 0 < i < 8:
            continue
        if base[i] == 0:
            continue
        base[i] -= 1
        if _pm_xiangting(base, meld_count, three_player) == 0:
            return True
        base[i] += 1
    return False


def _pm_has_hupai(
    counts: List[int],
    melds: List[Tuple[str, int]],
    encoded_melds: Optional[List[Tuple[int, int]]],
    win_tile: int,
    is_tsumo: bool,
    is_menqian: bool,
    is_riichi: bool,
    zhuangfeng: int,
    lunban: int,
    is_haidi: bool = False,
    is_lingshang: bool = False,
    is_qianggang: bool = False,
    three_player: bool = False,
) -> bool:
    pm_melds = encoded_melds if encoded_melds is not None else _encode_pm_melds(melds)
    if PM_FASTAPI_AVAILABLE:
        if PM_THREE_PLAYER_API_AVAILABLE and three_player:
            try:
                return bool(
                    pm.has_hupai(
                        tuple(counts),
                        pm_melds,
                        int(win_tile),
                        bool(is_tsumo),
                        bool(is_menqian),
                        bool(is_riichi),
                        int(zhuangfeng),
                        int(lunban),
                        bool(is_haidi),
                        bool(is_lingshang),
                        bool(is_qianggang),
                        bool(three_player),
                    )
                )
            except TypeError:
                pass
        return bool(
            pm.has_hupai(
                tuple(counts),
                pm_melds,
                int(win_tile),
                bool(is_tsumo),
                bool(is_menqian),
                bool(is_riichi),
                int(zhuangfeng),
                int(lunban),
                bool(is_haidi),
                bool(is_lingshang),
                bool(is_qianggang),
            )
        )

    shoupai = _make_pm_shoupai(counts, melds)
    if PM_THREE_PLAYER_API_AVAILABLE and three_player:
        try:
            option = pm.HuleOption(int(zhuangfeng), int(lunban), bool(three_player))
        except TypeError:
            option = pm.HuleOption(int(zhuangfeng), int(lunban))
    else:
        option = pm.HuleOption(int(zhuangfeng), int(lunban))
    option.is_menqian = bool(is_menqian)
    option.is_lizhi = bool(is_riichi)
    option.is_haidi = bool(is_haidi)
    option.is_lingshang = bool(is_lingshang)
    option.is_qianggang = bool(is_qianggang)
    if three_player and hasattr(option, "is_three_player"):
        option.is_three_player = True
    action_type = pm.ActionType.zimohu if is_tsumo else pm.ActionType.ronghu
    action = pm.Action(action_type, int(win_tile))
    hule = pm.Hule(shoupai, action, option)
    return bool(hule.has_hupai)


def _pm_has_hupai_multi(
    cases: List[
        Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int, bool, bool, bool, bool]
    ],
) -> List[bool]:
    if not cases:
        return []
    if PM_MULTI_HUPAI_AVAILABLE and not any(case[-1] for case in cases):
        encoded_cases_with_context: List[
            Tuple[Tuple[int, ...], List[Tuple[int, int]], int, bool, bool, bool, int, int, bool, bool, bool, bool]
        ] = []
        for (
            counts,
            melds,
            win_tile,
            is_tsumo,
            is_menqian,
            is_riichi,
            zhuangfeng,
            lunban,
            is_haidi,
            is_lingshang,
            is_qianggang,
            three_player,
        ) in cases:
            encoded_melds = [(MELD_TYPE_TO_PM_CODE[mtype], pai_34) for mtype, pai_34 in melds]
            encoded_cases_with_context.append(
                (
                    tuple(counts),
                    encoded_melds,
                    int(win_tile),
                    bool(is_tsumo),
                    bool(is_menqian),
                    bool(is_riichi),
                    int(zhuangfeng),
                    int(lunban),
                    bool(is_haidi),
                    bool(is_lingshang),
                    bool(is_qianggang),
                )
            )
        return [bool(x) for x in pm.has_hupai_multi(encoded_cases_with_context)]

    return [
        _pm_has_hupai(
            counts=counts,
            melds=melds,
            encoded_melds=None,
            win_tile=win_tile,
            is_tsumo=is_tsumo,
            is_menqian=is_menqian,
            is_riichi=is_riichi,
            zhuangfeng=zhuangfeng,
            lunban=lunban,
            is_haidi=is_haidi,
            is_lingshang=is_lingshang,
            is_qianggang=is_qianggang,
            three_player=three_player,
        )
        for (
            counts,
            melds,
            win_tile,
            is_tsumo,
            is_menqian,
            is_riichi,
            zhuangfeng,
            lunban,
            is_haidi,
            is_lingshang,
            is_qianggang,
            three_player,
        ) in cases
    ]


def _pm_evaluate_draw(
    counts: List[int],
    melds: List[Tuple[str, int]],
    encoded_melds: Optional[List[Tuple[int, int]]],
    win_tile: int,
    is_menqian: bool,
    is_riichi: bool,
    zhuangfeng: int,
    lunban: int,
    closed_kans: int,
    check_riichi_discard: bool,
    is_haidi: bool = False,
    is_lingshang: bool = False,
    three_player: bool = False,
) -> Tuple[bool, bool]:
    pm_melds = encoded_melds if encoded_melds is not None else _encode_pm_melds(melds)
    if PM_EVALUATE_DRAW_AVAILABLE:
        if PM_THREE_PLAYER_API_AVAILABLE and three_player:
            can_tsumo, can_riichi_discard = pm.evaluate_draw(
                tuple(counts),
                pm_melds,
                int(win_tile),
                bool(is_menqian),
                bool(is_riichi),
                int(zhuangfeng),
                int(lunban),
                int(closed_kans),
                bool(check_riichi_discard),
                bool(is_haidi),
                bool(is_lingshang),
                bool(three_player),
            )
        else:
            can_tsumo, can_riichi_discard = pm.evaluate_draw(
                tuple(counts),
                pm_melds,
                int(win_tile),
                bool(is_menqian),
                bool(is_riichi),
                int(zhuangfeng),
                int(lunban),
                int(closed_kans),
                bool(check_riichi_discard),
                bool(is_haidi),
                bool(is_lingshang),
            )
        return bool(can_tsumo), bool(can_riichi_discard)

    can_tsumo = _pm_has_hupai(
        counts=counts,
        melds=melds,
        encoded_melds=pm_melds,
        win_tile=win_tile,
        is_tsumo=True,
        is_menqian=is_menqian,
        is_riichi=is_riichi,
        zhuangfeng=zhuangfeng,
        lunban=lunban,
        is_haidi=is_haidi,
        is_lingshang=is_lingshang,
        three_player=three_player,
    )
    can_riichi_discard = (
        _pm_has_riichi_discard(counts, closed_kans, three_player) if check_riichi_discard else False
    )
    return can_tsumo, can_riichi_discard


@dataclass
class PlayerState:
    concealed: List[int]
    score: int
    red_fives: Dict[str, int] = field(default_factory=lambda: {"m": 0, "p": 0, "s": 0})
    sim_shoupai: Optional[pm.Shoupai] = None
    last_draw_tile: Optional[str] = None
    is_riichi: bool = False
    open_melds: int = 0
    closed_kans: int = 0
    open_pons: Dict[int, int] = field(default_factory=dict)
    open_pons_red: Dict[int, int] = field(default_factory=dict)
    melds: List[Tuple[str, int]] = field(default_factory=list)
    encoded_melds_cache: Optional[List[Tuple[int, int]]] = None
    furiten_tiles: Set[int] = field(default_factory=set)
    temporary_furiten: bool = False
    riichi_furiten: bool = False
    is_first_turn: bool = True
    wait_mask_cache: Optional[int] = None

    @property
    def meld_count(self) -> int:
        return self.open_melds + self.closed_kans


@dataclass
class SelfDecision:
    actor: int
    options: Set[str]
    option_tiles: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ReactionDecision:
    discarder: int
    discard_tile: int
    options_by_player: Dict[int, Set[str]]
    trigger: str = "discard"
    chosen: Dict[int, str] = field(default_factory=dict)
    emitted_chosen: Set[int] = field(default_factory=set)


@dataclass(frozen=True)
class EventTrace:
    round_index: int
    event_key: str
    start: int
    end: int


class TenhouTokenizer:
    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.event_traces: List[EventTrace] = []
        self.players: List[PlayerState] = []
        self.seat_count = 4
        self.pending_self: Optional[SelfDecision] = None
        self.pending_reaction: Optional[ReactionDecision] = None
        self.pending_multi_ron_baojia: Optional[int] = None
        self.pending_multi_ron_remaining: Set[int] = set()
        self.pending_multi_ron_deltas: Optional[List[int]] = None
        self.pending_riichi_actor: Optional[int] = None
        self.round_index = 0
        self.live_draws_left = 55 if self.seat_count == 3 else 70
        self.bakaze = 0
        self.kyoku = 0
        self.dealer_seat = 0
        self.initial_qijia = 0
        self.has_initial_qijia = False
        self.first_turn_open_calls_seen = False
        self.last_draw_was_gangzimo = False
        self.expected_draw_actor: Optional[int] = None
        self.expected_discard_actor: Optional[int] = None
        self.awaiting_kaigang = 0
        self.pending_kan_dora_modes: List[str] = []
        self.pending_dora_tiles: List[str] = []
        self.pending_dead_wall_draw = False

    def tokenize_game(self, game: dict) -> List[str]:
        if not isinstance(game, dict):
            raise TokenizeError("game must be a dict")
        log = game.get("log", [])
        if not isinstance(log, list):
            raise TokenizeError("game log must be a list")
        if not log:
            raise TokenizeError("game log cannot be empty")
        self.seat_count = self._infer_game_seat_count(game)
        self.tokens = [*self._build_game_rule_block(game), "game_start"]
        self.event_traces = []
        self.pending_self = None
        self.pending_reaction = None
        self.pending_multi_ron_baojia = None
        self.pending_multi_ron_remaining = set()
        self.pending_multi_ron_deltas = None
        self.pending_riichi_actor = None
        self.round_index = 0
        self.expected_draw_actor = None
        self.expected_discard_actor = None
        self.awaiting_kaigang = 0
        self.pending_kan_dora_modes = []
        self.pending_dora_tiles = []
        self.pending_dead_wall_draw = False
        self.initial_qijia = 0
        self.has_initial_qijia = False
        if "qijia" in game:
            self.initial_qijia = self._require_seat(game["qijia"], field="game.qijia")
            self.has_initial_qijia = True
        for round_index, round_data in enumerate(log):
            self._process_round(round_data)
            is_last_round = round_index == len(log) - 1
            self.tokens.append("round_end")
            if is_last_round:
                self.tokens.append("game_end")
            if is_last_round and ("defen" in game or "rank" in game):
                if "defen" in game:
                    self._require_seat_list(game["defen"], field="game.defen")
                    for seat, score in enumerate(game["defen"]):
                        self._require_score(score, field=f"game.defen[{seat}]")
                final_scores = self._current_game_order_scores()
                final_ranks = self._compute_final_rank_places(final_scores)
                if "rank" in game:
                    for seat, rank in enumerate(self._require_seat_list(game["rank"], field="game.rank")):
                        self._require_rank_place(rank, field=f"game.rank[{seat}]")
                self.tokens.extend(self._build_final_score_block(final_scores))
                self.tokens.extend(self._build_final_rank_block(final_ranks))
        self._flush_pending()
        return self.tokens

    def _require_round_initialized(self) -> None:
        if len(self.players) != self.seat_count:
            raise TokenizeError("round state is not initialized")

    def _require_int(self, value: object, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TokenizeError(f"{field} must be an integer")
        return value

    def _require_seat(self, seat: object, *, field: str) -> int:
        seat_int = self._require_int(seat, field=field)
        if seat_int < 0 or seat_int >= self.seat_count:
            raise TokenizeError(f"invalid seat for {field}: {seat}")
        return seat_int

    def _require_seat_list(self, values: object, *, field: str) -> List[object]:
        if not isinstance(values, list) or len(values) != self.seat_count:
            raise TokenizeError(f"{field} must be a list of length {self.seat_count}")
        return values

    def _require_dict(self, value: object, *, field: str) -> dict:
        if not isinstance(value, dict):
            raise TokenizeError(f"{field} must be a dict")
        return value

    def _require_str(self, value: object, *, field: str) -> str:
        if not isinstance(value, str):
            raise TokenizeError(f"{field} must be a string")
        return value

    def _require_non_negative_int(self, value: object, *, field: str) -> int:
        value_int = self._require_int(value, field=field)
        if value_int < 0:
            raise TokenizeError(f"{field} must be a non-negative integer")
        return value_int

    def _require_score(self, value: object, *, field: str) -> int:
        value = self._require_int(value, field=field)
        if value % 100 != 0:
            raise TokenizeError(f"score value must be a multiple of 100: {value}")
        return value

    def _require_rank_place(self, value: object, *, field: str) -> int:
        value_int = self._require_int(value, field=field)
        if value_int < 1 or value_int > self.seat_count:
            raise TokenizeError(f"{field} must be an integer between 1 and {self.seat_count}")
        return value_int

    def _infer_game_seat_count(self, game: dict) -> int:
        title = game.get("title")
        if isinstance(title, str):
            if "三" in title:
                return 3
            if "四" in title:
                return 4
        for key in ("defen", "rank", "player", "point"):
            value = game.get(key)
            if isinstance(value, list) and len(value) in {3, 4}:
                return len(value)
        log = game.get("log")
        if isinstance(log, list):
            for round_data in log:
                if not isinstance(round_data, list):
                    continue
                for event in round_data:
                    if not isinstance(event, dict):
                        continue
                    for payload_key in ("qipai", "hule", "pingju"):
                        payload = event.get(payload_key)
                        if not isinstance(payload, dict):
                            continue
                        for list_key in ("shoupai", "defen", "fenpei"):
                            values = payload.get(list_key)
                            if isinstance(values, list) and len(values) in {3, 4}:
                                return len(values)
        return 4

    def _process_round(self, round_data: list) -> None:
        if not isinstance(round_data, list):
            raise TokenizeError("round data must be a list")
        if not round_data:
            raise TokenizeError("round data cannot be empty")
        self.pending_self = None
        self.pending_reaction = None
        self.pending_multi_ron_baojia = None
        self.pending_multi_ron_remaining = set()
        self.pending_multi_ron_deltas = None
        self.pending_riichi_actor = None
        self.expected_draw_actor = None
        self.expected_discard_actor = None
        self.awaiting_kaigang = 0
        self.pending_kan_dora_modes = []
        self.pending_dora_tiles = []
        saw_qipai = False
        round_ended = False
        skipped_event_indices: Set[int] = set()

        for event_index, event in enumerate(round_data):
            if event_index in skipped_event_indices:
                continue
            if not isinstance(event, dict):
                raise TokenizeError("round event must be a dict")
            if not event:
                raise TokenizeError("round event cannot be empty")
            key, value = next(iter(event.items()))

            if not saw_qipai:
                if key != "qipai":
                    raise TokenizeError("round must start with qipai")
                saw_qipai = True
            elif key == "qipai":
                raise TokenizeError("round cannot contain multiple qipai")
            elif round_ended:
                is_multi_ron_continuation = (
                    key == "hule"
                    and isinstance(value, dict)
                    and (
                        (
                            self.pending_reaction is not None
                            and value.get("baojia") == self.pending_reaction.discarder
                        )
                        or (
                            self.pending_multi_ron_baojia is not None
                            and value.get("baojia") == self.pending_multi_ron_baojia
                        )
                    )
                )
                if not is_multi_ron_continuation:
                    raise TokenizeError("round already ended")

            if self.pending_reaction and not self._is_reaction_continuation(key, value):
                if key != "pingju":
                    self._finalize_reaction(close_reason="voluntary")

            if self.pending_self and not self._is_self_resolution(key, value):
                self._finalize_self(set())

            before_token_count = len(self.tokens)
            if key == "qipai":
                self._on_qipai(value)
            elif key == "zimo":
                is_dead_wall_draw = self.pending_dead_wall_draw
                self.pending_dead_wall_draw = False
                self._on_draw(
                    value,
                    is_gangzimo=is_dead_wall_draw,
                    is_replacement_draw=is_dead_wall_draw,
                )
            elif key == "gangzimo":
                reveal_count = self._kaigang_lookahead_count_before_gangzimo()
                if reveal_count:
                    self._consume_following_kaigang_events(round_data, event_index, reveal_count, skipped_event_indices)
                self.pending_dead_wall_draw = False
                self._on_draw(value, is_gangzimo=True, is_replacement_draw=False)
            elif key == "dapai":
                self._consume_following_kaigang_events(round_data, event_index, None, skipped_event_indices)
                self._on_discard(value)
            elif key == "fulou":
                self._on_fulou(value)
            elif key == "gang":
                self._on_gang(value)
            elif key == "kaigang":
                self._on_kaigang(value)
            elif key == "penuki":
                self._on_penuki(value)
            elif key == "hule":
                remaining_ron_winners: Set[int] | None = None
                more_ron_expected = False
                if isinstance(value, dict) and value.get("baojia") is not None:
                    remaining_ron_winners = set()
                    lookahead_index = event_index
                    expected_baojia = value.get("baojia")
                    while lookahead_index < len(round_data):
                        lookahead_event = round_data[lookahead_index]
                        if not isinstance(lookahead_event, dict) or "hule" not in lookahead_event:
                            break
                        lookahead_hule = lookahead_event["hule"]
                        if not isinstance(lookahead_hule, dict) or lookahead_hule.get("baojia") != expected_baojia:
                            break
                        winner = lookahead_hule.get("l")
                        if isinstance(winner, int) and not isinstance(winner, bool):
                            remaining_ron_winners.add(winner)
                        lookahead_index += 1
                    more_ron_expected = len(remaining_ron_winners) > 1
                self._on_hule(
                    value,
                    more_ron_expected=more_ron_expected,
                    remaining_ron_winners=remaining_ron_winners,
                )
            elif key == "pingju":
                self._on_pingju(value)
            else:
                self.tokens.append(f"event_unknown_{key}")

            self.event_traces.append(
                EventTrace(
                    round_index=self.round_index,
                    event_key=key,
                    start=before_token_count,
                    end=len(self.tokens),
                )
            )

            if key in {"hule", "pingju"}:
                round_ended = True

        self._flush_pending()
        self.round_index += 1

    def _flush_pending(self) -> None:
        if self.pending_reaction:
            self._finalize_reaction(close_reason="voluntary")
        if self.pending_self:
            self._finalize_self(set())

    def _apply_riichi_stick(self, actor: int) -> None:
        self.players[actor].score -= 1000

    def _build_dora_block(self, tile: str) -> List[str]:
        return ["dora", tile]

    def _record_kan_dora_mode(self, mode: str) -> None:
        if mode not in {"immediate", "delayed"}:
            raise TokenizeError(f"invalid kan dora mode: {mode}")
        self.pending_kan_dora_modes.append(mode)

    def _emit_pending_dora_reveals(self, count: Optional[int] = None) -> None:
        if count is None:
            count = len(self.pending_dora_tiles)
        if count > len(self.pending_dora_tiles):
            raise TokenizeError("not enough pending dora reveals")
        for _ in range(count):
            tile = self.pending_dora_tiles.pop(0)
            if self.pending_kan_dora_modes:
                self.pending_kan_dora_modes.pop(0)
            self.tokens.extend(self._build_dora_block(tile))

    def _kaigang_lookahead_count_before_gangzimo(self) -> int:
        if not self.pending_kan_dora_modes:
            return 0
        if "immediate" in self.pending_kan_dora_modes:
            return self.pending_kan_dora_modes.index("immediate") + 1
        return 0

    def _dora_reveal_count_before_gangzimo(self) -> int:
        if not self.pending_kan_dora_modes:
            return 0
        if "immediate" in self.pending_kan_dora_modes:
            return self.pending_kan_dora_modes.index("immediate") + 1
        if self.pending_dora_tiles:
            return 1
        return 0

    def _consume_following_kaigang_events(
        self,
        round_data: list,
        event_index: int,
        max_count: Optional[int],
        skipped_event_indices: Set[int],
    ) -> int:
        consumed = 0
        lookahead_index = event_index + 1
        while lookahead_index < len(round_data):
            if max_count is not None and consumed >= max_count:
                break
            lookahead_event = round_data[lookahead_index]
            if not isinstance(lookahead_event, dict) or "kaigang" not in lookahead_event:
                break
            self._on_kaigang(lookahead_event["kaigang"])
            skipped_event_indices.add(lookahead_index)
            consumed += 1
            lookahead_index += 1
        return consumed

    def _build_game_rule_block(self, game: dict) -> List[str]:
        title = game.get("title")
        block: List[str] = []
        if self.seat_count == 3:
            block.append("rule_player_3")
        elif self.seat_count == 4:
            block.append("rule_player_4")
        if title is None:
            return block
        title = self._require_str(title, field="game.title")
        if "東" in title:
            block.append("rule_length_tonpu")
        elif "南" in title:
            block.append("rule_length_hanchan")
        return block

    def _build_score_delta_block(self, deltas: List[int]) -> List[str]:
        block: List[str] = []
        for seat in range(self.seat_count):
            block.append(f"score_delta_{seat}")
            block.extend(encode_tenbo_tokens(deltas[seat]))
        return block

    def _build_rank_block(self, places: List[int]) -> List[str]:
        return [f"rank_{seat}_{place}" for seat, place in enumerate(places)]

    def _build_final_score_block(self, scores: List[int]) -> List[str]:
        block: List[str] = []
        for seat, score in enumerate(scores):
            block.append(f"final_score_{seat}")
            block.extend(encode_tenbo_tokens(score))
        return block

    def _build_final_rank_block(self, places: List[int]) -> List[str]:
        return [f"final_rank_{seat}_{place}" for seat, place in enumerate(places)]

    def _build_reaction_detail_block(
        self,
        *,
        action: str,
        chi_pos: Optional[str],
        red_choice: Optional[str],
    ) -> List[str]:
        block: List[str] = []
        if action == "chi" and chi_pos is not None:
            block.append(f"chi_pos_{chi_pos}")
        if red_choice is not None:
            block.append(red_choice)
        return block

    def _build_self_action_block(
        self,
        *,
        seat: int,
        kind: str,
        opt: str,
        tile_token: Optional[str] = None,
    ) -> List[str]:
        block = [f"{kind}_self_{seat}_{opt}"]
        if tile_token is not None:
            block.append(tile_token)
        return block

    def _build_reaction_action_block(
        self,
        *,
        seat: int,
        kind: str,
        opt: str,
        reason: Optional[str] = None,
    ) -> List[str]:
        if kind == "pass":
            if reason is None:
                raise TokenizeError("reaction pass requires a reason")
            return [f"pass_react_{seat}_{opt}_{reason}"]
        return [f"{kind}_react_{seat}_{opt}"]

    def _build_reaction_option_block(self, options_by_player: Dict[int, Set[str]]) -> List[str]:
        block: List[str] = []
        for seat, opt in self._iter_reaction_priority_entries(options_by_player):
            block.extend(self._build_reaction_action_block(seat=seat, kind="opt", opt=opt))
        return block

    def _self_option_order(self, options: Set[str]) -> List[str]:
        priority = {
            "tsumo": 0,
            "kyushukyuhai": 1,
            "penuki": 2,
            "riichi": 3,
            "ankan": 4,
            "kakan": 5,
        }
        return sorted(options, key=lambda opt: (priority.get(opt, 99), opt))

    def _reaction_priority_sort_key(self, seat: int, opt: str) -> Tuple[int, int, int, str]:
        priority = {"ron": 0, "pon": 1, "minkan": 1, "chi": 2}.get(opt, 99)
        # pon/minkan share priority by rule; keep deterministic tie-break order.
        tiebreak = {"ron": 0, "pon": 1, "minkan": 2, "chi": 3}.get(opt, 99)
        return (priority, seat, tiebreak, opt)

    def _iter_reaction_priority_entries(self, options_by_player: Dict[int, Set[str]]) -> List[Tuple[int, str]]:
        entries: List[Tuple[int, str]] = []
        for seat, opts in options_by_player.items():
            for opt in opts:
                entries.append((seat, opt))
        return sorted(entries, key=lambda item: self._reaction_priority_sort_key(item[0], item[1]))

    def _build_round_prelude_block(
        self,
        *,
        bakaze: int,
        kyoku: int,
        honba: int,
        riichi_sticks: int,
        dora_tile: str,
        scores: List[int],
        shoupai: List[str],
    ) -> List[str]:
        block = [
            "round_start",
            f"bakaze_{bakaze}",
            f"kyoku_{kyoku}",
            "honba",
            *encode_tenbo_tokens(honba * 100),
            "riichi_sticks",
            *encode_tenbo_tokens(riichi_sticks * 1000),
            *self._build_dora_block(dora_tile),
        ]
        for seat, score in enumerate(scores):
            block.append(f"score_{seat}")
            block.extend(encode_tenbo_tokens(score))
        for seat, hand in enumerate(shoupai):
            hand_tiles = _parse_tiles(hand, stop_at_comma=True, context="hand")
            block.append(f"haipai_{seat}")
            block.extend(sorted(hand_tiles, key=token_tile_sort_key))
        return block

    def _game_seat_for_round_seat(self, seat: int) -> int:
        return (self.initial_qijia + self.kyoku + seat) % self.seat_count

    def _compute_rank_places(self, scores: List[int], order_keys: List[int]) -> List[int]:
        sorted_seats = sorted(range(self.seat_count), key=lambda seat: (-scores[seat], order_keys[seat]))
        places = [0] * self.seat_count
        for place, seat in enumerate(sorted_seats, start=1):
            places[seat] = place
        return places

    def _compute_round_rank_places(self) -> List[int]:
        scores = [player.score for player in self.players]
        order_keys = [(self.kyoku + seat) % self.seat_count for seat in range(self.seat_count)]
        return self._compute_rank_places(scores, order_keys)

    def _current_game_order_scores(self) -> List[int]:
        scores = [0] * self.seat_count
        for seat, player in enumerate(self.players):
            scores[self._game_seat_for_round_seat(seat)] = player.score
        return scores

    def _compute_final_rank_places(self, scores: List[int]) -> List[int]:
        if not self.has_initial_qijia and len(set(scores)) != len(scores):
            raise TokenizeError("game.qijia is required to reconstruct tied final ranks")
        order_keys = [((seat - self.initial_qijia) % self.seat_count) for seat in range(self.seat_count)]
        return self._compute_rank_places(scores, order_keys)

    def _can_consume_concealed_token(
        self,
        counts: List[int],
        red_fives: Dict[str, int],
        tile_token: str,
    ) -> bool:
        idx = tile_to_index(tile_token)
        if counts[idx] <= 0:
            return False
        suit = tile_token[0]
        digit = tile_token[1]
        if digit == "0":
            return suit in red_fives and red_fives[suit] > 0
        if suit in red_fives and digit == "5":
            normal_fives = counts[idx] - red_fives[suit]
            return normal_fives > 0
        return True

    def _consume_concealed_token(self, p: PlayerState, tile_token: str) -> None:
        if not self._can_consume_concealed_token(p.concealed, p.red_fives, tile_token):
            raise TokenizeError(f"cannot consume tile from concealed hand: {tile_token}")
        idx = tile_to_index(tile_token)
        p.concealed[idx] -= 1
        if tile_token[1] == "0" and tile_token[0] in p.red_fives:
            p.red_fives[tile_token[0]] -= 1

    def _consume_unspecified_tile(self, p: PlayerState, tile_idx: int, n: int = 1) -> None:
        base_token = index_to_tile(tile_idx)
        suit = base_token[0]
        for _ in range(n):
            tile_token = base_token
            if suit in p.red_fives and base_token[1] == "5":
                normal_fives = p.concealed[tile_idx] - p.red_fives[suit]
                tile_token = f"{suit}5" if normal_fives > 0 else f"{suit}0"
            self._consume_concealed_token(p, tile_token)

    def _add_concealed_token(self, p: PlayerState, tile_token: str) -> int:
        idx = tile_to_index(tile_token)
        p.concealed[idx] += 1
        if tile_token[1] == "0" and tile_token[0] in p.red_fives:
            p.red_fives[tile_token[0]] += 1
        return idx

    def _infer_consumed_meld_tokens(
        self,
        p: PlayerState,
        meld_token_tiles: List[str],
        called_index: Optional[int],
    ) -> List[str]:
        target = max(0, len(meld_token_tiles) - 1)
        counts = list(p.concealed)
        red_fives = dict(p.red_fives)
        consumed: List[str] = []
        blocked = called_index if called_index is not None and 0 <= called_index < len(meld_token_tiles) else None
        order = [i for i in range(len(meld_token_tiles)) if i != blocked]
        order.extend(i for i in range(len(meld_token_tiles)) if i == blocked)

        for idx in order:
            if len(consumed) >= target:
                break
            tile_token = meld_token_tiles[idx]
            if not self._can_consume_concealed_token(counts, red_fives, tile_token):
                continue
            tile_idx = tile_to_index(tile_token)
            counts[tile_idx] -= 1
            if tile_token[1] == "0" and tile_token[0] in red_fives:
                red_fives[tile_token[0]] -= 1
            consumed.append(tile_token)

        if len(consumed) != target:
            raise TokenizeError("cannot infer fulou consumption from concealed hand")
        return consumed

    def _chi_pos_label(
        self,
        meld_tiles: List[int],
        called_index: Optional[int],
        discard_tile: Optional[int],
    ) -> Optional[str]:
        if len(meld_tiles) != 3:
            return None
        sorted_tiles = sorted(meld_tiles)
        if sorted_tiles[0] == sorted_tiles[1] or sorted_tiles[1] == sorted_tiles[2]:
            return None
        called_tile = discard_tile
        if called_index is not None and 0 <= called_index < len(meld_tiles):
            called_tile = meld_tiles[called_index]
        if called_tile is None:
            return None
        if called_tile == sorted_tiles[0]:
            return "low"
        if called_tile == sorted_tiles[1]:
            return "mid"
        if called_tile == sorted_tiles[2]:
            return "high"
        return None

    def _red_choice_token(
        self,
        action: str,
        consumed_tokens: List[str],
        pre_counts: List[int],
        pre_red_fives: Dict[str, int],
    ) -> Optional[str]:
        if action not in {"chi", "pon"}:
            return None

        five_tokens = [t for t in consumed_tokens if t[0] in {"m", "p", "s"} and t[1] in {"0", "5"}]
        if not five_tokens:
            return None
        used_red = any(t[1] == "0" for t in five_tokens)
        return f"red_{'used' if used_red else 'not_used'}"

    def _player_wind(self, seat: int) -> int:
        return (seat - self.dealer_seat) % self.seat_count

    def _invalidate_wait_mask(self, seat: int) -> None:
        self.players[seat].wait_mask_cache = None

    def _invalidate_meld_cache(self, seat: int) -> None:
        self.players[seat].encoded_melds_cache = None

    def _player_encoded_melds(self, seat: int) -> List[Tuple[int, int]]:
        p = self.players[seat]
        if p.encoded_melds_cache is None:
            p.encoded_melds_cache = _encode_pm_melds(p.melds)
        return p.encoded_melds_cache

    def _sim_apply(self, seat: int, action_type: object, pai_34: int, red: bool = False, bias: Optional[int] = None) -> None:
        p = self.players[seat]
        if p.sim_shoupai is None:
            return
        if bias is None:
            action = pm.Action(action_type, int(pai_34), bool(red))
        else:
            action = pm.Action(action_type, int(pai_34), bool(red), int(bias))
        p.sim_shoupai.apply(action)

    def _use_multi_player_simulation(self) -> bool:
        if self.seat_count == 3:
            return PM_STATELESS_SIMULATION_API_AVAILABLE and PM_THREE_PLAYER_API_AVAILABLE
        return PM_SIMULATION_API_AVAILABLE

    def _simulation_supports_three_player(self) -> bool:
        return self.seat_count == 3 and PM_THREE_PLAYER_API_AVAILABLE

    def _use_shoupai_simulation(self) -> bool:
        return (
            self.seat_count == 4
            and PM_SHOUPAI_SIMULATION_API_AVAILABLE
            and all(p.sim_shoupai is not None for p in self.players)
        )

    def _simulation_reaction_players_payload(self):
        return [
            (
                tuple(p.concealed),
                self._player_encoded_melds(seat),
                bool(p.is_riichi),
                bool(p.temporary_furiten),
                bool(p.riichi_furiten),
                _furiten_mask(p.furiten_tiles),
                int(p.open_melds),
            )
            for seat, p in enumerate(self.players)
        ]

    def _simulation_reaction_shoupai_payload(self):
        return [
            (
                p.sim_shoupai,
                bool(p.is_riichi),
                bool(p.temporary_furiten),
                bool(p.riichi_furiten),
                _furiten_mask(p.furiten_tiles),
                int(p.open_melds),
            )
            for p in self.players
        ]

    def _compute_simulation_reaction_pairs(self, discarder: int, tile_idx: int) -> List[Tuple[int, int]]:
        if self._use_shoupai_simulation():
            return pm.compute_reaction_option_masks_shoupai(
                self._simulation_reaction_shoupai_payload(),
                int(discarder),
                int(tile_idx),
                int(self.bakaze),
                int(self.dealer_seat),
                int(self.live_draws_left),
                bool(self.last_draw_was_gangzimo),
                *( [True] if self._simulation_supports_three_player() else [] ),
            )
        if PM_STATELESS_SIMULATION_API_AVAILABLE:
            return pm.compute_reaction_option_masks(
                self._simulation_reaction_players_payload(),
                int(discarder),
                int(tile_idx),
                int(self.bakaze),
                int(self.dealer_seat),
                int(self.live_draws_left),
                bool(self.last_draw_was_gangzimo),
                *( [True] if self._simulation_supports_three_player() else [] ),
            )
        return []

    def _compute_simulation_rob_kan_pairs(
        self,
        actor: int,
        tile_idx: int,
        require_kokushi: bool,
    ) -> List[Tuple[int, int]]:
        if self._use_shoupai_simulation():
            return pm.compute_rob_kan_option_masks_shoupai(
                self._simulation_reaction_shoupai_payload(),
                int(actor),
                int(tile_idx),
                int(self.bakaze),
                int(self.dealer_seat),
                bool(require_kokushi),
                *( [True] if self._simulation_supports_three_player() else [] ),
            )
        if PM_STATELESS_SIMULATION_API_AVAILABLE:
            return pm.compute_rob_kan_option_masks(
                self._simulation_reaction_players_payload(),
                int(actor),
                int(tile_idx),
                int(self.bakaze),
                int(self.dealer_seat),
                bool(require_kokushi),
                *( [True] if self._simulation_supports_three_player() else [] ),
            )
        return []

    def _decode_simulation_reaction_options(
        self,
        result_pairs: List[Tuple[int, int]],
    ) -> Dict[int, Set[str]]:
        options_by_player: Dict[int, Set[str]] = {}
        for seat, mask in result_pairs:
            opts: Set[str] = set()
            if mask & REACT_OPT_RON:
                opts.add("ron")
            if mask & REACT_OPT_CHI:
                opts.add("chi")
            if mask & REACT_OPT_PON:
                opts.add("pon")
            if mask & REACT_OPT_MINKAN:
                opts.add("minkan")
            if opts:
                options_by_player[int(seat)] = opts
        return options_by_player

    def _wait_mask(self, seat: int) -> int:
        p = self.players[seat]
        if p.wait_mask_cache is None:
            if self.seat_count == 3:
                p.wait_mask_cache = _pm_wait_mask(p.concealed, p.meld_count, True)
            else:
                p.wait_mask_cache = _pm_wait_mask(p.concealed, p.meld_count)
        return p.wait_mask_cache

    def _can_win(self, seat: int, win_tile: int, is_tsumo: bool) -> bool:
        p = self.players[seat]
        return self._can_win_with_counts(
            seat=seat,
            counts=p.concealed,
            win_tile=win_tile,
            is_tsumo=is_tsumo,
        )

    def _can_win_with_counts(
        self,
        seat: int,
        counts: List[int],
        win_tile: int,
        is_tsumo: bool,
    ) -> bool:
        p = self.players[seat]
        return _pm_has_hupai(
            counts=counts,
            melds=p.melds,
            encoded_melds=self._player_encoded_melds(seat),
            win_tile=win_tile,
            is_tsumo=is_tsumo,
            is_menqian=(p.open_melds == 0),
            is_riichi=p.is_riichi,
            zhuangfeng=self.bakaze,
            lunban=self._player_wind(seat),
            three_player=(self.seat_count == 3),
        )

    def _has_riichi_discard(self, seat: int) -> bool:
        p = self.players[seat]
        if self.seat_count == 3:
            return _pm_has_riichi_discard(p.concealed, p.closed_kans, True)
        return _pm_has_riichi_discard(p.concealed, p.closed_kans)

    def _evaluate_draw(
        self,
        seat: int,
        drawn_tile: int,
        check_riichi_discard: bool,
        is_haidi: bool = False,
        is_lingshang: bool = False,
    ) -> Tuple[bool, bool]:
        p = self.players[seat]
        kwargs = dict(
            counts=p.concealed,
            melds=p.melds,
            encoded_melds=self._player_encoded_melds(seat),
            win_tile=drawn_tile,
            is_menqian=(p.open_melds == 0),
            is_riichi=p.is_riichi,
            zhuangfeng=self.bakaze,
            lunban=self._player_wind(seat),
            closed_kans=p.closed_kans,
            check_riichi_discard=check_riichi_discard,
            is_haidi=is_haidi,
            is_lingshang=is_lingshang,
        )
        if self.seat_count == 3:
            return _pm_evaluate_draw(**kwargs, three_player=True)
        return _pm_evaluate_draw(**kwargs)

    def _is_permanent_furiten(self, seat: int, wait_mask: Optional[int] = None) -> bool:
        p = self.players[seat]
        seat_wait_mask = self._wait_mask(seat) if wait_mask is None else wait_mask
        if seat_wait_mask == 0:
            return False
        for tile in p.furiten_tiles:
            if (seat_wait_mask >> tile) & 1:
                return True
        return False

    def _on_qipai(self, q: dict) -> None:
        if not isinstance(q, dict):
            raise TokenizeError("qipai payload must be a dict")
        shoupai = self._require_seat_list(q["shoupai"], field="qipai.shoupai")
        defen = self._require_seat_list(q["defen"], field="qipai.defen")
        if not all(isinstance(hand, str) for hand in shoupai):
            raise TokenizeError("qipai.shoupai entries must be strings")
        baopai = self._require_str(q["baopai"], field="qipai.baopai")
        token_tile(_strip_tile_suffix(baopai))

        hand_counts_list = [parse_hand_counts(hand) for hand in shoupai]
        hand_red_fives_list = [parse_hand_red_fives(hand) for hand in shoupai]
        for counts in hand_counts_list:
            if sum(counts) != 13:
                raise TokenizeError("qipai hand must contain exactly 13 tiles")
            if any(count > 4 for count in counts):
                raise TokenizeError("qipai hand cannot contain more than four of a tile")
        scores = [self._require_score(score, field=f"qipai.defen[{seat}]") for seat, score in enumerate(defen)]
        self.players = [
            PlayerState(
                concealed=list(hand_counts_list[seat]),
                score=scores[seat],
                red_fives=dict(hand_red_fives_list[seat]),
                sim_shoupai=(
                    pm.Shoupai(tuple(hand_counts_list[seat]))
                    if PM_SHOUPAI_SIMULATION_API_AVAILABLE
                    else None
                ),
            )
            for seat in range(self.seat_count)
        ]
        bakaze = self._require_non_negative_int(q["zhuangfeng"], field="qipai.zhuangfeng")
        kyoku = self._require_non_negative_int(q["jushu"], field="qipai.jushu")
        honba = self._require_non_negative_int(q["changbang"], field="qipai.changbang")
        riichi_sticks = self._require_non_negative_int(q["lizhibang"], field="qipai.lizhibang")

        self.live_draws_left = 55 if self.seat_count == 3 else 70
        self.bakaze = bakaze
        self.kyoku = kyoku
        # Tenhou JSON used here is seat-rotated so dealer is always seat 0.
        self.dealer_seat = 0
        self.first_turn_open_calls_seen = False
        self.last_draw_was_gangzimo = False
        self.expected_draw_actor = None
        self.expected_discard_actor = None
        self.awaiting_kaigang = 0
        self.pending_kan_dora_modes = []
        self.pending_dora_tiles = []
        self.pending_dead_wall_draw = False

        self.tokens.extend(
            self._build_round_prelude_block(
                bakaze=bakaze,
                kyoku=kyoku,
                honba=honba,
                riichi_sticks=riichi_sticks,
                dora_tile=_strip_tile_suffix(baopai).replace("0", "5"),
                scores=scores,
                shoupai=shoupai,
            )
        )

    def _on_draw(self, z: dict, is_gangzimo: bool, is_replacement_draw: bool = False) -> None:
        self._require_round_initialized()
        z = self._require_dict(z, field="draw")
        if self.live_draws_left <= 0 and not is_replacement_draw:
            raise TokenizeError("no live draws remaining")
        if self.expected_discard_actor is not None:
            raise TokenizeError("draw is not allowed before discard resolution")
        actor = self._require_seat(z["l"], field="draw.l")
        if self.expected_draw_actor is not None and actor != self.expected_draw_actor:
            raise TokenizeError(f"unexpected draw actor: {actor}")
        tile_str = _strip_tile_suffix(self._require_str(z["p"], field="draw.p"))
        tile_token = token_tile(tile_str)
        tile_idx = self._add_concealed_token(self.players[actor], tile_token)
        self._sim_apply(actor, pm.ActionType.zimo, tile_idx, red=(tile_token[1] == "0"))
        self.players[actor].last_draw_tile = tile_token
        self.players[actor].temporary_furiten = False
        self._invalidate_wait_mask(actor)
        if self.live_draws_left > 0:
            self.live_draws_left -= 1
        self.last_draw_was_gangzimo = is_gangzimo

        if is_gangzimo:
            reveal_count = min(len(self.pending_dora_tiles), self._dora_reveal_count_before_gangzimo())
            if reveal_count:
                self._emit_pending_dora_reveals(reveal_count)
        self.tokens.append(f"draw_{actor}_{tile_token}")

        options = self._compute_self_options(actor, tile_idx, is_gangzimo=is_gangzimo)
        option_tiles = self._self_option_tiles(actor, drawn_tile=tile_idx)
        self._emit_self_options(actor, options, option_tiles)
        self.pending_self = SelfDecision(actor=actor, options=options, option_tiles=option_tiles)
        self.players[actor].is_first_turn = False
        self.expected_draw_actor = None
        self.expected_discard_actor = actor

    def _compute_self_options(self, actor: int, drawn_tile: int, is_gangzimo: bool = False) -> Set[str]:
        p = self.players[actor]
        options: Set[str]
        if self._use_multi_player_simulation():
            if self.seat_count == 4 and PM_SHOUPAI_SIMULATION_API_AVAILABLE and p.sim_shoupai is not None:
                option_mask = int(
                    pm.compute_self_option_mask_shoupai(
                        p.sim_shoupai,
                        int(drawn_tile),
                        bool(p.is_riichi),
                        int(p.score),
                        int(self.bakaze),
                        int(self._player_wind(actor)),
                        int(self.live_draws_left),
                        int(p.closed_kans),
                        int(p.open_melds),
                        [int(tile) for tile in p.open_pons],
                        bool(p.is_first_turn),
                        bool(self.first_turn_open_calls_seen),
                        bool(is_gangzimo),
                        *( [True] if self._simulation_supports_three_player() else [] ),
                    )
                )
            elif PM_STATELESS_SIMULATION_API_AVAILABLE:
                option_mask = int(
                    pm.compute_self_option_mask(
                        tuple(p.concealed),
                        self._player_encoded_melds(actor),
                        int(drawn_tile),
                        bool(p.is_riichi),
                        int(p.score),
                        int(self.bakaze),
                        int(self._player_wind(actor)),
                        int(self.live_draws_left),
                        int(p.closed_kans),
                        int(p.open_melds),
                        [int(tile) for tile in p.open_pons],
                        bool(p.is_first_turn),
                        bool(self.first_turn_open_calls_seen),
                        bool(is_gangzimo),
                        *( [True] if self._simulation_supports_three_player() else [] ),
                    )
                )
            else:
                option_mask = 0
            options = set()
            if option_mask & SELF_OPT_TSUMO:
                options.add("tsumo")
            if option_mask & SELF_OPT_RIICHI:
                options.add("riichi")
            if option_mask & SELF_OPT_ANKAN:
                options.add("ankan")
            if option_mask & SELF_OPT_KAKAN:
                options.add("kakan")
            if option_mask & SELF_OPT_KYUSHUKYUHAI:
                options.add("kyushukyuhai")
            if option_mask & SELF_OPT_PENUKI:
                options.add("penuki")
        else:
            options = set()
            can_riichi = (
                not p.is_riichi
                and p.open_melds == 0
                and p.score >= 1000
                and self.live_draws_left >= 4
            )
            is_haidi = self.live_draws_left == 0 and not is_gangzimo
            can_tsumo, has_riichi_discard = self._evaluate_draw(
                seat=actor,
                drawn_tile=drawn_tile,
                check_riichi_discard=can_riichi,
                is_haidi=is_haidi,
                is_lingshang=is_gangzimo,
            )
            if can_tsumo:
                options.add("tsumo")

            if can_riichi and has_riichi_discard:
                options.add("riichi")

            if self._can_ankan(actor, drawn_tile):
                options.add("ankan")

            if any(p.concealed[tile] > 0 for tile in p.open_pons):
                options.add("kakan")

            if self._can_kyushukyuhai(actor):
                options.add("kyushukyuhai")

            if self.seat_count == 3 and p.concealed[tile_to_index("z4")] > 0:
                options.add("penuki")

        return options

    def _can_kyushukyuhai(self, actor: int) -> bool:
        p = self.players[actor]
        if not p.is_first_turn:
            return False
        if self.first_turn_open_calls_seen:
            return False
        if p.open_melds > 0 or p.closed_kans > 0 or p.melds:
            return False
        uniq_yaochu = sum(1 for i in YAOCHU_INDICES if p.concealed[i] > 0)
        return uniq_yaochu >= 9
    def _can_ankan(self, actor: int, drawn_tile: Optional[int] = None) -> bool:
        return bool(self._ankan_candidate_tiles(actor, drawn_tile=drawn_tile))

    def _on_discard(self, d: dict) -> None:
        self._require_round_initialized()
        d = self._require_dict(d, field="dapai")
        actor = self._require_seat(d["l"], field="dapai.l")
        if self.expected_discard_actor is None or actor != self.expected_discard_actor:
            raise TokenizeError(f"unexpected discard actor: {actor}")
        raw_tile = self._require_str(d["p"], field="dapai.p")
        tile_str = _strip_tile_suffix(raw_tile)
        tile_token = token_tile(tile_str)
        tile_idx = tile_to_index(tile_token)
        is_riichi = "*" in raw_tile
        is_tsumogiri = "_" in raw_tile

        chosen: Set[str] = {"riichi"} if is_riichi else set()
        self._finalize_self(chosen, actor=actor)

        self._consume_concealed_token(self.players[actor], tile_token)
        self._sim_apply(actor, pm.ActionType.lizhi if is_riichi else pm.ActionType.dapai, tile_idx, red=(tile_token[1] == "0"))
        self.players[actor].last_draw_tile = None
        self.players[actor].furiten_tiles.add(tile_idx)
        self._invalidate_wait_mask(actor)

        discard_kind = "tsumogiri" if is_tsumogiri else "tedashi"
        self.tokens.append(f"discard_{actor}_{tile_token}_{discard_kind}")
        self._emit_pending_dora_reveals()

        if is_riichi:
            self.players[actor].is_riichi = True

        reaction = self._compute_reaction_options(actor, tile_idx)
        self.last_draw_was_gangzimo = False
        self.expected_discard_actor = None
        if reaction:
            self.pending_reaction = reaction
            if is_riichi:
                has_ron_option = any("ron" in opts for opts in reaction.options_by_player.values())
                if has_ron_option:
                    self.pending_riichi_actor = actor
                else:
                    self._apply_riichi_stick(actor)
            self.tokens.extend(self._build_reaction_option_block(reaction.options_by_player))
        elif is_riichi:
            self._apply_riichi_stick(actor)
            self.expected_draw_actor = (actor + 1) % self.seat_count
        else:
            self.expected_draw_actor = (actor + 1) % self.seat_count

    def _compute_reaction_options(self, discarder: int, tile_idx: int) -> Optional[ReactionDecision]:
        use_simulation_api = self._use_multi_player_simulation() and not (
            self.live_draws_left == 0 and self.last_draw_was_gangzimo
        )
        if use_simulation_api:
            result_pairs = self._compute_simulation_reaction_pairs(discarder, tile_idx)
            if not result_pairs:
                return None
            options_by_player = self._decode_simulation_reaction_options(result_pairs)
            if not options_by_player:
                return None
            options_by_player = self._filter_reaction_call_options(
                discarder=discarder,
                tile_idx=tile_idx,
                options_by_player=options_by_player,
            )
            if not options_by_player:
                return None
            return ReactionDecision(
                discarder=discarder,
                discard_tile=tile_idx,
                options_by_player=options_by_player,
                trigger="discard",
            )
        options_by_player: Dict[int, Set[str]] = {}
        ron_cases: List[
            Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int, bool, bool, bool, bool]
        ] = []
        ron_case_seats: List[int] = []
        # A ron on the last discard after a rinshan draw is still houtei.
        is_haidi = self.live_draws_left == 0

        for offset in range(1, self.seat_count):
            seat = (discarder + offset) % self.seat_count
            p = self.players[seat]
            if not p.temporary_furiten and not p.riichi_furiten:
                wait_mask = self._wait_mask(seat)
                permanent_furiten = self._is_permanent_furiten(seat, wait_mask=wait_mask)
                if not permanent_furiten and ((wait_mask >> tile_idx) & 1):
                    counts_plus = list(p.concealed)
                    counts_plus[tile_idx] += 1
                    ron_cases.append(
                        (
                            counts_plus,
                            p.melds,
                            tile_idx,
                            False,
                            (p.open_melds == 0),
                            p.is_riichi,
                            self.bakaze,
                            self._player_wind(seat),
                            is_haidi,
                            False,
                            False,
                            self.seat_count == 3,
                        )
                    )
                    ron_case_seats.append(seat)

        ron_seats: Set[int] = set()
        if ron_cases:
            for seat, can_ron in zip(ron_case_seats, _pm_has_hupai_multi(ron_cases)):
                if can_ron:
                    ron_seats.add(seat)

        for offset in range(1, self.seat_count):
            seat = (discarder + offset) % self.seat_count
            p = self.players[seat]
            options: Set[str] = set()
            if seat in ron_seats:
                options.add("ron")
            if p.is_riichi:
                if options:
                    options_by_player[seat] = options
                continue

            if self.live_draws_left > 0:
                if self.seat_count == 4 and offset == 1 and self._can_chi(p.concealed, tile_idx):
                    options.add("chi")
                if p.concealed[tile_idx] >= 2:
                    options.add("pon")
                if p.concealed[tile_idx] >= 3:
                    options.add("minkan")

            if options:
                options_by_player[seat] = options

        if not options_by_player:
            return None
        options_by_player = self._filter_reaction_call_options(
            discarder=discarder,
            tile_idx=tile_idx,
            options_by_player=options_by_player,
        )
        if not options_by_player:
            return None
        return ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player=options_by_player,
            trigger="discard",
        )

    def _filter_reaction_call_options(
        self,
        discarder: int,
        tile_idx: int,
        options_by_player: Dict[int, Set[str]],
    ) -> Dict[int, Set[str]]:
        filtered: Dict[int, Set[str]] = {}
        for seat, opts in options_by_player.items():
            seat_filtered = set(opts)
            if "chi" in seat_filtered and not self._can_chi_with_tenhou_kuikae(seat, tile_idx):
                seat_filtered.remove("chi")
            if "pon" in seat_filtered and not self._can_pon_with_tenhou_kuikae(seat, tile_idx):
                seat_filtered.remove("pon")
            if seat_filtered:
                filtered[seat] = seat_filtered
        return filtered

    def _compute_kakan_reaction_options(self, actor: int, tile_idx: int) -> Optional[ReactionDecision]:
        if self._use_multi_player_simulation():
            result_pairs = self._compute_simulation_rob_kan_pairs(actor, tile_idx, require_kokushi=False)
            options_by_player = {
                int(seat): {"ron"} for seat, mask in result_pairs if mask & REACT_OPT_RON
            }
            if not options_by_player:
                return None
            return ReactionDecision(
                discarder=actor,
                discard_tile=tile_idx,
                options_by_player=options_by_player,
                trigger="kakan",
            )
        options_by_player: Dict[int, Set[str]] = {}
        ron_cases: List[
            Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int, bool, bool, bool, bool]
        ] = []
        ron_case_seats: List[int] = []

        for seat in range(self.seat_count):
            if seat == actor:
                continue

            p = self.players[seat]
            if p.temporary_furiten or p.riichi_furiten:
                continue

            wait_mask = self._wait_mask(seat)
            if self._is_permanent_furiten(seat, wait_mask=wait_mask) or not ((wait_mask >> tile_idx) & 1):
                continue

            counts_plus = list(p.concealed)
            counts_plus[tile_idx] += 1
            ron_cases.append(
                (
                    counts_plus,
                    p.melds,
                    tile_idx,
                    False,
                    (p.open_melds == 0),
                    p.is_riichi,
                    self.bakaze,
                    self._player_wind(seat),
                    False,
                    False,
                    True,
                    self.seat_count == 3,
                )
            )
            ron_case_seats.append(seat)

        if ron_cases:
            for seat, can_ron in zip(ron_case_seats, _pm_has_hupai_multi(ron_cases)):
                if can_ron:
                    options_by_player[seat] = {"ron"}

        if not options_by_player:
            return None
        return ReactionDecision(
            discarder=actor,
            discard_tile=tile_idx,
            options_by_player=options_by_player,
            trigger="kakan",
        )

    def _compute_ankan_reaction_options(self, actor: int, tile_idx: int) -> Optional[ReactionDecision]:
        # Tenhou does not allow kokushi ron/chankan on another player's closed kan.
        # Therefore ankan never opens a reaction decision window in this corpus.
        return None

    def _compute_penuki_reaction_options(self, actor: int, tile_idx: int) -> Optional[ReactionDecision]:
        if self.seat_count != 3:
            return None
        options_by_player: Dict[int, Set[str]] = {}
        ron_cases: List[
            Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int, bool, bool, bool, bool]
        ] = []
        ron_case_seats: List[int] = []

        for seat in range(self.seat_count):
            if seat == actor:
                continue

            p = self.players[seat]
            if p.temporary_furiten or p.riichi_furiten:
                continue

            wait_mask = self._wait_mask(seat)
            if self._is_permanent_furiten(seat, wait_mask=wait_mask) or not ((wait_mask >> tile_idx) & 1):
                continue

            counts_plus = list(p.concealed)
            counts_plus[tile_idx] += 1
            ron_cases.append(
                (
                    counts_plus,
                    p.melds,
                    tile_idx,
                    False,
                    (p.open_melds == 0),
                    p.is_riichi,
                    self.bakaze,
                    self._player_wind(seat),
                    False,
                    False,
                    True,
                    True,
                )
            )
            ron_case_seats.append(seat)

        if ron_cases:
            for seat, can_ron in zip(ron_case_seats, _pm_has_hupai_multi(ron_cases)):
                if can_ron:
                    options_by_player[seat] = {"ron"}

        if not options_by_player:
            return None
        return ReactionDecision(
            discarder=actor,
            discard_tile=tile_idx,
            options_by_player=options_by_player,
            trigger="penuki",
        )

    def _can_chi(self, concealed: List[int], tile_idx: int) -> bool:
        if tile_idx >= 27:
            return False
        suit_base = (tile_idx // 9) * 9
        n = tile_idx % 9 + 1

        patterns = ((n - 2, n - 1), (n - 1, n + 1), (n + 1, n + 2))
        for a, b in patterns:
            if a < 1 or b > 9:
                continue
            ai = suit_base + (a - 1)
            bi = suit_base + (b - 1)
            if concealed[ai] > 0 and concealed[bi] > 0:
                return True
        return False

    def _can_chi_with_tenhou_kuikae(self, seat: int, tile_idx: int) -> bool:
        concealed = self.players[seat].concealed
        if tile_idx >= 27:
            return False
        suit_base = (tile_idx // 9) * 9
        n = tile_idx % 9 + 1

        patterns = ((n - 2, n - 1), (n - 1, n + 1), (n + 1, n + 2))
        for a, b in patterns:
            if a < 1 or b > 9:
                continue
            ai = suit_base + (a - 1)
            bi = suit_base + (b - 1)
            if concealed[ai] <= 0 or concealed[bi] <= 0:
                continue
            remaining = list(concealed)
            remaining[ai] -= 1
            remaining[bi] -= 1
            seq_low = min(ai, bi, tile_idx)
            seq_high = max(ai, bi, tile_idx)
            forbidden = {tile_idx}
            if tile_idx == seq_low and seq_high + 1 < suit_base + 9:
                forbidden.add(seq_high + 1)
            elif tile_idx == seq_high and seq_low - 1 >= suit_base:
                forbidden.add(seq_low - 1)
            if _has_legal_post_call_discard(remaining, forbidden):
                return True
        return False

    def _can_pon_with_tenhou_kuikae(self, seat: int, tile_idx: int) -> bool:
        concealed = self.players[seat].concealed
        if concealed[tile_idx] < 2:
            return False
        remaining = list(concealed)
        remaining[tile_idx] -= 2
        return _has_legal_post_call_discard(remaining, {tile_idx})

    def _on_fulou(self, f: dict) -> None:
        self._require_round_initialized()
        f = self._require_dict(f, field="fulou")
        actor = self._require_seat(f["l"], field="fulou.l")
        meld_text = self._require_str(f["m"], field="fulou.m")
        meld_token_tiles, called_index_hint = parse_meld_token_tiles_and_called(meld_text)
        meld_tiles = [tile_to_index(tile) for tile in meld_token_tiles]
        action = classify_fulou(meld_tiles)
        if not self.pending_reaction or self.pending_reaction.trigger != "discard":
            raise TokenizeError("fulou requires a pending discard reaction")
        offered = self.pending_reaction.options_by_player.get(actor, set())
        if action not in offered:
            raise TokenizeError("fulou action was not offered")
        self.first_turn_open_calls_seen = True

        self.pending_reaction.chosen[actor] = action
        discard_tile = self.pending_reaction.discard_tile

        p = self.players[actor]
        pre_counts = list(p.concealed)
        pre_red_fives = dict(p.red_fives)

        called_index = called_index_hint
        if discard_tile is not None:
            if called_index is None or called_index >= len(meld_tiles) or meld_tiles[called_index] != discard_tile:
                matches = [i for i, tile in enumerate(meld_tiles) if tile == discard_tile]
                if matches:
                    called_index = matches[0]

        try:
            consumed_tokens = self._infer_consumed_meld_tokens(p, meld_token_tiles, called_index)
            for tile_token in consumed_tokens:
                self._consume_concealed_token(p, tile_token)
        except TokenizeError:
            self.tokens.append("event_unknown_fulou_detail")
            consumed_tokens = []
            need = Counter(meld_tiles)
            if discard_tile is not None and need.get(discard_tile, 0) > 0:
                need[discard_tile] -= 1
            else:
                need = Counter()
                remaining = len(meld_tiles) - 1
                for tile in meld_tiles:
                    if remaining == 0:
                        break
                    if p.concealed[tile] > need[tile]:
                        need[tile] += 1
                        remaining -= 1
                if remaining != 0:
                    raise TokenizeError("cannot infer fulou consumption from concealed hand")
            for tile, n in need.items():
                if n > 0:
                    self._consume_unspecified_tile(p, tile, n=n)
        self._invalidate_wait_mask(actor)

        p.open_melds += 1
        p.is_riichi = False
        p.temporary_furiten = False

        if action == "chi":
            anchor = min(meld_tiles)
            p.melds.append(("chi", anchor))
            called_tile = discard_tile if discard_tile is not None else meld_tiles[called_index or 0]
            self._sim_apply(actor, pm.ActionType.chi, called_tile, bias=called_tile - anchor)
        elif action == "pon":
            tile = meld_tiles[0]
            p.open_pons[tile] = p.open_pons.get(tile, 0) + 1
            red_in_pon = sum(1 for tile_token in meld_token_tiles if tile_token[1] == "0")
            if red_in_pon:
                p.open_pons_red[tile] = p.open_pons_red.get(tile, 0) + red_in_pon
            p.melds.append(("pon", tile))
            self._sim_apply(actor, pm.ActionType.peng, tile)
        elif action == "minkan":
            tile = meld_tiles[0]
            p.melds.append(("minkan", tile))
            self._sim_apply(actor, pm.ActionType.minggang, tile)
        self._invalidate_meld_cache(actor)

        chi_pos: Optional[str] = None
        if action == "chi":
            chi_pos = self._chi_pos_label(meld_tiles, called_index, discard_tile)
        red_choice = self._red_choice_token(
            action=action,
            consumed_tokens=consumed_tokens,
            pre_counts=pre_counts,
            pre_red_fives=pre_red_fives,
        )
        detail_tokens = self._build_reaction_detail_block(
            action=action,
            chi_pos=chi_pos,
            red_choice=red_choice,
        )
        self._finalize_reaction(chosen_details={(actor, action): detail_tokens})
        if action == "minkan":
            self.awaiting_kaigang += 1
            self._record_kan_dora_mode("delayed")
            p.last_draw_tile = None
            self.expected_draw_actor = actor
            self.expected_discard_actor = None
        else:
            p.last_draw_tile = None
            self.expected_draw_actor = None
            self.expected_discard_actor = actor

    def _on_gang(self, g: dict) -> None:
        self._require_round_initialized()
        g = self._require_dict(g, field="gang")
        actor = self._require_seat(g["l"], field="gang.l")
        meld_text = self._require_str(g["m"], field="gang.m")
        kind = classify_gang(meld_text)
        meld_token_tiles, _called_index = parse_meld_token_tiles_and_called(meld_text)
        meld_tiles = [tile_to_index(tile) for tile in meld_token_tiles]
        tile = meld_tiles[0]
        if not self.pending_self or self.pending_self.actor != actor:
            raise TokenizeError("gang requires a pending self decision")
        if kind not in self.pending_self.options:
            raise TokenizeError("gang action was not offered")
        if self.expected_discard_actor is None or actor != self.expected_discard_actor:
            raise TokenizeError(f"unexpected gang actor: {actor}")

        chosen = {kind}
        had_pending_self = True
        self._finalize_self(chosen, actor=actor, chosen_tiles={kind: index_to_tile(tile)})

        p = self.players[actor]

        if kind == "ankan":
            for tile_token in meld_token_tiles:
                try:
                    self._consume_concealed_token(p, tile_token)
                except TokenizeError:
                    # Tolerate synthetic tests where concealed counts are patched without red_fives sync.
                    self._consume_unspecified_tile(p, tile_to_index(tile_token), n=1)
            p.closed_kans += 1
            p.melds.append(("ankan", tile))
            self._sim_apply(actor, pm.ActionType.angang, tile)
        else:
            tile_token = index_to_tile(tile)
            suit = tile_token[0]
            if suit in p.red_fives and tile_token[1] == "5":
                meld_red_fives = sum(1 for t in meld_token_tiles if t[1] == "0")
                open_pon_red = p.open_pons_red.get(tile, 0)
                added_red = max(0, meld_red_fives - open_pon_red)
                tile_token = f"{suit}0" if added_red > 0 else f"{suit}5"
            try:
                self._consume_concealed_token(p, tile_token)
            except TokenizeError:
                self._consume_unspecified_tile(p, tile, n=1)
            if p.open_pons.get(tile, 0) > 0:
                p.open_pons[tile] -= 1
                if p.open_pons[tile] == 0:
                    del p.open_pons[tile]
            if p.open_pons_red.get(tile, 0) > 0:
                p.open_pons_red[tile] -= 1
                if p.open_pons_red[tile] == 0:
                    del p.open_pons_red[tile]
            replaced = False
            for i, (mtype, mtile) in enumerate(p.melds):
                if mtype == "pon" and mtile == tile:
                    p.melds[i] = ("minkan", tile)
                    replaced = True
                    break
            if not replaced:
                p.melds.append(("minkan", tile))
            self._sim_apply(actor, pm.ActionType.jiagang, tile)
        self._invalidate_meld_cache(actor)
        self._invalidate_wait_mask(actor)

        reaction: Optional[ReactionDecision] = None
        if kind == "kakan":
            reaction = self._compute_kakan_reaction_options(actor, tile)
        self.awaiting_kaigang += 1
        self._record_kan_dora_mode("immediate" if kind == "ankan" else "delayed")
        if reaction:
            self.pending_reaction = reaction
            self.tokens.extend(self._build_reaction_option_block(reaction.options_by_player))
        p.last_draw_tile = None
        self.expected_discard_actor = None
        self.expected_draw_actor = actor

    def _on_kaigang(self, k: dict) -> None:
        if self.awaiting_kaigang <= 0:
            raise TokenizeError("kaigang is not expected")
        k = self._require_dict(k, field="kaigang")
        tile = token_tile(_strip_tile_suffix(self._require_str(k["baopai"], field="kaigang.baopai"))).replace("0", "5")
        self.pending_dora_tiles.append(tile)
        self.awaiting_kaigang -= 1

    def _on_penuki(self, n: dict) -> None:
        self._require_round_initialized()
        if self.seat_count != 3:
            raise TokenizeError("penuki is only valid in three-player games")
        n = self._require_dict(n, field="penuki")
        actor = self._require_seat(n["l"], field="penuki.l")
        tile_token = token_tile(_strip_tile_suffix(self._require_str(n["p"], field="penuki.p")))
        if tile_token != "z4":
            raise TokenizeError("penuki currently only supports North extraction")
        if not self.pending_self or self.pending_self.actor != actor:
            raise TokenizeError("penuki requires a pending self decision")
        if "penuki" not in self.pending_self.options:
            raise TokenizeError("penuki action was not offered")
        self._finalize_self({"penuki"}, actor=actor, chosen_tiles={"penuki": tile_token})
        self.first_turn_open_calls_seen = True
        # Nuki-dora affects final han, but like other dora it is not itself a yaku.
        # Win-legality checks here intentionally depend on actual yaku only, so we
        # track the extraction in round flow/state but do not feed a nuki count into
        # has_hupai/evaluate_draw.
        self._consume_concealed_token(self.players[actor], tile_token)
        self.players[actor].last_draw_tile = None
        self._invalidate_wait_mask(actor)
        reaction = self._compute_penuki_reaction_options(actor, tile_to_index(tile_token))
        if reaction:
            self.pending_reaction = reaction
            self.tokens.extend(self._build_reaction_option_block(reaction.options_by_player))
        self.expected_discard_actor = None
        self.expected_draw_actor = None if reaction else actor
        self.pending_dead_wall_draw = True

    def _emit_non_winning_reaction_passes_before_result(self, remaining_ron_winners: Set[int] | None = None) -> None:
        if not self.pending_reaction:
            return
        had_ron_winner = any(action == "ron" for action in self.pending_reaction.chosen.values())
        if not had_ron_winner:
            return
        remaining_options_by_player: Dict[int, Set[str]] = {
            seat: set() for seat in self.pending_reaction.options_by_player
        }
        for seat, opt in self._iter_reaction_priority_entries(self.pending_reaction.options_by_player):
            chosen = self.pending_reaction.chosen.get(seat)
            defer_until_ron_recorded = (
                remaining_ron_winners is not None
                and seat in remaining_ron_winners
                and chosen != "ron"
            )
            if defer_until_ron_recorded:
                remaining_options_by_player[seat].add(opt)
                continue
            if opt == "ron":
                if chosen == "ron":
                    remaining_options_by_player[seat].add(opt)
                    continue
                if remaining_ron_winners is not None and seat not in remaining_ron_winners:
                    reason = self._reaction_pass_reason(seat, opt, "voluntary")
                    self._emit_reaction_pass(seat, opt, reason)
                    continue
                remaining_options_by_player[seat].add(opt)
                continue
            if chosen is not None:
                self._emit_reaction_pass(seat, opt, "voluntary")
                continue
            reason = self._reaction_pass_reason(seat, opt, "voluntary")
            self._emit_reaction_pass(seat, opt, reason)

        for seat in self.pending_reaction.options_by_player:
            self.pending_reaction.options_by_player[seat] = remaining_options_by_player[seat]

    def _finish_hule_result(self, deltas: List[int]) -> None:
        self.tokens.extend(self._build_score_delta_block(deltas))
        self.tokens.extend(self._build_rank_block(self._compute_round_rank_places()))

    def _on_hule(
        self,
        h: dict,
        *,
        more_ron_expected: bool = False,
        remaining_ron_winners: Set[int] | None = None,
    ) -> None:
        self._require_round_initialized()
        h = self._require_dict(h, field="hule")
        winner = self._require_seat(h["l"], field="hule.l")
        baojia = h.get("baojia")
        if baojia is not None:
            baojia = self._require_seat(baojia, field="hule.baojia")
        fenpei = self._require_seat_list(h["fenpei"], field="hule.fenpei")

        if baojia is not None:
            if self.pending_reaction is not None:
                if baojia != self.pending_reaction.discarder:
                    raise TokenizeError("ron requires a pending matching reaction")
                winners = remaining_ron_winners or {winner}
                if len(winners) >= 3:
                    raise TokenizeError("triple ron should be represented as pingju_sanchahou")
                for ron_winner in sorted(winners):
                    offered = self.pending_reaction.options_by_player.get(ron_winner, set())
                    if "ron" not in offered:
                        raise TokenizeError("ron action was not offered")
                    self.pending_reaction.chosen[ron_winner] = "ron"
                self._finalize_reaction()
                if more_ron_expected:
                    self.pending_multi_ron_baojia = baojia
                    self.pending_multi_ron_remaining = set(winners)
                    self.pending_multi_ron_deltas = [0] * self.seat_count
            elif self.pending_multi_ron_baojia != baojia:
                raise TokenizeError("ron requires a pending matching reaction")
            if self.pending_multi_ron_baojia is not None:
                if winner not in self.pending_multi_ron_remaining:
                    raise TokenizeError("unexpected multi-ron winner")
                self.pending_multi_ron_remaining.remove(winner)
        else:
            if not self.pending_self or self.pending_self.actor != winner:
                raise TokenizeError("tsumo requires a pending self decision")
            if "tsumo" not in self.pending_self.options:
                raise TokenizeError("tsumo action was not offered")
            chosen_tiles: Dict[str, str] = {}
            shoupai = h.get("shoupai")
            if shoupai is not None:
                if not isinstance(shoupai, str):
                    raise TokenizeError("hule.shoupai must be a string")
                winning_tile = _hule_winning_tile_token(shoupai)
                if winning_tile is not None:
                    self.pending_self.option_tiles["tsumo"] = [winning_tile]
                    chosen_tiles["tsumo"] = winning_tile
            self._finalize_self({"tsumo"}, actor=winner, chosen_tiles=chosen_tiles)

        detail_tokens = [f"hule_{winner}", *self._build_hule_detail_block(h, winner=winner)]
        deltas = [self._require_score(delta, field=f"hule.fenpei[{seat}]") for seat, delta in enumerate(fenpei)]
        self.tokens.extend(detail_tokens)
        for seat in range(self.seat_count):
            self.players[seat].score += deltas[seat]
        if self.pending_multi_ron_baojia is not None:
            if self.pending_multi_ron_deltas is None:
                raise TokenizeError("missing multi-ron delta accumulator")
            for seat, delta in enumerate(deltas):
                self.pending_multi_ron_deltas[seat] += delta
            if not self.pending_multi_ron_remaining:
                total_deltas = self.pending_multi_ron_deltas
                self.pending_multi_ron_baojia = None
                self.pending_multi_ron_remaining = set()
                self.pending_multi_ron_deltas = None
                self._finish_hule_result(total_deltas)
        else:
            self._finish_hule_result(deltas)

    def _on_pingju(self, p: dict) -> None:
        self._require_round_initialized()
        p = self._require_dict(p, field="pingju")
        name = p.get("name", "unknown")
        if not isinstance(name, str):
            raise TokenizeError("pingju.name must be a string")
        fenpei = self._require_seat_list(p["fenpei"], field="pingju.fenpei")
        normalized_name = self._normalize_pingju_name(name)
        kyushukyuhai_actor = None
        if name == "九種九牌":
            kyushukyuhai_actor = self._kyushukyuhai_actor(p.get("shoupai"))
        if self.pending_reaction:
            if normalized_name == "sanchahou":
                ron_seats = [
                    seat
                    for seat, opts in sorted(self.pending_reaction.options_by_player.items())
                    if "ron" in opts
                ]
                if len(ron_seats) != 3:
                    raise TokenizeError("sanchahou requires exactly three ron takers")
                for seat in ron_seats:
                    self.pending_reaction.chosen[seat] = "ron"
            self._finalize_reaction(close_reason="voluntary")
        if name == "九種九牌" and kyushukyuhai_actor is not None:
            self._finalize_self({"kyushukyuhai"}, actor=kyushukyuhai_actor)
        self.tokens.append(f"pingju_{normalized_name}")
        self.tokens.extend(self._build_pingju_opened_hand_block(p, kyushukyuhai_actor=kyushukyuhai_actor))
        deltas = [self._require_score(delta, field=f"pingju.fenpei[{seat}]") for seat, delta in enumerate(fenpei)]
        for seat in range(self.seat_count):
            self.players[seat].score += deltas[seat]
        self.tokens.extend(self._build_score_delta_block(deltas))
        self.tokens.extend(self._build_rank_block(self._compute_round_rank_places()))

    def _kyushukyuhai_actor(self, shoupai: object) -> Optional[int]:
        if self.pending_self and "kyushukyuhai" in self.pending_self.options:
            return self.pending_self.actor
        if not isinstance(shoupai, list):
            return None
        for seat, hand in enumerate(shoupai):
            if isinstance(hand, str) and hand:
                return seat
        return None

    def _normalize_pingju_name(self, text: str) -> str:
        mapping = {
            "流局": "ryukyoku",
            "九種九牌": "kyushukyuhai",
            "流し満貫": "nagashimangan",
            "四風連打": "sufurenda",
            "四槓散了": "sukantsu",
            "四家立直": "suuchariichi",
            "三家和了": "sanchahou",
        }
        if text in mapping:
            return mapping[text]
        raise TokenizeError(f"unknown pingju.name: {text}")

    def _build_pingju_opened_hand_block(self, p: dict, *, kyushukyuhai_actor: int | None = None) -> List[str]:
        name = p.get("name")
        if name in {"流し満貫", "四風連打", "四槓散了"}:
            return []
        shoupai = p.get("shoupai")
        if shoupai is None:
            return []
        if not isinstance(shoupai, list):
            raise TokenizeError("pingju.shoupai must be a list")
        block: List[str] = []
        for seat, hand in enumerate(shoupai):
            # Converter variants may use false/null for unrevealed hands in pingju.shoupai.
            if hand in (None, False):
                continue
            if not isinstance(hand, str):
                raise TokenizeError(f"pingju.shoupai[{seat}] must be a string")
            if kyushukyuhai_actor is not None and seat != kyushukyuhai_actor:
                continue
            block.extend(_opened_hand_tokens(seat, hand))
        return block

    def _build_hule_detail_block(self, h: dict, *, winner: int) -> List[str]:
        block: List[str] = []
        shoupai = h.get("shoupai")
        if shoupai is not None:
            if not isinstance(shoupai, str):
                raise TokenizeError("hule.shoupai must be a string")
            block.extend(_opened_hule_hand_tokens(winner, shoupai))
        hupai = h.get("hupai")
        has_riichi_yaku = False
        if isinstance(hupai, list):
            for idx, entry in enumerate(hupai):
                if not isinstance(entry, dict):
                    raise TokenizeError(f"hule.hupai[{idx}] must be a dict")
                name = entry.get("name")
                if isinstance(name, str) and name in RIICHI_HUPAI_NAMES:
                    has_riichi_yaku = True
                    break
        fubaopai = h.get("fubaopai")
        if fubaopai is not None:
            if not isinstance(fubaopai, list):
                raise TokenizeError("hule.fubaopai must be a list")
            if has_riichi_yaku:
                block.append("ura_dora")
                for idx, tile in enumerate(fubaopai):
                    block.append(token_tile(_strip_tile_suffix(self._require_str(tile, field=f"hule.fubaopai[{idx}]"))))
        if hupai is not None:
            if not isinstance(hupai, list):
                raise TokenizeError("hule.hupai must be a list")
            for idx, entry in enumerate(hupai):
                if not isinstance(entry, dict):
                    raise TokenizeError(f"hule.hupai[{idx}] must be a dict")
                fanshu_value = entry.get("fanshu")
                if isinstance(fanshu_value, int) and fanshu_value <= 0:
                    continue
                name = entry.get("name")
                if not isinstance(name, str):
                    raise TokenizeError(f"hule.hupai[{idx}].name must be a string")
                if name == "":
                    # The bundled Tenhou converter still emits a blank placeholder for one yaku slot.
                    continue
                token = HUPAI_TOKEN_BY_NAME.get(name)
                if token is None:
                    raise TokenizeError(f"unknown hule.hupai name: {name}")
                repeat = 1
                if name in DORA_HUPAI_NAMES and isinstance(fanshu_value, int):
                    repeat = fanshu_value
                block.extend([token] * repeat)

        damanguan = h.get("damanguan")
        if damanguan is not None:
            if isinstance(damanguan, bool) or not isinstance(damanguan, int) or damanguan <= 0:
                raise TokenizeError("hule.damanguan must be a positive integer")
            block.append(f"yakuman_{damanguan}")
            return block

        fanshu = h.get("fanshu")
        if fanshu is not None:
            if isinstance(fanshu, bool) or not isinstance(fanshu, int) or fanshu <= 0:
                raise TokenizeError("hule.fanshu must be a positive integer")
            if fanshu > 13:
                fanshu = 13
            block.append(f"han_{fanshu}")

        fu = h.get("fu")
        if fu is not None:
            if isinstance(fu, bool) or not isinstance(fu, int) or fu <= 0:
                raise TokenizeError("hule.fu must be a positive integer")
            if fu == 25:
                block.append("fu_25")
                return block
            if fu < 20 or fu % 10 != 0 or fu > 140:
                raise TokenizeError("hule.fu must be 25 or a multiple of 10 between 20 and 140")
            block.append(f"fu_{fu}")

        return block

    def _finalize_self(
        self,
        chosen: Set[str],
        actor: Optional[int] = None,
        chosen_tiles: Optional[Dict[str, str]] = None,
    ) -> None:
        if not self.pending_self:
            return
        if actor is not None and actor != self.pending_self.actor:
            self.pending_self = None
            return

        seat = self.pending_self.actor
        chosen_effective = chosen & self.pending_self.options
        skipped_tsumo_for_penuki = "penuki" in chosen_effective
        if (
            self.players[seat].is_riichi
            and "tsumo" in self.pending_self.options
            and "tsumo" not in chosen_effective
            and not skipped_tsumo_for_penuki
        ):
            self.players[seat].riichi_furiten = True
        for opt in self._self_option_order(self.pending_self.options):
            if opt in chosen_effective:
                if opt in self.pending_self.option_tiles:
                    chosen_tile = self._resolve_chosen_self_tile(opt, chosen_tiles or {})
                    self.tokens.extend(
                        self._build_self_action_block(
                            seat=seat,
                            kind="take",
                            opt=opt,
                            tile_token=chosen_tile if self._self_action_reveals_tile(opt, kind="take") else None,
                        )
                    )
                    continue
                self.tokens.extend(self._build_self_action_block(seat=seat, kind="take", opt=opt))
                continue
            if opt in self.pending_self.option_tiles:
                if self._self_action_reveals_tile(opt, kind="pass"):
                    for tile_token in self.pending_self.option_tiles[opt]:
                        self.tokens.extend(
                            self._build_self_action_block(
                                seat=seat,
                                kind="pass",
                                opt=opt,
                                tile_token=tile_token,
                            )
                        )
                else:
                    self.tokens.extend(self._build_self_action_block(seat=seat, kind="pass", opt=opt))
                continue
            self.tokens.extend(self._build_self_action_block(seat=seat, kind="pass", opt=opt))
        self.pending_self = None

    def _self_option_tiles(self, actor: int, drawn_tile: int) -> Dict[str, List[str]]:
        option_tiles: Dict[str, List[str]] = {}
        option_tiles["tsumo"] = [index_to_tile(drawn_tile)]
        if self.seat_count == 3 and self.players[actor].concealed[tile_to_index("z4")] > 0:
            option_tiles["penuki"] = ["z4"]
        ankan_tiles = self._ankan_candidate_tiles(actor, drawn_tile=drawn_tile)
        if ankan_tiles:
            option_tiles["ankan"] = ankan_tiles
        kakan_tiles = self._kakan_candidate_tiles(actor)
        if kakan_tiles:
            option_tiles["kakan"] = kakan_tiles
        return option_tiles

    def _emit_self_options(self, actor: int, options: Set[str], option_tiles: Dict[str, List[str]]) -> None:
        for opt in self._self_option_order(options):
            if opt in option_tiles and self._self_action_reveals_tile(opt, kind="opt"):
                self.tokens.extend(self._build_self_action_block(seat=actor, kind="opt", opt=opt))
                for tile_token in option_tiles[opt]:
                    self.tokens.append(tile_token)
                continue
            self.tokens.extend(self._build_self_action_block(seat=actor, kind="opt", opt=opt))

    def _self_action_reveals_tile(self, opt: str, *, kind: str) -> bool:
        if opt in {"ankan", "kakan", "tsumo"}:
            return kind == "take"
        return False

    def _ankan_candidate_tiles(self, actor: int, drawn_tile: Optional[int] = None) -> List[str]:
        p = self.players[actor]
        candidates = [tile for tile, count in enumerate(p.concealed) if count >= 4]
        if not candidates:
            return []
        if not p.is_riichi:
            return [index_to_tile(tile) for tile in candidates]

        if drawn_tile is None:
            return []
        candidates = [tile for tile in candidates if tile == drawn_tile]
        if not candidates:
            return []

        pre_draw_counts = list(p.concealed)
        if pre_draw_counts[drawn_tile] <= 0:
            return []
        pre_draw_counts[drawn_tile] -= 1

        waits_before_mask = _pm_wait_mask(pre_draw_counts, p.meld_count, self.seat_count == 3)
        if waits_before_mask == 0:
            return []

        allowed: List[str] = []
        for tile in candidates:
            next_counts = list(p.concealed)
            next_counts[tile] -= 4
            waits_after_mask = _pm_wait_mask(next_counts, p.meld_count + 1, self.seat_count == 3)
            if waits_after_mask != 0 and waits_after_mask == waits_before_mask:
                allowed.append(index_to_tile(tile))
        return allowed

    def _kakan_candidate_tiles(self, actor: int) -> List[str]:
        p = self.players[actor]
        return [index_to_tile(tile) for tile in sorted(p.open_pons) if p.concealed[tile] > 0]

    def _resolve_chosen_self_tile(self, opt: str, chosen_tiles: Dict[str, str]) -> str:
        if not self.pending_self:
            raise TokenizeError("self decision is missing")
        candidates = self.pending_self.option_tiles.get(opt, [])
        if not candidates:
            raise TokenizeError(f"no tile-qualified candidates for self action: {opt}")
        chosen_tile = chosen_tiles.get(opt)
        if chosen_tile is None:
            if len(candidates) == 1:
                return candidates[0]
            raise TokenizeError(f"self action {opt} requires an explicit tile choice")
        if chosen_tile not in candidates:
            raise TokenizeError(f"self action {opt} tile was not offered: {chosen_tile}")
        return chosen_tile

    def _reaction_pass_reason(self, seat: int, opt: str, close_reason: str) -> str:
        if close_reason == "forced_rule":
            raise TokenizeError("forced_rule reaction close is not valid for Tenhou logs")
        if not self.pending_reaction:
            return "voluntary"

        chosen_other_actions = [
            action
            for other_seat, action in self.pending_reaction.chosen.items()
            if other_seat != seat
        ]
        if not chosen_other_actions:
            return "voluntary"

        if opt == "chi":
            if any(action in {"ron", "pon", "minkan"} for action in chosen_other_actions):
                return "forced_priority"
            return "voluntary"

        if opt in {"pon", "minkan"}:
            if any(action in {"ron", "pon", "minkan"} for action in chosen_other_actions):
                return "forced_priority"
            return "voluntary"

        if opt == "ron":
            # Do not infer atamahane/room-rule restrictions without explicit rule metadata.
            return "voluntary"

        return "voluntary"

    def _emit_reaction_pass(self, seat: int, opt: str, reason: str) -> None:
        self.tokens.extend(
            self._build_reaction_action_block(
                seat=seat,
                kind="pass",
                opt=opt,
                reason=reason,
            )
        )

    def _finalize_reaction(
        self,
        close_reason: str = "voluntary",
        chosen_details: Optional[Dict[Tuple[int, str], List[str]]] = None,
    ) -> None:
        if not self.pending_reaction:
            return

        had_ron_winner = any(action == "ron" for action in self.pending_reaction.chosen.values())
        had_call_winner = any(action in {"chi", "pon", "minkan"} for action in self.pending_reaction.chosen.values())
        if (
            self.pending_riichi_actor is not None
            and self.pending_reaction.trigger == "discard"
            and self.pending_riichi_actor == self.pending_reaction.discarder
        ):
            if not had_ron_winner:
                self._apply_riichi_stick(self.pending_riichi_actor)
            self.pending_riichi_actor = None

        for seat, opt in self._iter_reaction_priority_entries(self.pending_reaction.options_by_player):
            opts = self.pending_reaction.options_by_player.get(seat, set())
            chosen = self.pending_reaction.chosen.get(seat)
            if chosen and chosen in opts and chosen == opt:
                if seat not in self.pending_reaction.emitted_chosen:
                    self.tokens.extend(
                        self._build_reaction_action_block(
                            seat=seat,
                            kind="take",
                            opt=chosen,
                        )
                    )
                    if chosen_details is not None:
                        self.tokens.extend(chosen_details.get((seat, chosen), []))
                    self.pending_reaction.emitted_chosen.add(seat)
                continue

            if chosen and chosen in opts:
                self._emit_reaction_pass(seat, opt, "voluntary")
                continue

            reason = self._reaction_pass_reason(seat, opt, close_reason)
            self._emit_reaction_pass(seat, opt, reason)

        for seat, opts in sorted(self.pending_reaction.options_by_player.items()):
            chosen = self.pending_reaction.chosen.get(seat)
            if "ron" in opts and chosen != "ron":
                ron_reason = "voluntary"
                if not (chosen and chosen in opts):
                    ron_reason = self._reaction_pass_reason(seat, "ron", close_reason)
                chose_non_ron_call = chosen in {"chi", "pon", "minkan"}
                if ron_reason == "voluntary" and not chose_non_ron_call:
                    if self.players[seat].is_riichi:
                        self.players[seat].riichi_furiten = True
                    else:
                        self.players[seat].temporary_furiten = True
        if not had_ron_winner and not had_call_winner:
            self.expected_discard_actor = None
            self.expected_draw_actor = self.pending_reaction.discarder
            if self.pending_reaction.trigger == "discard":
                self.expected_draw_actor = (self.pending_reaction.discarder + 1) % self.seat_count
        self.pending_reaction = None

    def _is_reaction_continuation(self, key: str, value: dict) -> bool:
        if not self.pending_reaction:
            return False
        # kaigang can appear between discard and the actual fulou/hule reaction.
        # It is a dora reveal side event and should not close reaction windows.
        if key == "kaigang":
            return True
        if not isinstance(value, dict):
            return False
        if key == "fulou" and self.pending_reaction.trigger == "discard":
            return True
        if key == "hule" and value.get("baojia") == self.pending_reaction.discarder:
            return True
        return False

    def _is_self_resolution(self, key: str, value: dict) -> bool:
        if not self.pending_self:
            return False
        actor = self.pending_self.actor
        if key == "kaigang":
            return True
        if not isinstance(value, dict):
            return False
        if key == "penuki" and value.get("l") == actor:
            return True
        if key == "dapai" and value.get("l") == actor:
            return True
        if key == "gang" and value.get("l") == actor:
            return True
        if key == "hule" and value.get("l") == actor and value.get("baojia") is None:
            return True
        if key == "pingju" and value.get("name") == "九種九牌":
            return True
        return False


def iter_tokenized_games(
    zip_path: str,
    max_games: Optional[int] = None,
    start_index: int = 0,
) -> Iterable[Tuple[str, List[str]]]:
    import json
    import zipfile

    tokenizer = TenhouTokenizer()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        produced = 0
        for idx, name in enumerate(names):
            if idx < start_index:
                continue
            if max_games is not None and produced >= max_games:
                break

            with zf.open(name) as f:
                game = json.load(f)
            tokens = tokenizer.tokenize_game(game)
            produced += 1
            yield name, tokens
