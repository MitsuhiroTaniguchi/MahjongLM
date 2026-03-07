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
    (5000, "TENBO_5000"),
    (1000, "TENBO_1000"),
    (100, "TENBO_100"),
)


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
    tokens = [sign]
    for unit_value, unit_token in TENBO_UNITS:
        count, remaining = divmod(remaining, unit_value)
        if count:
            tokens.extend([unit_token] * count)
    if remaining != 0:
        raise TokenizeError(f"score value cannot be represented by tenbo units: {value}")
    return tokens


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


def _pm_xiangting(counts: List[int], meld_count: int) -> int:
    x, _mode, _disc, _wait = PM_XIANGTING.calculate(tuple(counts), 4 - meld_count, 7, False, False)
    return int(x)


def _pm_wait_mask(counts: List[int], meld_count: int) -> int:
    if PM_FASTAPI_AVAILABLE:
        return int(pm.wait_mask(tuple(counts), int(meld_count)))
    x, _mode, _disc, wait = PM_XIANGTING.calculate(tuple(counts), 4 - meld_count, 7, False, False)
    if int(x) != 0:
        return 0
    return int(wait)


def _pm_wait_tiles(counts: List[int], meld_count: int) -> Set[int]:
    wait_mask = _pm_wait_mask(counts, meld_count)
    if wait_mask == 0:
        return set()
    return {i for i in range(34) if (wait_mask >> i) & 1}


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


def _pm_has_riichi_discard(counts: List[int], meld_count: int) -> bool:
    if PM_FASTAPI_AVAILABLE:
        return bool(pm.has_riichi_discard(tuple(counts), int(meld_count)))

    base = list(counts)
    for i in range(34):
        if base[i] == 0:
            continue
        base[i] -= 1
        if _pm_xiangting(base, meld_count) == 0:
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
) -> bool:
    pm_melds = encoded_melds if encoded_melds is not None else _encode_pm_melds(melds)
    if PM_FASTAPI_AVAILABLE:
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
    option = pm.HuleOption(int(zhuangfeng), int(lunban))
    option.is_menqian = bool(is_menqian)
    option.is_lizhi = bool(is_riichi)
    option.is_haidi = bool(is_haidi)
    option.is_lingshang = bool(is_lingshang)
    option.is_qianggang = bool(is_qianggang)
    action_type = pm.ActionType.zimohu if is_tsumo else pm.ActionType.ronghu
    action = pm.Action(action_type, int(win_tile))
    hule = pm.Hule(shoupai, action, option)
    return bool(hule.has_hupai)


def _pm_has_hupai_multi(
    cases: List[
        Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int, bool, bool, bool]
    ],
) -> List[bool]:
    if not cases:
        return []
    if PM_MULTI_HUPAI_AVAILABLE:
        encoded_cases_with_context: List[
            Tuple[Tuple[int, ...], List[Tuple[int, int]], int, bool, bool, bool, int, int, bool, bool, bool]
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
            win_tile=win_tile,
            is_tsumo=is_tsumo,
            is_menqian=is_menqian,
            is_riichi=is_riichi,
            zhuangfeng=zhuangfeng,
            lunban=lunban,
            is_haidi=is_haidi,
            is_lingshang=is_lingshang,
            is_qianggang=is_qianggang,
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
) -> Tuple[bool, bool]:
    pm_melds = encoded_melds if encoded_melds is not None else _encode_pm_melds(melds)
    if PM_EVALUATE_DRAW_AVAILABLE:
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
    )
    can_riichi_discard = (
        _pm_has_riichi_discard(counts, closed_kans) if check_riichi_discard else False
    )
    return can_tsumo, can_riichi_discard


@dataclass
class PlayerState:
    concealed: List[int]
    score: int
    red_fives: Dict[str, int] = field(default_factory=lambda: {"m": 0, "p": 0, "s": 0})
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


@dataclass
class ReactionDecision:
    discarder: int
    discard_tile: int
    options_by_player: Dict[int, Set[str]]
    trigger: str = "discard"
    chosen: Dict[int, str] = field(default_factory=dict)


class TenhouTokenizer:
    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.players: List[PlayerState] = []
        self.pending_self: Optional[SelfDecision] = None
        self.pending_reaction: Optional[ReactionDecision] = None
        self.pending_riichi_actor: Optional[int] = None
        self.round_index = 0
        self.live_draws_left = 70
        self.bakaze = 0
        self.dealer_seat = 0
        self.first_turn_open_calls_seen = False
        self.last_draw_was_gangzimo = False
        self.expected_draw_actor: Optional[int] = None
        self.expected_discard_actor: Optional[int] = None
        self.awaiting_kaigang = False

    def tokenize_game(self, game: dict) -> List[str]:
        if not isinstance(game, dict):
            raise TokenizeError("game must be a dict")
        log = game.get("log", [])
        if not isinstance(log, list):
            raise TokenizeError("game log must be a list")
        if not log:
            raise TokenizeError("game log cannot be empty")
        self.tokens = ["game_start"]
        self.pending_self = None
        self.pending_reaction = None
        self.pending_riichi_actor = None
        self.round_index = 0
        self.expected_draw_actor = None
        self.expected_discard_actor = None
        self.awaiting_kaigang = False
        for round_data in log:
            self._process_round(round_data)
        self._flush_pending()
        self.tokens.append("game_end")
        return self.tokens

    def _require_round_initialized(self) -> None:
        if len(self.players) != 4:
            raise TokenizeError("round state is not initialized")

    def _require_int(self, value: object, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TokenizeError(f"{field} must be an integer")
        return value

    def _require_seat(self, seat: object, *, field: str) -> int:
        seat_int = self._require_int(seat, field=field)
        if seat_int < 0 or seat_int >= 4:
            raise TokenizeError(f"invalid seat for {field}: {seat}")
        return seat_int

    def _require_four(self, values: object, *, field: str) -> List[object]:
        if not isinstance(values, list) or len(values) != 4:
            raise TokenizeError(f"{field} must be a list of length 4")
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

    def _process_round(self, round_data: list) -> None:
        if not isinstance(round_data, list):
            raise TokenizeError("round data must be a list")
        if not round_data:
            raise TokenizeError("round data cannot be empty")
        self.pending_self = None
        self.pending_reaction = None
        self.pending_riichi_actor = None
        self.expected_draw_actor = None
        self.expected_discard_actor = None
        self.awaiting_kaigang = False
        saw_qipai = False
        round_ended = False

        for event in round_data:
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
                    and self.pending_reaction is not None
                    and value.get("baojia") == self.pending_reaction.discarder
                )
                if not is_multi_ron_continuation:
                    raise TokenizeError("round already ended")

            if self.pending_reaction and not self._is_reaction_continuation(key, value):
                close_reason = "forced_rule" if key == "pingju" else "voluntary"
                self._finalize_reaction(close_reason=close_reason)

            if self.pending_self and not self._is_self_resolution(key, value):
                self._finalize_self(set())

            if key == "qipai":
                self._on_qipai(value)
            elif key == "zimo":
                self._on_draw(value, is_gangzimo=False)
            elif key == "gangzimo":
                self._on_draw(value, is_gangzimo=True)
            elif key == "dapai":
                self._on_discard(value)
            elif key == "fulou":
                self._on_fulou(value)
            elif key == "gang":
                self._on_gang(value)
            elif key == "kaigang":
                self._on_kaigang(value)
            elif key == "hule":
                self._on_hule(value)
            elif key == "pingju":
                self._on_pingju(value)
            else:
                self.tokens.append(f"event_unknown_{key}")

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

        five_suits = [t[0] for t in consumed_tokens if t[0] in {"m", "p", "s"} and t[1] in {"0", "5"}]
        if not five_suits:
            return None
        suit = five_suits[0]
        if any(s != suit for s in five_suits):
            return None

        consumed_fives = len(five_suits)
        if consumed_fives == 0:
            return None

        idx = tile_to_index(f"{suit}5")
        total_fives = pre_counts[idx]
        red_fives = pre_red_fives.get(suit, 0)
        normal_fives = total_fives - red_fives
        min_red_used = max(0, consumed_fives - normal_fives)
        max_red_used = min(consumed_fives, red_fives)
        if min_red_used == max_red_used:
            return None

        used_red = any(t == f"{suit}0" for t in consumed_tokens)
        return f"red_{action}_{'used' if used_red else 'not_used'}"

    def _player_wind(self, seat: int) -> int:
        return (seat - self.dealer_seat) % 4

    def _invalidate_wait_mask(self, seat: int) -> None:
        self.players[seat].wait_mask_cache = None

    def _invalidate_meld_cache(self, seat: int) -> None:
        self.players[seat].encoded_melds_cache = None

    def _player_encoded_melds(self, seat: int) -> List[Tuple[int, int]]:
        p = self.players[seat]
        if p.encoded_melds_cache is None:
            p.encoded_melds_cache = _encode_pm_melds(p.melds)
        return p.encoded_melds_cache

    def _wait_mask(self, seat: int) -> int:
        p = self.players[seat]
        if p.wait_mask_cache is None:
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
        )

    def _has_riichi_discard(self, seat: int) -> bool:
        p = self.players[seat]
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
        return _pm_evaluate_draw(
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
        shoupai = self._require_four(q["shoupai"], field="qipai.shoupai")
        defen = self._require_four(q["defen"], field="qipai.defen")
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
            )
            for seat in range(4)
        ]
        bakaze = self._require_non_negative_int(q["zhuangfeng"], field="qipai.zhuangfeng")
        kyoku = self._require_non_negative_int(q["jushu"], field="qipai.jushu")
        honba = self._require_non_negative_int(q["changbang"], field="qipai.changbang")
        riichi_sticks = self._require_non_negative_int(q["lizhibang"], field="qipai.lizhibang")

        self.live_draws_left = 70
        self.bakaze = bakaze
        # Tenhou JSON used here is seat-rotated so dealer is always seat 0.
        self.dealer_seat = 0
        self.first_turn_open_calls_seen = False
        self.last_draw_was_gangzimo = False
        self.expected_draw_actor = None
        self.expected_discard_actor = None
        self.awaiting_kaigang = False

        self.tokens.append("round_start")
        self.tokens.append(f"round_seq_{self.round_index}")
        self.tokens.append(f"bakaze_{bakaze}")
        self.tokens.append(f"kyoku_{kyoku}")
        self.tokens.append(f"honba_{honba}")
        self.tokens.append(f"riichi_sticks_{riichi_sticks}")
        self.tokens.append(f"dora_{_strip_tile_suffix(baopai).replace('0', '5')}")

        for seat, score in enumerate(scores):
            self.tokens.append(f"score_{seat}")
            self.tokens.extend(encode_tenbo_tokens(score))

        for seat, hand in enumerate(shoupai):
            hand_tiles = _parse_tiles(hand, stop_at_comma=True, context="hand")
            for tile in sorted(hand_tiles, key=token_tile_sort_key):
                self.tokens.append(f"haipai_{seat}_{tile}")

    def _on_draw(self, z: dict, is_gangzimo: bool) -> None:
        self._require_round_initialized()
        z = self._require_dict(z, field="draw")
        if self.live_draws_left <= 0:
            raise TokenizeError("no live draws remaining")
        if self.awaiting_kaigang and not is_gangzimo:
            raise TokenizeError("kaigang must occur before the next live draw")
        if self.expected_discard_actor is not None:
            raise TokenizeError("draw is not allowed before discard resolution")
        actor = self._require_seat(z["l"], field="draw.l")
        if self.expected_draw_actor is not None and actor != self.expected_draw_actor:
            raise TokenizeError(f"unexpected draw actor: {actor}")
        tile_str = _strip_tile_suffix(self._require_str(z["p"], field="draw.p"))
        tile_token = token_tile(tile_str)
        tile_idx = self._add_concealed_token(self.players[actor], tile_token)
        self.players[actor].temporary_furiten = False
        self._invalidate_wait_mask(actor)
        self.live_draws_left -= 1
        self.last_draw_was_gangzimo = is_gangzimo

        action = "gang_draw" if is_gangzimo else "draw"
        self.tokens.append(f"{action}_{actor}_{tile_token}")

        options = self._compute_self_options(actor, tile_idx, is_gangzimo=is_gangzimo)
        for opt in sorted(options):
            self.tokens.append(f"opt_self_{actor}_{opt}")
        self.pending_self = SelfDecision(actor=actor, options=options)
        self.players[actor].is_first_turn = False
        self.expected_draw_actor = None
        self.expected_discard_actor = actor

    def _compute_self_options(self, actor: int, drawn_tile: int, is_gangzimo: bool = False) -> Set[str]:
        p = self.players[actor]
        options: Set[str] = set()
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
        p = self.players[actor]
        candidates = [tile for tile, count in enumerate(p.concealed) if count >= 4]
        if not candidates:
            return False
        if not p.is_riichi:
            return True

        if drawn_tile is None:
            return False
        # Disallow okuri-kan after riichi: only a quad including the drawn tile can be declared.
        candidates = [tile for tile in candidates if tile == drawn_tile]
        if not candidates:
            return False

        pre_draw_counts = list(p.concealed)
        if pre_draw_counts[drawn_tile] <= 0:
            return False
        pre_draw_counts[drawn_tile] -= 1

        waits_before_mask = _pm_wait_mask(pre_draw_counts, p.meld_count)
        if waits_before_mask == 0:
            return False

        for tile in candidates:
            next_counts = list(p.concealed)
            next_counts[tile] -= 4
            waits_after_mask = _pm_wait_mask(next_counts, p.meld_count + 1)
            if waits_after_mask != 0 and waits_after_mask == waits_before_mask:
                return True
        return False

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
        self.players[actor].furiten_tiles.add(tile_idx)
        self._invalidate_wait_mask(actor)

        suffix = "_tsumogiri" if is_tsumogiri else ""
        self.tokens.append(f"discard_{actor}_{tile_token}{suffix}")

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
            for seat, opts in sorted(reaction.options_by_player.items()):
                for opt in sorted(opts):
                    self.tokens.append(f"opt_react_{seat}_{opt}")
        elif is_riichi:
            self._apply_riichi_stick(actor)
            self.expected_draw_actor = (actor + 1) % 4
        else:
            self.expected_draw_actor = (actor + 1) % 4

    def _compute_reaction_options(self, discarder: int, tile_idx: int) -> Optional[ReactionDecision]:
        options_by_player: Dict[int, Set[str]] = {}
        ron_cases: List[
            Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int, bool, bool, bool]
        ] = []
        ron_case_seats: List[int] = []
        is_haidi = self.live_draws_left == 0 and not self.last_draw_was_gangzimo

        for offset in range(1, 4):
            seat = (discarder + offset) % 4
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
                        )
                    )
                    ron_case_seats.append(seat)

        ron_seats: Set[int] = set()
        if ron_cases:
            for seat, can_ron in zip(ron_case_seats, _pm_has_hupai_multi(ron_cases)):
                if can_ron:
                    ron_seats.add(seat)

        for offset in range(1, 4):
            seat = (discarder + offset) % 4
            p = self.players[seat]
            options: Set[str] = set()
            if seat in ron_seats:
                options.add("ron")
            if p.is_riichi:
                if options:
                    options_by_player[seat] = options
                continue

            if self.live_draws_left > 0:
                if offset == 1 and self._can_chi(p.concealed, tile_idx):
                    options.add("chi")
                if p.concealed[tile_idx] >= 2:
                    options.add("pon")
                if p.concealed[tile_idx] >= 3:
                    options.add("minkan")

            if options:
                options_by_player[seat] = options

        if not options_by_player:
            return None
        return ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player=options_by_player,
            trigger="discard",
        )

    def _compute_kakan_reaction_options(self, actor: int, tile_idx: int) -> Optional[ReactionDecision]:
        options_by_player: Dict[int, Set[str]] = {}
        ron_cases: List[
            Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int, bool, bool, bool]
        ] = []
        ron_case_seats: List[int] = []

        for seat in range(4):
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
        options_by_player: Dict[int, Set[str]] = {}
        ron_cases: List[
            Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int, bool, bool, bool]
        ] = []
        ron_case_seats: List[int] = []

        for seat in range(4):
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
            if not _is_kokushi_agari_shape(counts_plus, p.meld_count):
                continue

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
            trigger="ankan",
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

        discard_tile = None
        had_pending_reaction = True
        self.pending_reaction.chosen[actor] = action
        discard_tile = self.pending_reaction.discard_tile
        self._finalize_reaction()

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

        if action == "chi":
            anchor = min(meld_tiles)
            p.melds.append(("chi", anchor))
        elif action == "pon":
            tile = meld_tiles[0]
            p.open_pons[tile] = p.open_pons.get(tile, 0) + 1
            red_in_pon = sum(1 for tile_token in meld_token_tiles if tile_token[1] == "0")
            if red_in_pon:
                p.open_pons_red[tile] = p.open_pons_red.get(tile, 0) + red_in_pon
            p.melds.append(("pon", tile))
        elif action == "minkan":
            tile = meld_tiles[0]
            p.melds.append(("minkan", tile))
        self._invalidate_meld_cache(actor)

        if not had_pending_reaction:
            self.tokens.append(f"take_react_{actor}_{action}")
        if action == "chi":
            chi_pos = self._chi_pos_label(meld_tiles, called_index, discard_tile)
            if chi_pos is not None:
                self.tokens.append(f"chi_pos_{chi_pos}")
        red_choice = self._red_choice_token(
            action=action,
            consumed_tokens=consumed_tokens,
            pre_counts=pre_counts,
            pre_red_fives=pre_red_fives,
        )
        if red_choice is not None:
            self.tokens.append(red_choice)
        if action == "minkan":
            self.awaiting_kaigang = True
            self.expected_draw_actor = actor
            self.expected_discard_actor = None
        else:
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
        self._finalize_self(chosen, actor=actor)

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
        self._invalidate_meld_cache(actor)
        self._invalidate_wait_mask(actor)

        reaction: Optional[ReactionDecision] = None
        if kind == "kakan":
            reaction = self._compute_kakan_reaction_options(actor, tile)
        elif kind == "ankan":
            reaction = self._compute_ankan_reaction_options(actor, tile)
        self.awaiting_kaigang = True
        if reaction:
            self.pending_reaction = reaction
            for seat, opts in sorted(reaction.options_by_player.items()):
                for opt in sorted(opts):
                    self.tokens.append(f"opt_react_{seat}_{opt}")
        self.expected_discard_actor = None
        self.expected_draw_actor = actor

    def _on_kaigang(self, k: dict) -> None:
        if not self.awaiting_kaigang:
            raise TokenizeError("kaigang is not expected")
        k = self._require_dict(k, field="kaigang")
        tile = token_tile(_strip_tile_suffix(self._require_str(k["baopai"], field="kaigang.baopai"))).replace("0", "5")
        self.tokens.append(f"dora_{tile}")
        self.awaiting_kaigang = False

    def _on_hule(self, h: dict) -> None:
        self._require_round_initialized()
        h = self._require_dict(h, field="hule")
        winner = self._require_seat(h["l"], field="hule.l")
        baojia = h.get("baojia")
        if baojia is not None:
            baojia = self._require_seat(baojia, field="hule.baojia")
        fenpei = self._require_four(h["fenpei"], field="hule.fenpei")

        if baojia is not None:
            if not self.pending_reaction or baojia != self.pending_reaction.discarder:
                raise TokenizeError("ron requires a pending matching reaction")
            offered = self.pending_reaction.options_by_player.get(winner, set())
            if "ron" not in offered:
                raise TokenizeError("ron action was not offered")
            self.pending_reaction.chosen[winner] = "ron"
            self.tokens.append(f"ron_from_{winner}_{baojia}")
        else:
            if not self.pending_self or self.pending_self.actor != winner:
                raise TokenizeError("tsumo requires a pending self decision")
            if "tsumo" not in self.pending_self.options:
                raise TokenizeError("tsumo action was not offered")
            self._finalize_self({"tsumo"}, actor=winner)

        deltas = [self._require_score(delta, field=f"hule.fenpei[{seat}]") for seat, delta in enumerate(fenpei)]
        for seat in range(4):
            self.players[seat].score += deltas[seat]
            self.tokens.append(f"score_delta_{seat}")
            self.tokens.extend(encode_tenbo_tokens(deltas[seat]))

    def _on_pingju(self, p: dict) -> None:
        self._require_round_initialized()
        p = self._require_dict(p, field="pingju")
        name = p.get("name", "unknown")
        if not isinstance(name, str):
            raise TokenizeError("pingju.name must be a string")
        fenpei = self._require_four(p["fenpei"], field="pingju.fenpei")
        if name == "九種九牌":
            actor = self._kyushukyuhai_actor(p.get("shoupai"))
            if actor is not None:
                self._finalize_self({"kyushukyuhai"}, actor=actor)
        self.tokens.append(f"draw_{self._normalize_name(name)}")
        deltas = [self._require_score(delta, field=f"pingju.fenpei[{seat}]") for seat, delta in enumerate(fenpei)]
        for seat in range(4):
            self.players[seat].score += deltas[seat]
            self.tokens.append(f"score_delta_{seat}")
            self.tokens.extend(encode_tenbo_tokens(deltas[seat]))

    def _kyushukyuhai_actor(self, shoupai: object) -> Optional[int]:
        if self.pending_self and "kyushukyuhai" in self.pending_self.options:
            return self.pending_self.actor
        if not isinstance(shoupai, list):
            return None
        for seat, hand in enumerate(shoupai):
            if isinstance(hand, str) and hand:
                return seat
        return None

    def _normalize_name(self, text: str) -> str:
        mapping = {
            "流局": "ryukyoku",
            "九種九牌": "kyushukyuhai",
            "流し満貫": "nagashimangan",
            "四風連打": "sufonrenda",
            "四槓散了": "sukantsu",
            "四家立直": "suuchariichi",
            "三家和了": "sanchahou",
        }
        if text in mapping:
            return mapping[text]
        out = []
        for ch in text:
            out.append(ch if ch.isalnum() else "_")
        return "".join(out).strip("_") or "unknown"

    def _finalize_self(self, chosen: Set[str], actor: Optional[int] = None) -> None:
        if not self.pending_self:
            return
        if actor is not None and actor != self.pending_self.actor:
            self.pending_self = None
            return

        seat = self.pending_self.actor
        chosen_effective = chosen & self.pending_self.options
        if self.players[seat].is_riichi and "tsumo" in self.pending_self.options and "tsumo" not in chosen_effective:
            self.players[seat].riichi_furiten = True
        for opt in sorted(chosen_effective):
            self.tokens.append(f"take_self_{seat}_{opt}")
        for opt in sorted(self.pending_self.options - chosen_effective):
            self.tokens.append(f"pass_self_{seat}_{opt}")
        self.pending_self = None

    def _reaction_pass_reason(self, seat: int, opt: str, close_reason: str) -> str:
        if close_reason == "forced_rule":
            return "forced_rule"
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
        self.tokens.append(f"pass_react_{seat}_{opt}_{reason}")

    def _finalize_reaction(self, close_reason: str = "voluntary") -> None:
        if not self.pending_reaction:
            return

        had_ron_winner = any(action == "ron" for action in self.pending_reaction.chosen.values())
        had_call_winner = any(action in {"chi", "pon", "minkan"} for action in self.pending_reaction.chosen.values())
        if (
            self.pending_riichi_actor is not None
            and self.pending_reaction.trigger == "discard"
            and self.pending_riichi_actor == self.pending_reaction.discarder
        ):
            if close_reason != "forced_rule" and not had_ron_winner:
                self._apply_riichi_stick(self.pending_riichi_actor)
            self.pending_riichi_actor = None

        for seat, opts in sorted(self.pending_reaction.options_by_player.items()):
            chosen = self.pending_reaction.chosen.get(seat)
            if chosen and chosen in opts:
                self.tokens.append(f"take_react_{seat}_{chosen}")
                for opt in sorted(opts - {chosen}):
                    self._emit_reaction_pass(seat, opt, "voluntary")
            else:
                for opt in sorted(opts):
                    reason = self._reaction_pass_reason(seat, opt, close_reason)
                    self._emit_reaction_pass(seat, opt, reason)
            if "ron" in opts and chosen != "ron":
                ron_reason = "voluntary"
                if not (chosen and chosen in opts):
                    ron_reason = self._reaction_pass_reason(seat, "ron", close_reason)
                if ron_reason == "voluntary":
                    if self.players[seat].is_riichi:
                        self.players[seat].riichi_furiten = True
                    else:
                        self.players[seat].temporary_furiten = True
        if not had_ron_winner and not had_call_winner:
            self.expected_discard_actor = None
            self.expected_draw_actor = self.pending_reaction.discarder
            if self.pending_reaction.trigger == "discard":
                self.expected_draw_actor = (self.pending_reaction.discarder + 1) % 4
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
