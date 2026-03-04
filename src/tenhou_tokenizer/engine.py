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


class TokenizeError(RuntimeError):
    pass


def _strip_tile_suffix(tile: str) -> str:
    return tile.replace("*", "").replace("_", "")


def _norm_digit(digit: str) -> str:
    return "5" if digit == "0" else digit


def tile_to_index(tile: str) -> int:
    if len(tile) != 2:
        raise TokenizeError(f"invalid tile format: {tile}")
    suit, digit = tile[0], _norm_digit(tile[1])
    if suit not in SUIT_BASE or not digit.isdigit():
        raise TokenizeError(f"invalid tile format: {tile}")
    number = int(digit)
    if suit == "z":
        if number < 1 or number > 7:
            raise TokenizeError(f"invalid honor tile: {tile}")
    else:
        if number < 1 or number > 9:
            raise TokenizeError(f"invalid suited tile: {tile}")
    return SUIT_BASE[suit] + number - 1


def index_to_tile(index: int) -> str:
    return INDEX_TO_TILE[index]


def parse_hand_counts(hand: str) -> List[int]:
    counts = [0] * 34
    suit: Optional[str] = None
    for ch in hand:
        if ch in SUIT_BASE:
            suit = ch
            continue
        if ch == ",":
            break
        if ch in "-=+":
            continue
        if ch.isdigit():
            if suit is None:
                raise TokenizeError(f"digit without suit in hand: {hand}")
            idx = tile_to_index(f"{suit}{_norm_digit(ch)}")
            counts[idx] += 1
    return counts


def parse_meld_tiles(meld: str) -> List[int]:
    tiles: List[int] = []
    suit: Optional[str] = None
    for ch in meld:
        if ch in SUIT_BASE:
            suit = ch
            continue
        if ch in "-=+,":
            continue
        if ch.isdigit():
            if suit is None:
                raise TokenizeError(f"digit without suit in meld: {meld}")
            tiles.append(tile_to_index(f"{suit}{_norm_digit(ch)}"))
    if not tiles:
        raise TokenizeError(f"empty meld parse: {meld}")
    return tiles


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
    win_tile: int,
    is_tsumo: bool,
    is_menqian: bool,
    is_riichi: bool,
    zhuangfeng: int,
    lunban: int,
) -> bool:
    encoded_melds: List[Tuple[int, int]] = [
        (MELD_TYPE_TO_PM_CODE[meld_type], pai_34) for meld_type, pai_34 in melds
    ]
    if PM_FASTAPI_AVAILABLE:
        return bool(
            pm.has_hupai(
                tuple(counts),
                encoded_melds,
                int(win_tile),
                bool(is_tsumo),
                bool(is_menqian),
                bool(is_riichi),
                int(zhuangfeng),
                int(lunban),
            )
        )

    shoupai = _make_pm_shoupai(counts, melds)
    option = pm.HuleOption(int(zhuangfeng), int(lunban))
    option.is_menqian = bool(is_menqian)
    option.is_lizhi = bool(is_riichi)
    action_type = pm.ActionType.zimohu if is_tsumo else pm.ActionType.ronghu
    action = pm.Action(action_type, int(win_tile))
    hule = pm.Hule(shoupai, action, option)
    return bool(hule.has_hupai)


def _pm_has_hupai_multi(
    cases: List[
        Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int]
    ],
) -> List[bool]:
    if not cases:
        return []
    if PM_MULTI_HUPAI_AVAILABLE:
        encoded_cases: List[
            Tuple[Tuple[int, ...], List[Tuple[int, int]], int, bool, bool, bool, int, int]
        ] = []
        for counts, melds, win_tile, is_tsumo, is_menqian, is_riichi, zhuangfeng, lunban in cases:
            encoded_melds = [(MELD_TYPE_TO_PM_CODE[mtype], pai_34) for mtype, pai_34 in melds]
            encoded_cases.append(
                (
                    tuple(counts),
                    encoded_melds,
                    int(win_tile),
                    bool(is_tsumo),
                    bool(is_menqian),
                    bool(is_riichi),
                    int(zhuangfeng),
                    int(lunban),
                )
            )
        return [bool(x) for x in pm.has_hupai_multi(encoded_cases)]

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
        )
        for counts, melds, win_tile, is_tsumo, is_menqian, is_riichi, zhuangfeng, lunban in cases
    ]


def _pm_evaluate_draw(
    counts: List[int],
    melds: List[Tuple[str, int]],
    win_tile: int,
    is_menqian: bool,
    is_riichi: bool,
    zhuangfeng: int,
    lunban: int,
    closed_kans: int,
    check_riichi_discard: bool,
) -> Tuple[bool, bool]:
    encoded_melds: List[Tuple[int, int]] = [
        (MELD_TYPE_TO_PM_CODE[meld_type], pai_34) for meld_type, pai_34 in melds
    ]
    if PM_EVALUATE_DRAW_AVAILABLE:
        can_tsumo, can_riichi_discard = pm.evaluate_draw(
            tuple(counts),
            encoded_melds,
            int(win_tile),
            bool(is_menqian),
            bool(is_riichi),
            int(zhuangfeng),
            int(lunban),
            int(closed_kans),
            bool(check_riichi_discard),
        )
        return bool(can_tsumo), bool(can_riichi_discard)

    can_tsumo = _pm_has_hupai(
        counts=counts,
        melds=melds,
        win_tile=win_tile,
        is_tsumo=True,
        is_menqian=is_menqian,
        is_riichi=is_riichi,
        zhuangfeng=zhuangfeng,
        lunban=lunban,
    )
    can_riichi_discard = (
        _pm_has_riichi_discard(counts, closed_kans) if check_riichi_discard else False
    )
    return can_tsumo, can_riichi_discard


@dataclass
class PlayerState:
    concealed: List[int]
    score: int
    is_riichi: bool = False
    open_melds: int = 0
    closed_kans: int = 0
    open_pons: Dict[int, int] = field(default_factory=dict)
    melds: List[Tuple[str, int]] = field(default_factory=list)
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
        self.round_index = 0
        self.live_draws_left = 70
        self.bakaze = 0
        self.dealer_seat = 0

    def tokenize_game(self, game: dict) -> List[str]:
        self.tokens = ["game_start"]
        self.pending_self = None
        self.pending_reaction = None
        self.round_index = 0
        for round_data in game.get("log", []):
            self._process_round(round_data)
        self._flush_pending()
        self.tokens.append("game_end")
        return self.tokens

    def _process_round(self, round_data: list) -> None:
        self.pending_self = None
        self.pending_reaction = None

        for event in round_data:
            if not event:
                continue
            key, value = next(iter(event.items()))

            if self.pending_reaction and not self._is_reaction_continuation(key, value):
                self._finalize_reaction()

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

        self._flush_pending()
        self.round_index += 1

    def _flush_pending(self) -> None:
        if self.pending_reaction:
            self._finalize_reaction()
        if self.pending_self:
            self._finalize_self(set())

    def _player_wind(self, seat: int) -> int:
        return (seat - self.dealer_seat) % 4

    def _invalidate_wait_mask(self, seat: int) -> None:
        self.players[seat].wait_mask_cache = None

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

    def _evaluate_draw(self, seat: int, drawn_tile: int, check_riichi_discard: bool) -> Tuple[bool, bool]:
        p = self.players[seat]
        return _pm_evaluate_draw(
            counts=p.concealed,
            melds=p.melds,
            win_tile=drawn_tile,
            is_menqian=(p.open_melds == 0),
            is_riichi=p.is_riichi,
            zhuangfeng=self.bakaze,
            lunban=self._player_wind(seat),
            closed_kans=p.closed_kans,
            check_riichi_discard=check_riichi_discard,
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
        hand_counts_list = [parse_hand_counts(hand) for hand in q["shoupai"]]
        self.players = [
            PlayerState(concealed=list(hand_counts_list[seat]), score=q["defen"][seat])
            for seat in range(4)
        ]
        self.live_draws_left = 70
        self.bakaze = int(q["zhuangfeng"])
        # Tenhou JSON used here is seat-rotated so dealer is always seat 0.
        self.dealer_seat = 0

        self.tokens.append("round_start")
        self.tokens.append(f"round_seq_{self.round_index}")
        self.tokens.append(f"bakaze_{q['zhuangfeng']}")
        self.tokens.append(f"kyoku_{q['jushu']}")
        self.tokens.append(f"honba_{q['changbang']}")
        self.tokens.append(f"riichi_sticks_{q['lizhibang']}")
        self.tokens.append(f"dora_{_strip_tile_suffix(q['baopai']).replace('0', '5')}")

        for seat, score in enumerate(q["defen"]):
            self.tokens.append(f"score_{seat}_{score}")

        for seat, hand_counts in enumerate(hand_counts_list):
            for tile_idx, count in enumerate(hand_counts):
                for _ in range(count):
                    self.tokens.append(f"haipai_{seat}_{index_to_tile(tile_idx)}")

    def _on_draw(self, z: dict, is_gangzimo: bool) -> None:
        actor = z["l"]
        tile_str = _strip_tile_suffix(z["p"])
        tile_idx = tile_to_index(tile_str)

        self.players[actor].concealed[tile_idx] += 1
        self.players[actor].temporary_furiten = False
        self._invalidate_wait_mask(actor)
        self.live_draws_left -= 1

        action = "gang_draw" if is_gangzimo else "draw"
        self.tokens.append(f"{action}_{actor}_{index_to_tile(tile_idx)}")

        options = self._compute_self_options(actor, tile_idx)
        for opt in sorted(options):
            self.tokens.append(f"opt_self_{actor}_{opt}")
        self.pending_self = SelfDecision(actor=actor, options=options)
        self.players[actor].is_first_turn = False

    def _compute_self_options(self, actor: int, drawn_tile: int) -> Set[str]:
        p = self.players[actor]
        options: Set[str] = set()
        can_riichi = (
            not p.is_riichi
            and p.open_melds == 0
            and p.score >= 1000
            and self.live_draws_left >= 4
        )
        can_tsumo, has_riichi_discard = self._evaluate_draw(
            seat=actor,
            drawn_tile=drawn_tile,
            check_riichi_discard=can_riichi,
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
        actor = d["l"]
        raw_tile = d["p"]
        tile_str = _strip_tile_suffix(raw_tile)
        tile_idx = tile_to_index(tile_str)
        is_riichi = "*" in raw_tile
        is_tsumogiri = "_" in raw_tile

        chosen: Set[str] = {"riichi"} if is_riichi else set()
        self._finalize_self(chosen, actor=actor)

        _remove_tiles(self.players[actor].concealed, tile_idx, 1)
        self.players[actor].furiten_tiles.add(tile_idx)
        self._invalidate_wait_mask(actor)

        suffix = "_tsumogiri" if is_tsumogiri else ""
        self.tokens.append(f"discard_{actor}_{index_to_tile(tile_idx)}{suffix}")

        if is_riichi:
            self.players[actor].is_riichi = True
            self.players[actor].score -= 1000
            self.tokens.append(f"riichi_{actor}")

        reaction = self._compute_reaction_options(actor, tile_idx)
        if reaction:
            self.pending_reaction = reaction
            for seat, opts in sorted(reaction.options_by_player.items()):
                for opt in sorted(opts):
                    self.tokens.append(f"opt_react_{seat}_{opt}")

    def _compute_reaction_options(self, discarder: int, tile_idx: int) -> Optional[ReactionDecision]:
        options_by_player: Dict[int, Set[str]] = {}
        ron_cases: List[
            Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int]
        ] = []
        ron_case_seats: List[int] = []

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
            Tuple[List[int], List[Tuple[str, int]], int, bool, bool, bool, int, int]
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
        actor = f["l"]
        meld_text = f["m"]
        meld_tiles = parse_meld_tiles(meld_text)
        action = classify_fulou(meld_tiles)

        discard_tile = None
        if self.pending_reaction:
            self.pending_reaction.chosen[actor] = action
            discard_tile = self.pending_reaction.discard_tile
            self._finalize_reaction()

        p = self.players[actor]

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
                _remove_tiles(p.concealed, tile, n)
        self._invalidate_wait_mask(actor)

        p.open_melds += 1
        p.is_riichi = False

        if action == "chi":
            anchor = min(meld_tiles)
            p.melds.append(("chi", anchor))
        elif action == "pon":
            tile = meld_tiles[0]
            p.open_pons[tile] = p.open_pons.get(tile, 0) + 1
            p.melds.append(("pon", tile))
        elif action == "minkan":
            tile = meld_tiles[0]
            p.melds.append(("minkan", tile))

        meld_repr = "_".join(index_to_tile(t) for t in sorted(meld_tiles))
        self.tokens.append(f"call_{action}_{actor}_{meld_repr}")

    def _on_gang(self, g: dict) -> None:
        actor = g["l"]
        meld_text = g["m"]
        kind = classify_gang(meld_text)
        meld_tiles = parse_meld_tiles(meld_text)
        tile = meld_tiles[0]

        chosen = {kind}
        self._finalize_self(chosen, actor=actor)

        p = self.players[actor]

        if kind == "ankan":
            _remove_tiles(p.concealed, tile, 4)
            p.closed_kans += 1
            p.melds.append(("ankan", tile))
        else:
            _remove_tiles(p.concealed, tile, 1)
            if p.open_pons.get(tile, 0) > 0:
                p.open_pons[tile] -= 1
                if p.open_pons[tile] == 0:
                    del p.open_pons[tile]
            replaced = False
            for i, (mtype, mtile) in enumerate(p.melds):
                if mtype == "pon" and mtile == tile:
                    p.melds[i] = ("minkan", tile)
                    replaced = True
                    break
            if not replaced:
                p.melds.append(("minkan", tile))
        self._invalidate_wait_mask(actor)

        self.tokens.append(f"kan_{kind}_{actor}_{index_to_tile(tile)}")

        if kind == "kakan":
            reaction = self._compute_kakan_reaction_options(actor, tile)
            if reaction:
                self.pending_reaction = reaction
                for seat, opts in sorted(reaction.options_by_player.items()):
                    for opt in sorted(opts):
                        self.tokens.append(f"opt_react_{seat}_{opt}")

    def _on_kaigang(self, k: dict) -> None:
        tile = _strip_tile_suffix(k["baopai"]).replace("0", "5")
        self.tokens.append(f"dora_{tile}")

    def _on_hule(self, h: dict) -> None:
        winner = h["l"]
        baojia = h.get("baojia")

        if baojia is not None:
            if self.pending_reaction and baojia == self.pending_reaction.discarder:
                self.pending_reaction.chosen[winner] = "ron"
            self.tokens.append(f"win_ron_{winner}_from_{baojia}")
        else:
            self._finalize_self({"tsumo"}, actor=winner)
            self.tokens.append(f"win_tsumo_{winner}")

        for seat in range(4):
            self.players[seat].score += h["fenpei"][seat]
            self.tokens.append(f"score_delta_{seat}_{h['fenpei'][seat]}")

    def _on_pingju(self, p: dict) -> None:
        name = p.get("name", "unknown")
        if name == "九種九牌":
            actor = self._kyushukyuhai_actor(p.get("shoupai"))
            if actor is not None:
                self._finalize_self({"kyushukyuhai"}, actor=actor)
        self.tokens.append(f"draw_{self._normalize_name(name)}")
        for seat in range(4):
            self.players[seat].score += p["fenpei"][seat]
            self.tokens.append(f"score_delta_{seat}_{p['fenpei'][seat]}")

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
        for opt in sorted(chosen_effective):
            self.tokens.append(f"take_self_{seat}_{opt}")
        for opt in sorted(self.pending_self.options - chosen_effective):
            self.tokens.append(f"pass_self_{seat}_{opt}")
        self.pending_self = None

    def _finalize_reaction(self) -> None:
        if not self.pending_reaction:
            return

        for seat, opts in sorted(self.pending_reaction.options_by_player.items()):
            chosen = self.pending_reaction.chosen.get(seat)
            if chosen and chosen in opts:
                self.tokens.append(f"take_react_{seat}_{chosen}")
                for opt in sorted(opts - {chosen}):
                    self.tokens.append(f"pass_react_{seat}_{opt}")
            else:
                for opt in sorted(opts):
                    self.tokens.append(f"pass_react_{seat}_{opt}")
            if "ron" in opts and chosen != "ron":
                if self.players[seat].is_riichi:
                    self.players[seat].riichi_furiten = True
                else:
                    self.players[seat].temporary_furiten = True
        self.pending_reaction = None

    def _is_reaction_continuation(self, key: str, value: dict) -> bool:
        if not self.pending_reaction:
            return False
        # kaigang can appear between discard and the actual fulou/hule reaction.
        # It is a dora reveal side event and should not close reaction windows.
        if key == "kaigang":
            return True
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
        if key == "dapai" and value.get("l") == actor:
            return True
        if key == "gang" and value.get("l") == actor:
            return True
        if key == "hule" and value.get("l") == actor and value.get("baojia") is None:
            return True
        if key == "pingju":
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
