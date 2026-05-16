from __future__ import annotations

import base64
import hashlib
import struct
from collections import Counter
from typing import Iterable

from .engine import TokenizeError, _parse_tiles, token_tile


def _expected_wall_counter(seat_count: int = 4) -> Counter[str]:
    counts: Counter[str] = Counter()
    if seat_count == 3:
        counts["m1"] = 4
        counts["m9"] = 4
        for suit in ("p", "s"):
            for number in range(1, 10):
                if number == 5:
                    counts[f"{suit}0"] = 1
                    counts[f"{suit}5"] = 3
                else:
                    counts[f"{suit}{number}"] = 4
        for number in range(1, 8):
            counts[f"z{number}"] = 4
        return counts
    for suit in ("m", "p", "s"):
        for number in range(1, 10):
            if number == 5:
                counts[f"{suit}0"] = 1
                counts[f"{suit}5"] = 3
            else:
                counts[f"{suit}{number}"] = 4
    for number in range(1, 8):
        counts[f"z{number}"] = 4
    return counts


class MT19937ar:
    """Minimal mt19937ar port for Tenhou shuffle seed replay."""

    N = 624
    M = 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF

    def __init__(self) -> None:
        self.mt = [0] * self.N
        self.mti = self.N + 1

    def init_genrand(self, seed: int) -> None:
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self.N):
            self.mt[i] = (
                1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i
            ) & 0xFFFFFFFF
        self.mti = self.N

    def init_by_array(self, init_key: Iterable[int]) -> None:
        key = list(init_key)
        if not key:
            raise TokenizeError("Tenhou shuffle seed init array is empty")
        self.init_genrand(19650218)
        i = 1
        j = 0
        for _ in range(max(self.N, len(key))):
            self.mt[i] = (
                (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1664525))
                + key[j]
                + j
            ) & 0xFFFFFFFF
            i += 1
            j += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
            if j >= len(key):
                j = 0
        for _ in range(self.N - 1):
            self.mt[i] = (
                (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1566083941))
                - i
            ) & 0xFFFFFFFF
            i += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
        self.mt[0] = 0x80000000

    def genrand_int32(self) -> int:
        mag01 = (0, self.MATRIX_A)
        if self.mti >= self.N:
            for kk in range(self.N - self.M):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk + 1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + self.M] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(self.N - self.M, self.N - 1):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk + 1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + (self.M - self.N)] ^ (y >> 1) ^ mag01[y & 1]
            y = (self.mt[self.N - 1] & self.UPPER_MASK) | (self.mt[0] & self.LOWER_MASK)
            self.mt[self.N - 1] = self.mt[self.M - 1] ^ (y >> 1) ^ mag01[y & 1]
            self.mti = 0

        y = self.mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def tile_id_to_token(tile_id: int, *, seat_count: int = 4) -> str:
    if seat_count == 3:
        return sanma_tile_id_to_token(tile_id)
    if tile_id < 0 or tile_id >= 136:
        raise TokenizeError(f"Tenhou tile id out of range: {tile_id}")
    suit = ("m", "p", "s", "z")[tile_id // 36]
    number = (tile_id % 36) // 4 + 1
    copy = tile_id % 4
    if suit != "z" and number == 5 and copy == 0:
        return f"{suit}0"
    return f"{suit}{number}"


def sanma_tile_id_to_token(tile_id: int) -> str:
    if tile_id < 0 or tile_id >= 108:
        raise TokenizeError(f"Tenhou sanma tile id out of range: {tile_id}")
    base = tile_id // 4
    copy = tile_id % 4
    if base == 0:
        return "m1"
    if base == 1:
        return "m9"
    if 2 <= base <= 10:
        number = base - 1
        if number == 5 and copy == 0:
            return "p0"
        return f"p{number}"
    if 11 <= base <= 19:
        number = base - 10
        if number == 5 and copy == 0:
            return "s0"
        return f"s{number}"
    return f"z{base - 19}"


def raw_tenhou_tile_id_to_token(tile_id: int, *, seat_count: int) -> str:
    if seat_count == 3:
        if 0 <= tile_id <= 3:
            compact = tile_id
        elif 32 <= tile_id <= 35:
            compact = 4 + (tile_id - 32)
        elif 36 <= tile_id <= 71:
            compact = 8 + (tile_id - 36)
        elif 72 <= tile_id <= 107:
            compact = 44 + (tile_id - 72)
        elif 108 <= tile_id <= 135:
            compact = 80 + (tile_id - 108)
        else:
            raise TokenizeError(f"Tenhou sanma raw tile id is not used: {tile_id}")
        return sanma_tile_id_to_token(compact)
    return tile_id_to_token(tile_id, seat_count=4)


def generate_tenhou_wall_ids(seed_b64: str, round_count: int, *, seat_count: int = 4) -> list[list[int]]:
    seed = base64.b64decode(seed_b64)
    if len(seed) < 624 * 4:
        raise TokenizeError(f"Tenhou shuffle seed is too short: {len(seed)} bytes")
    wall_size = 108 if seat_count == 3 else 136
    mt = MT19937ar()
    mt.init_by_array(struct.unpack("<624I", seed[: 624 * 4]))

    walls: list[list[int]] = []
    for _round in range(round_count):
        src = [mt.genrand_int32() for _ in range(288)]
        rnd: list[int] = []
        for i in range(9):
            packed = struct.pack("<32I", *src[32 * i : 32 * (i + 1)])
            rnd.extend(struct.unpack("<16I", hashlib.sha512(packed).digest()))

        wall = list(range(wall_size))
        for i in range(wall_size):
            j = rnd[i] % (wall_size - i) + i
            wall[i], wall[j] = wall[j], wall[i]
        # Tenhou draws from the tail of the generated array. Store wall blocks in
        # actual consumption order so omniscient prompts can be checked directly.
        walls.append(list(reversed(wall)))
    return walls


def generate_tenhou_wall_tokens(seed_b64: str, round_count: int, *, seat_count: int = 4) -> list[list[str]]:
    return [
        [tile_id_to_token(tile_id, seat_count=seat_count) for tile_id in wall]
        for wall in generate_tenhou_wall_ids(seed_b64, round_count, seat_count=seat_count)
    ]


def _round_observed_tile_tokens(round_data: list[dict]) -> list[str]:
    observed: list[str] = []
    pending_hule_fubaopai: list[Counter[str]] = []

    def flush_hule_fubaopai() -> None:
        nonlocal pending_hule_fubaopai
        if not pending_hule_fubaopai:
            return
        merged: Counter[str] = Counter()
        for counter in pending_hule_fubaopai:
            for tile, count in counter.items():
                merged[tile] = max(merged[tile], count)
        observed.extend(merged.elements())
        pending_hule_fubaopai = []

    for event in round_data:
        if not isinstance(event, dict) or not event:
            continue
        key, value = next(iter(event.items()))
        if not isinstance(value, dict):
            continue
        if key != "hule":
            flush_hule_fubaopai()
        if key == "qipai":
            shoupai = value.get("shoupai")
            if isinstance(shoupai, list):
                for hand in shoupai:
                    if isinstance(hand, str):
                        observed.extend(_parse_tiles(hand, stop_at_comma=True, context="omniscient_wall_qipai"))
            baopai = value.get("baopai")
            if isinstance(baopai, str):
                observed.append(token_tile(baopai.replace("*", "").replace("_", "")))
        elif key in {"zimo", "gangzimo"}:
            tile = value.get("p")
            if isinstance(tile, str):
                observed.append(token_tile(tile.replace("*", "").replace("_", "")))
        elif key == "kaigang":
            baopai = value.get("baopai")
            if isinstance(baopai, str):
                observed.append(token_tile(baopai.replace("*", "").replace("_", "")))
        elif key == "hule":
            fubaopai = value.get("fubaopai")
            if isinstance(fubaopai, list):
                counter: Counter[str] = Counter()
                for tile in fubaopai:
                    if isinstance(tile, str):
                        counter[token_tile(tile.replace("*", "").replace("_", ""))] += 1
                pending_hule_fubaopai.append(counter)
    flush_hule_fubaopai()
    return observed


def _round_seat_count(round_data: list[dict], fallback: int) -> int:
    for event in round_data:
        if not isinstance(event, dict):
            continue
        qipai = event.get("qipai")
        if not isinstance(qipai, dict):
            continue
        shoupai = qipai.get("shoupai")
        if isinstance(shoupai, list) and len(shoupai) in {3, 4}:
            return len(shoupai)
        defen = qipai.get("defen")
        if isinstance(defen, list) and len(defen) in {3, 4}:
            return len(defen)
    return fallback


def _parse_hand_tiles(hand: object, *, context: str) -> list[str]:
    if not isinstance(hand, str):
        return []
    return _parse_tiles(hand, stop_at_comma=True, context=context)


class _WallOrderChecker:
    def __init__(self, wall_tokens: list[str], seat_count: int, round_index: int) -> None:
        self.wall_tokens = wall_tokens
        self.seat_count = seat_count
        self.round_index = round_index
        self.live_cursor = 0
        self.rinshan_cursor = 0
        self.dora_count = 0
        self.pending_replacement_draw = False

    def check_qipai(self, qipai: dict) -> None:
        hands = qipai.get("shoupai")
        if not isinstance(hands, list) or len(hands) != self.seat_count:
            raise TokenizeError(f"qipai shoupai seat count mismatch: round={self.round_index}")
        ordered_hands = qipai.get("_ordered_shoupai_tokens")
        if isinstance(ordered_hands, list) and len(ordered_hands) == self.seat_count:
            hand_tiles = [
                [tile for tile in hand if isinstance(tile, str)]
                for hand in ordered_hands
            ]
        else:
            raise TokenizeError(
                f"ordered qipai tiles are required for strict wall assertion: round={self.round_index}"
            )
        if any(len(tiles) < 13 for tiles in hand_tiles):
            raise TokenizeError(f"qipai hand is too short for wall order assertion: round={self.round_index}")
        oya = qipai.get("_wall_oya", 0)
        if not isinstance(oya, int) or oya < 0 or oya >= self.seat_count:
            raise TokenizeError(f"qipai dealer seat is invalid for wall assertion: round={self.round_index}")
        hand_offsets = [0] * self.seat_count
        for _chunk in range(3):
            for offset in range(self.seat_count):
                seat = (oya + offset) % self.seat_count
                self._expect_live(hand_tiles[seat][hand_offsets[seat] : hand_offsets[seat] + 4])
                hand_offsets[seat] += 4
        for offset in range(self.seat_count):
            seat = (oya + offset) % self.seat_count
            self._expect_live(hand_tiles[seat][hand_offsets[seat] : hand_offsets[seat] + 1])
            hand_offsets[seat] += 1
        baopai = qipai.get("baopai")
        if isinstance(baopai, str):
            self._expect_dora(token_tile(baopai.replace("*", "").replace("_", "")), dora_index=0)

    def check_event(self, key: str, value: dict) -> None:
        if key == "zimo":
            tile = value.get("p")
            if isinstance(tile, str):
                self._expect_draw(token_tile(tile.replace("*", "").replace("_", "")))
            return
        if key == "gangzimo":
            tile = value.get("p")
            if isinstance(tile, str):
                self._expect_rinshan(token_tile(tile.replace("*", "").replace("_", "")))
            return
        if key == "penuki":
            self.pending_replacement_draw = True
            return
        if key == "kaigang":
            baopai = value.get("baopai")
            if isinstance(baopai, str):
                self.dora_count += 1
                self._expect_dora(token_tile(baopai.replace("*", "").replace("_", "")), dora_index=self.dora_count)
            return

    def check_ura(self, fubaopai_counters: list[Counter[str]]) -> None:
        if not fubaopai_counters:
            return
        expected = Counter(self._ura_tiles())
        merged: Counter[str] = Counter()
        for counter in fubaopai_counters:
            for tile, count in counter.items():
                merged[tile] = max(merged[tile], count)
        if merged != expected:
            raise TokenizeError(
                f"ura indicator contradicts reconstructed wall: round={self.round_index} "
                f"expected={dict(expected)} actual={dict(merged)}"
            )

    def _expect_draw(self, tile: str) -> None:
        if self.pending_replacement_draw:
            self.pending_replacement_draw = False
            self._expect_rinshan(tile)
        else:
            self._expect_live([tile])

    def _expect_live(self, tiles: list[str]) -> None:
        expected = self.wall_tokens[self.live_cursor : self.live_cursor + len(tiles)]
        if expected != tiles:
            raise TokenizeError(
                f"wall order mismatch: round={self.round_index} live_index={self.live_cursor} "
                f"expected={expected} actual={tiles}"
            )
        self.live_cursor += len(tiles)

    def _expect_rinshan(self, tile: str) -> None:
        index = self._rinshan_index(self.rinshan_cursor)
        expected = self.wall_tokens[index]
        if expected != tile:
            raise TokenizeError(
                f"rinshan tile contradicts reconstructed wall: round={self.round_index} "
                f"rinshan_index={self.rinshan_cursor} expected={expected} actual={tile}"
            )
        self.rinshan_cursor += 1

    def _expect_dora(self, tile: str, *, dora_index: int) -> None:
        index = self._dora_index(dora_index)
        expected = self.wall_tokens[index]
        if expected != tile:
            raise TokenizeError(
                f"dora indicator contradicts reconstructed wall: round={self.round_index} "
                f"dora_index={dora_index} wall_index={index} expected={expected} actual={tile}"
            )

    def _ura_tiles(self) -> list[str]:
        return [self.wall_tokens[self._ura_index(index)] for index in range(self.dora_count + 1)]

    def _dora_index(self, index: int) -> int:
        if index >= 5:
            raise TokenizeError(f"too many dora indicators for wall assertion: round={self.round_index}")
        first_indicator_offset = 10 if self.seat_count == 3 else 6
        return len(self.wall_tokens) - first_indicator_offset - 2 * index

    def _ura_index(self, index: int) -> int:
        if index >= 5:
            raise TokenizeError(f"too many ura indicators for wall assertion: round={self.round_index}")
        first_indicator_offset = 9 if self.seat_count == 3 else 5
        return len(self.wall_tokens) - first_indicator_offset - 2 * index

    def _rinshan_index(self, index: int) -> int:
        order = (-2, -1, -4, -3, -6, -5, -8, -7) if self.seat_count == 3 else (-2, -1, -4, -3)
        if index >= len(order):
            raise TokenizeError(f"too many replacement draws for wall assertion: round={self.round_index}")
        return len(self.wall_tokens) + order[index]


def assert_wall_consistent_with_game(game: dict, wall_tokens_by_round: list[list[str]]) -> None:
    log = game.get("log")
    if not isinstance(log, list):
        raise TokenizeError("game log must be a list for wall assertion")
    if len(log) != len(wall_tokens_by_round):
        raise TokenizeError(
            f"wall round count mismatch: log={len(log)} wall={len(wall_tokens_by_round)}"
        )
    fallback_seat_count = 3 if "三" in str(game.get("title", "")) else 4
    for round_index, (round_data, wall_tokens) in enumerate(zip(log, wall_tokens_by_round)):
        if not isinstance(round_data, list):
            raise TokenizeError("round data must be a list for wall assertion")
        seat_count = _round_seat_count(round_data, fallback_seat_count)
        expected_len = 108 if seat_count == 3 else 136
        if len(wall_tokens) != expected_len:
            raise TokenizeError(
                f"wall must contain {expected_len} tiles: round={round_index} len={len(wall_tokens)}"
            )
        if Counter(wall_tokens) != _expected_wall_counter(seat_count):
            raise TokenizeError(f"wall tile multiset is invalid: round={round_index}")
        remaining = Counter(wall_tokens)
        for tile in _round_observed_tile_tokens(round_data):
            remaining[tile] -= 1
            if remaining[tile] < 0:
                raise TokenizeError(
                    f"observed tile contradicts reconstructed wall: round={round_index} tile={tile}"
                )
        checker = _WallOrderChecker(wall_tokens, seat_count, round_index)
        pending_hule_fubaopai: list[Counter[str]] = []
        for event in round_data:
            if not isinstance(event, dict) or not event:
                continue
            key, value = next(iter(event.items()))
            if not isinstance(value, dict):
                continue
            if key != "hule":
                checker.check_ura(pending_hule_fubaopai)
                pending_hule_fubaopai = []
            if key == "qipai":
                checker.check_qipai(value)
            elif key == "hule":
                fubaopai = value.get("fubaopai")
                if isinstance(fubaopai, list):
                    counter: Counter[str] = Counter()
                    for tile in fubaopai:
                        if isinstance(tile, str):
                            counter[token_tile(tile.replace("*", "").replace("_", ""))] += 1
                    pending_hule_fubaopai.append(counter)
            else:
                checker.check_event(key, value)
        checker.check_ura(pending_hule_fubaopai)
