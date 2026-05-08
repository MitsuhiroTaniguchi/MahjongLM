from __future__ import annotations

import base64
import hashlib
import struct
from collections import Counter
from typing import Iterable

from .engine import TokenizeError, _parse_tiles, token_tile


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


def tile_id_to_token(tile_id: int) -> str:
    if tile_id < 0 or tile_id >= 136:
        raise TokenizeError(f"Tenhou tile id out of range: {tile_id}")
    suit = ("m", "p", "s", "z")[tile_id // 36]
    number = (tile_id % 36) // 4 + 1
    copy = tile_id % 4
    if suit != "z" and number == 5 and copy == 0:
        return f"{suit}0"
    return f"{suit}{number}"


def generate_tenhou_wall_ids(seed_b64: str, round_count: int) -> list[list[int]]:
    seed = base64.b64decode(seed_b64)
    if len(seed) < 624 * 4:
        raise TokenizeError(f"Tenhou shuffle seed is too short: {len(seed)} bytes")
    mt = MT19937ar()
    mt.init_by_array(struct.unpack("<624I", seed[: 624 * 4]))

    walls: list[list[int]] = []
    for _round in range(round_count):
        src = [mt.genrand_int32() for _ in range(288)]
        rnd: list[int] = []
        for i in range(9):
            packed = struct.pack("<32I", *src[32 * i : 32 * (i + 1)])
            rnd.extend(struct.unpack("<16I", hashlib.sha512(packed).digest()))

        wall = list(range(136))
        for i in range(136):
            j = rnd[i] % (136 - i) + i
            wall[i], wall[j] = wall[j], wall[i]
        walls.append(wall)
    return walls


def generate_tenhou_wall_tokens(seed_b64: str, round_count: int) -> list[list[str]]:
    return [[tile_id_to_token(tile_id) for tile_id in wall] for wall in generate_tenhou_wall_ids(seed_b64, round_count)]


def _round_observed_tile_tokens(round_data: list[dict]) -> list[str]:
    observed: list[str] = []
    for event in round_data:
        if not isinstance(event, dict) or not event:
            continue
        key, value = next(iter(event.items()))
        if not isinstance(value, dict):
            continue
        if key == "qipai":
            shoupai = value.get("shoupai")
            if isinstance(shoupai, list):
                for hand in shoupai:
                    if isinstance(hand, str):
                        observed.extend(_parse_tiles(hand, stop_at_comma=True, context="omniscient_wall_qipai"))
        elif key in {"zimo", "gangzimo"}:
            tile = value.get("p")
            if isinstance(tile, str):
                observed.append(token_tile(tile.replace("*", "").replace("_", "")))
    return observed


def assert_wall_consistent_with_game(game: dict, wall_tokens_by_round: list[list[str]]) -> None:
    log = game.get("log")
    if not isinstance(log, list):
        raise TokenizeError("game log must be a list for wall assertion")
    if len(log) != len(wall_tokens_by_round):
        raise TokenizeError(
            f"wall round count mismatch: log={len(log)} wall={len(wall_tokens_by_round)}"
        )
    for round_index, (round_data, wall_tokens) in enumerate(zip(log, wall_tokens_by_round)):
        if not isinstance(round_data, list):
            raise TokenizeError("round data must be a list for wall assertion")
        if len(wall_tokens) != 136:
            raise TokenizeError(f"wall must contain 136 tiles: round={round_index} len={len(wall_tokens)}")
        remaining = Counter(wall_tokens)
        for tile in _round_observed_tile_tokens(round_data):
            remaining[tile] -= 1
            if remaining[tile] < 0:
                raise TokenizeError(
                    f"observed tile contradicts reconstructed wall: round={round_index} tile={tile}"
                )
