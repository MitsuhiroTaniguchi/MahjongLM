from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_VOCAB_PATH = Path(__file__).resolve().parents[2] / "tokenizer" / "vocab.txt"
TOKEN_ID_MAGIC = b"TIDS"
TOKEN_ID_HEADER = struct.Struct("<4sBI16s")


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]
    token_to_id: dict[str, int]
    fingerprint: bytes

    @classmethod
    def load(cls, path: Path | None = None) -> Vocabulary:
        vocab_path = path if path is not None else DEFAULT_VOCAB_PATH
        text = vocab_path.read_text(encoding="utf-8")
        tokens = tuple(text.splitlines())
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"duplicate tokens found in {vocab_path}")
        return cls(
            tokens=tokens,
            token_to_id={token: idx for idx, token in enumerate(tokens)},
            fingerprint=hashlib.sha256(text.encode("utf-8")).digest()[:16],
        )

    def encode(self, tokens: Iterable[str]) -> list[int]:
        ids: list[int] = []
        for token in tokens:
            try:
                ids.append(self.token_to_id[token])
            except KeyError as exc:
                raise KeyError(f"token not found in vocabulary: {token}") from exc
        return ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        decoded: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.tokens):
                raise IndexError(f"token id out of range: {idx}")
            decoded.append(self.tokens[idx])
        return decoded


def _array_typecode(max_id: int) -> str:
    if max_id < 2**16:
        return "H"
    if max_id < 2**32:
        return "I"
    raise ValueError(f"token id exceeds uint32 range: {max_id}")


def save_token_ids(path: Path, ids: Iterable[int], *, vocab_fingerprint: bytes) -> None:
    values = list(ids)
    max_id = max(values, default=0)
    typecode = _array_typecode(max_id)
    width = array(typecode).itemsize
    payload = array(typecode, values)
    if sys.byteorder != "little":
        payload.byteswap()

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            f.write(TOKEN_ID_HEADER.pack(TOKEN_ID_MAGIC, width, len(values), vocab_fingerprint))
            f.write(payload.tobytes())
        temp_path.replace(path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def load_token_ids(path: Path, *, expected_vocab_fingerprint: bytes | None = None) -> list[int]:
    data = path.read_bytes()
    if len(data) < TOKEN_ID_HEADER.size:
        raise ValueError(f"token id file too short: {path}")
    magic, width, count, vocab_fingerprint = TOKEN_ID_HEADER.unpack_from(data, 0)
    if magic != TOKEN_ID_MAGIC:
        raise ValueError(f"invalid token id file magic: {path}")
    if width not in {2, 4}:
        raise ValueError(f"unsupported token id width {width}: {path}")
    if expected_vocab_fingerprint is not None and vocab_fingerprint != expected_vocab_fingerprint:
        raise ValueError(f"token id vocab fingerprint mismatch: {path}")

    payload = data[TOKEN_ID_HEADER.size :]
    expected = count * width
    if len(payload) != expected:
        raise ValueError(
            f"token id payload size mismatch for {path}: expected {expected} bytes, got {len(payload)}"
        )

    typecode = "H" if width == 2 else "I"
    values = array(typecode)
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)
