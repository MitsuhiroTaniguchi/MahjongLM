from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from datasets import Dataset, concatenate_datasets, load_from_disk


@dataclass(frozen=True)
class PackedGroupStats:
    segment_count: int
    packed_row_count: int
    packed_tokens: int
    padded_tokens: int

    @property
    def packing_efficiency(self) -> float:
        if self.padded_tokens == 0:
            return 0.0
        return self.packed_tokens / self.padded_tokens


@dataclass(frozen=True)
class PackedBatch:
    input_ids: object
    attention_mask: object
    labels: object
    stats: PackedGroupStats
    group_ids: list[str]
    packed_group_ids: list[list[str]]
    packed_view_counts: list[int]

    def pin_memory(self):
        pin = lambda tensor: tensor.pin_memory() if hasattr(tensor, "pin_memory") else tensor
        return PackedBatch(
            input_ids=pin(self.input_ids),
            attention_mask=pin(self.attention_mask),
            labels=pin(self.labels),
            stats=self.stats,
            group_ids=self.group_ids,
            packed_group_ids=self.packed_group_ids,
            packed_view_counts=self.packed_view_counts,
        )


@dataclass(frozen=True)
class UnpackedBatch:
    input_ids: object
    attention_mask: object
    labels: object
    stats: PackedGroupStats
    group_ids: list[str]
    packed_group_ids: list[list[str]]
    packed_view_counts: list[int]

    def pin_memory(self):
        pin = lambda tensor: tensor.pin_memory() if hasattr(tensor, "pin_memory") else tensor
        return UnpackedBatch(
            input_ids=pin(self.input_ids),
            attention_mask=pin(self.attention_mask),
            labels=pin(self.labels),
            stats=self.stats,
            group_ids=self.group_ids,
            packed_group_ids=self.packed_group_ids,
            packed_view_counts=self.packed_view_counts,
        )


def load_grouped_dataset(dataset_dirs: Sequence[str | Path]) -> Dataset:
    datasets = [load_from_disk(str(path)) for path in dataset_dirs]
    if not datasets:
        raise ValueError("dataset_dirs must not be empty")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def validate_grouped_dataset(dataset: Dataset) -> None:
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    grouped: dict[str, list[dict]] = {}
    for row in dataset:
        grouped.setdefault(row["group_id"], []).append(row)
        if row["length"] != len(row["input_ids"]):
            raise ValueError(f"length mismatch for {row['group_id']} {row['view_type']}")
    for group_id, rows in grouped.items():
        seat_counts = {int(row["seat_count"]) for row in rows}
        if len(seat_counts) != 1:
            raise ValueError(f"inconsistent seat_count within group {group_id}")
        seat_count = seat_counts.pop()
        unknown_view_types = {
            row["view_type"]
            for row in rows
            if row["view_type"] not in {"complete", "imperfect", "omniscient"}
        }
        if unknown_view_types:
            raise ValueError(f"group {group_id} contains unknown view types: {sorted(unknown_view_types)}")
        complete_rows = [row for row in rows if row["view_type"] == "complete"]
        if len(complete_rows) != 1:
            raise ValueError(f"group {group_id} must contain exactly one complete view")
        if int(complete_rows[0]["viewer_seat"]) != -1:
            raise ValueError(f"group {group_id} complete view must use viewer_seat=-1")
        omniscient_rows = [row for row in rows if row["view_type"] == "omniscient"]
        if len(omniscient_rows) > 1:
            raise ValueError(f"group {group_id} must contain at most one omniscient view")
        if omniscient_rows and int(omniscient_rows[0]["viewer_seat"]) != -1:
            raise ValueError(f"group {group_id} omniscient view must use viewer_seat=-1")
        expected_rows = seat_count + 1 + len(omniscient_rows)
        if len(rows) != expected_rows:
            raise ValueError(f"group {group_id} expected {expected_rows} views, found {len(rows)}")
        imperfect_viewers = sorted(row["viewer_seat"] for row in rows if row["view_type"] == "imperfect")
        if imperfect_viewers != list(range(seat_count)):
            raise ValueError(
                f"group {group_id} imperfect viewers mismatch: {imperfect_viewers} expected {list(range(seat_count))}"
            )


def limit_groups(dataset: Dataset, max_groups: int) -> Dataset:
    if max_groups <= 0:
        return dataset
    group_ids: list[str] = []
    seen: set[str] = set()
    for group_id in dataset["group_id"]:
        if group_id in seen:
            continue
        seen.add(group_id)
        group_ids.append(group_id)
        if len(group_ids) >= max_groups:
            break
    chosen = set(group_ids)
    indices = [idx for idx, group_id in enumerate(dataset["group_id"]) if group_id in chosen]
    return dataset.select(indices)


def split_grouped_dataset(dataset: Dataset, *, eval_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if not (0.0 <= eval_ratio < 1.0):
        raise ValueError("eval_ratio must be in [0, 1)")
    group_ids = list(dict.fromkeys(dataset["group_id"]))
    if not group_ids:
        raise ValueError("dataset is empty")
    if eval_ratio == 0.0:
        return dataset, dataset.select([])
    eval_group_count = max(1, math.ceil(len(group_ids) * eval_ratio))
    rng = random.Random(seed)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    eval_groups = set(shuffled[:eval_group_count])
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for idx, group_id in enumerate(dataset["group_id"]):
        if group_id in eval_groups:
            eval_indices.append(idx)
        else:
            train_indices.append(idx)
    if not train_indices:
        raise ValueError("train split is empty; lower eval_ratio or provide more data")
    if not eval_indices:
        raise ValueError("eval split is empty; raise eval_ratio or provide eval_dataset_dirs")
    return dataset.select(train_indices), dataset.select(eval_indices)


def count_unique_groups(dataset: Dataset) -> int:
    return len(dict.fromkeys(dataset["group_id"]))


def _dataset_dir_cache_signature(path: str | Path) -> dict[str, object]:
    dataset_dir = Path(path).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    files: list[dict[str, object]] = []
    for child in sorted(dataset_dir.iterdir(), key=lambda item: item.name):
        if child.name not in {"dataset_info.json", "state.json"} and child.suffix != ".arrow":
            continue
        stat = child.stat()
        files.append(
            {
                "name": child.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "path": str(dataset_dir).lower(),
        "files": files,
    }


def build_split_cache_paths(
    dataset_dirs: Sequence[str | Path],
    *,
    eval_ratio: float,
    seed: int,
    cache_dir: str | Path,
) -> tuple[Path, Path, Path]:
    cache_key_payload = {
        "datasets": [_dataset_dir_cache_signature(path) for path in dataset_dirs],
        "eval_ratio": eval_ratio,
        "seed": seed,
    }
    digest = hashlib.sha256(json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    split_root = Path(cache_dir) / "splits" / f"{digest}_eval{eval_ratio:g}_seed{seed}"
    return split_root / "train", split_root / "eval", split_root / "metadata.json"


def load_cached_split(
    dataset_dirs: Sequence[str | Path],
    *,
    eval_ratio: float,
    seed: int,
    cache_dir: str | Path,
) -> tuple[Dataset, Dataset, dict] | None:
    train_path, eval_path, metadata_path = build_split_cache_paths(
        dataset_dirs,
        eval_ratio=eval_ratio,
        seed=seed,
        cache_dir=cache_dir,
    )
    if not train_path.exists() or not eval_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return load_from_disk(str(train_path)), load_from_disk(str(eval_path)), metadata


def save_split_cache(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    *,
    dataset_dirs: Sequence[str | Path],
    eval_ratio: float,
    seed: int,
    cache_dir: str | Path,
) -> dict:
    train_path, eval_path, metadata_path = build_split_cache_paths(
        dataset_dirs,
        eval_ratio=eval_ratio,
        seed=seed,
        cache_dir=cache_dir,
    )
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(train_path))
    eval_dataset.save_to_disk(str(eval_path))
    metadata = {
        "dataset_dirs": [str(Path(path).resolve()) for path in dataset_dirs],
        "dataset_cache_signatures": [_dataset_dir_cache_signature(path) for path in dataset_dirs],
        "eval_ratio": eval_ratio,
        "seed": seed,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "train_groups": count_unique_groups(train_dataset),
        "eval_groups": count_unique_groups(eval_dataset),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


class PackedGroupCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int,
        pad_to_multiple_of: int = 1,
        return_tensors: str = "pt",
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self._torch_causal_block_cache: dict[int, object] = {}
        self._numpy_causal_block_cache: dict[int, object] = {}

    def _segment_from_feature(self, feature: dict) -> dict:
        input_ids = list(feature["input_ids"])
        segment = [self.bos_token_id] + input_ids + [self.eos_token_id]
        if len(segment) > self.max_length:
            raise ValueError(
                f"sequence for group {feature['group_id']} view {feature['view_type']} exceeds max_length: "
                f"{len(segment)} > {self.max_length}"
            )
        return {
            "group_id": feature["group_id"],
            "segment": segment,
            "view_type": feature.get("view_type"),
            "viewer_seat": feature.get("viewer_seat"),
        }

    def _pack_segments(self, segments: list[dict]) -> tuple[list[list[int]], list[list[str]], list[int], list[list[int]]]:
        packed_rows: list[list[int]] = []
        packed_group_ids: list[list[str]] = []
        packed_lengths: list[int] = []
        packed_segment_lengths: list[list[int]] = []
        row_group_sets: list[set[str]] = []

        segments.sort(key=lambda item: len(item["segment"]), reverse=True)
        for item in segments:
            segment = item["segment"]
            group_id = item["group_id"]
            best_idx: int | None = None
            best_remaining: int | None = None
            for idx, row in enumerate(packed_rows):
                remaining = self.max_length - len(row)
                if group_id in row_group_sets[idx]:
                    continue
                if len(segment) > remaining:
                    continue
                after = remaining - len(segment)
                if best_remaining is None or after < best_remaining:
                    best_remaining = after
                    best_idx = idx
            if best_idx is None:
                packed_rows.append(list(segment))
                packed_group_ids.append([group_id])
                packed_lengths.append(len(segment))
                packed_segment_lengths.append([len(segment)])
                row_group_sets.append({group_id})
                continue
            packed_rows[best_idx].extend(segment)
            packed_group_ids[best_idx].append(group_id)
            packed_lengths[best_idx] += len(segment)
            packed_segment_lengths[best_idx].append(len(segment))
            row_group_sets[best_idx].add(group_id)
        return packed_rows, packed_group_ids, packed_lengths, packed_segment_lengths

    def _get_torch_causal_block(self, length: int):
        import torch

        cached = self._torch_causal_block_cache.get(length)
        if cached is None:
            cached = torch.tril(torch.ones((length, length), dtype=torch.bool))
            self._torch_causal_block_cache[length] = cached
        return cached

    def _get_numpy_causal_block(self, length: int):
        import numpy as np

        cached = self._numpy_causal_block_cache.get(length)
        if cached is None:
            cached = np.tri(length, length, k=0, dtype=np.bool_)
            self._numpy_causal_block_cache[length] = cached
        return cached

    def _build_attention_mask_pt(self, packed_segment_lengths: list[list[int]], batch_max_len: int):
        import torch

        attention_mask = torch.zeros((len(packed_segment_lengths), batch_max_len, batch_max_len), dtype=torch.bool)
        for row_idx, segment_lengths in enumerate(packed_segment_lengths):
            offset = 0
            for segment_length in segment_lengths:
                next_offset = offset + segment_length
                attention_mask[row_idx, offset:next_offset, offset:next_offset] = self._get_torch_causal_block(
                    segment_length
                )
                offset = next_offset
        return attention_mask.unsqueeze(1)

    def _build_attention_mask_np(self, packed_segment_lengths: list[list[int]], batch_max_len: int):
        import numpy as np

        attention_mask = np.zeros((len(packed_segment_lengths), 1, batch_max_len, batch_max_len), dtype=np.bool_)
        for row_idx, segment_lengths in enumerate(packed_segment_lengths):
            offset = 0
            for segment_length in segment_lengths:
                next_offset = offset + segment_length
                attention_mask[row_idx, 0, offset:next_offset, offset:next_offset] = self._get_numpy_causal_block(
                    segment_length
                )
                offset = next_offset
        return attention_mask

    def __call__(self, features: list[dict]) -> PackedBatch | dict:
        if not features:
            raise ValueError("features must not be empty")
        group_ids = [str(feature["group_id"]) for feature in features]
        segments = [self._segment_from_feature(feature) for feature in features]
        packed_rows, packed_group_ids, packed_lengths, packed_segment_lengths = self._pack_segments(segments)
        batch_max_len = max(packed_lengths)
        if self.pad_to_multiple_of > 1 and batch_max_len % self.pad_to_multiple_of:
            batch_max_len += self.pad_to_multiple_of - (batch_max_len % self.pad_to_multiple_of)

        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        packed_segment_counts: list[int] = []
        for row in packed_rows:
            pad_len = batch_max_len - len(row)
            input_ids.append(row + [self.pad_token_id] * pad_len)
            labels.append(row + [-100] * pad_len)
            packed_segment_counts.append(0)
        for idx, group_ids_for_row in enumerate(packed_group_ids):
            packed_segment_counts[idx] = len(group_ids_for_row)

        stats = PackedGroupStats(
            segment_count=len(segments),
            packed_row_count=len(packed_rows),
            packed_tokens=sum(packed_lengths),
            padded_tokens=len(packed_rows) * batch_max_len,
        )

        if self.return_tensors == "pt":
            import torch

            return PackedBatch(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=self._build_attention_mask_pt(packed_segment_lengths, batch_max_len),
                labels=torch.tensor(labels, dtype=torch.long),
                stats=stats,
                group_ids=sorted(set(group_ids)),
                packed_group_ids=packed_group_ids,
                packed_view_counts=packed_segment_counts,
            )
        if self.return_tensors == "np":
            import numpy as np

            return PackedBatch(
                input_ids=np.asarray(input_ids, dtype=np.int64),
                attention_mask=self._build_attention_mask_np(packed_segment_lengths, batch_max_len),
                labels=np.asarray(labels, dtype=np.int64),
                stats=stats,
                group_ids=sorted(set(group_ids)),
                packed_group_ids=packed_group_ids,
                packed_view_counts=packed_segment_counts,
            )
        raise ValueError(f"unsupported return_tensors: {self.return_tensors}")


class UnpackedCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int,
        pad_to_multiple_of: int = 1,
        return_tensors: str = "pt",
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def _segment_from_feature(self, feature: dict) -> tuple[str, list[int]]:
        input_ids = list(feature["input_ids"])
        segment = [self.bos_token_id] + input_ids + [self.eos_token_id]
        if len(segment) > self.max_length:
            raise ValueError(
                f"sequence for group {feature['group_id']} view {feature['view_type']} exceeds max_length: "
                f"{len(segment)} > {self.max_length}"
            )
        return str(feature["group_id"]), segment

    def __call__(self, features: list[dict]) -> UnpackedBatch:
        if not features:
            raise ValueError("features must not be empty")
        rows = [self._segment_from_feature(feature) for feature in features]
        lengths = [len(segment) for _group_id, segment in rows]
        batch_max_len = max(lengths)
        if self.pad_to_multiple_of > 1 and batch_max_len % self.pad_to_multiple_of:
            batch_max_len += self.pad_to_multiple_of - (batch_max_len % self.pad_to_multiple_of)

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        group_ids: list[str] = []
        packed_group_ids: list[list[str]] = []
        packed_view_counts: list[int] = []

        for group_id, segment in rows:
            pad_len = batch_max_len - len(segment)
            input_ids.append(segment + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(segment) + [0] * pad_len)
            labels.append(segment + [-100] * pad_len)
            group_ids.append(group_id)
            packed_group_ids.append([group_id])
            packed_view_counts.append(1)

        stats = PackedGroupStats(
            segment_count=len(rows),
            packed_row_count=len(rows),
            packed_tokens=sum(lengths),
            padded_tokens=len(rows) * batch_max_len,
        )

        if self.return_tensors == "pt":
            import torch

            return UnpackedBatch(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long),
                labels=torch.tensor(labels, dtype=torch.long),
                stats=stats,
                group_ids=group_ids,
                packed_group_ids=packed_group_ids,
                packed_view_counts=packed_view_counts,
            )
        if self.return_tensors == "np":
            import numpy as np

            return UnpackedBatch(
                input_ids=np.asarray(input_ids, dtype=np.int64),
                attention_mask=np.asarray(attention_mask, dtype=np.int64),
                labels=np.asarray(labels, dtype=np.int64),
                stats=stats,
                group_ids=group_ids,
                packed_group_ids=packed_group_ids,
                packed_view_counts=packed_view_counts,
            )
        raise ValueError(f"unsupported return_tensors: {self.return_tensors}")


class TokenBudgetGroupBatchSampler:
    def __init__(
        self,
        group_ids: Sequence[str],
        group_token_counts: Sequence[int],
        *,
        max_tokens_per_batch: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive")
        if len(group_ids) != len(group_token_counts):
            raise ValueError("group_ids and group_token_counts must have the same length")
        self.group_ids = list(group_ids)
        self.group_token_counts = list(group_token_counts)
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.seed = seed

    def _group_state(self) -> tuple[dict[str, list[int]], dict[str, int], list[str]]:
        grouped_indices: dict[str, list[int]] = {}
        grouped_tokens: dict[str, int] = {}
        group_order: list[str] = []
        for idx, (group_id, group_token_count) in enumerate(zip(self.group_ids, self.group_token_counts)):
            if group_id not in grouped_indices:
                grouped_indices[group_id] = []
                grouped_tokens[group_id] = 0
                group_order.append(group_id)
            grouped_indices[group_id].append(idx)
            grouped_tokens[group_id] += group_token_count
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(group_order)
        return grouped_indices, grouped_tokens, group_order

    def _build_batches(self) -> list[list[int]]:
        grouped_indices, grouped_tokens, group_order = self._group_state()
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_tokens = 0
        for group_id in group_order:
            group_tokens = grouped_tokens[group_id]
            group_indices = grouped_indices[group_id]
            if current_batch and current_tokens + group_tokens > self.max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.extend(group_indices)
            current_tokens += group_tokens
        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())


class FixedBatchSizeSampler:
    def __init__(self, dataset_length: int, *, batch_size: int, shuffle: bool = False, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if dataset_length < 0:
            raise ValueError("dataset_length must be non-negative")
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def _build_batches(self) -> list[list[int]]:
        indices = list(range(self.dataset_length))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        return [indices[offset : offset + self.batch_size] for offset in range(0, len(indices), self.batch_size)]

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())


def build_group_batch_sampler(
    dataset: Dataset,
    *,
    max_tokens_per_batch: int,
    shuffle: bool,
    seed: int,
) -> TokenBudgetGroupBatchSampler:
    return TokenBudgetGroupBatchSampler(
        dataset["group_id"],
        [int(length) + 1 for length in dataset["length"]],
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=shuffle,
        seed=seed,
    )


def build_fixed_batch_size_sampler(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> FixedBatchSizeSampler:
    return FixedBatchSizeSampler(
        len(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
