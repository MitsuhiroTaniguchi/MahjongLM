from __future__ import annotations

import importlib
from pathlib import Path

from datasets import Dataset

from gpt2.config import TrainingConfig
from gpt2.train import _build_cosine_scheduler_with_floor, _prepare_train_eval_datasets


def _dataset_rows() -> list[dict]:
    return [
        {"group_id": "g0", "seat_count": 4, "view_type": "complete", "viewer_seat": -1, "length": 3, "input_ids": [4, 5, 6]},
        {"group_id": "g0", "seat_count": 4, "view_type": "imperfect", "viewer_seat": 0, "length": 3, "input_ids": [7, 8, 9]},
        {"group_id": "g0", "seat_count": 4, "view_type": "imperfect", "viewer_seat": 1, "length": 3, "input_ids": [10, 11, 12]},
        {"group_id": "g0", "seat_count": 4, "view_type": "imperfect", "viewer_seat": 2, "length": 3, "input_ids": [13, 14, 15]},
        {"group_id": "g0", "seat_count": 4, "view_type": "imperfect", "viewer_seat": 3, "length": 3, "input_ids": [16, 17, 18]},
        {"group_id": "g1", "seat_count": 3, "view_type": "complete", "viewer_seat": -1, "length": 2, "input_ids": [21, 22]},
        {"group_id": "g1", "seat_count": 3, "view_type": "imperfect", "viewer_seat": 0, "length": 2, "input_ids": [23, 24]},
        {"group_id": "g1", "seat_count": 3, "view_type": "imperfect", "viewer_seat": 1, "length": 2, "input_ids": [25, 26]},
        {"group_id": "g1", "seat_count": 3, "view_type": "imperfect", "viewer_seat": 2, "length": 2, "input_ids": [27, 28]},
    ]


def test_scheduler_initializes_optimizer_at_first_warmup_lr() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = _build_cosine_scheduler_with_floor(
        optimizer,
        warmup_steps=4,
        total_steps=8,
        min_lr_ratio=0.1,
        torch=torch,
    )

    assert optimizer.param_groups[0]["lr"] == 0.25

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_prepare_train_eval_datasets_allows_zero_eval_ratio(monkeypatch) -> None:
    dataset = Dataset.from_list(_dataset_rows())
    train_module = importlib.import_module("gpt2.train")
    monkeypatch.setattr(train_module, "load_grouped_dataset", lambda _paths: dataset)

    config = TrainingConfig(
        output_dir=Path("out"),
        dataset_dirs=(Path("dataset"),),
        train_split_eval_ratio=0.0,
    )

    train_dataset, eval_dataset, split_metadata = _prepare_train_eval_datasets(config)

    assert train_dataset.num_rows == dataset.num_rows
    assert eval_dataset is None
    assert split_metadata["eval_rows"] == 0
