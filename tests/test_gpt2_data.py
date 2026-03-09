from __future__ import annotations

from datasets import Dataset

from gpt2.data import PackedGroupCollator, split_grouped_dataset, validate_grouped_dataset


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


def test_validate_grouped_dataset_accepts_complete_plus_all_views() -> None:
    dataset = Dataset.from_list(_dataset_rows())
    validate_grouped_dataset(dataset)


def test_validate_grouped_dataset_rejects_empty_dataset() -> None:
    dataset = Dataset.from_list([])
    try:
        validate_grouped_dataset(dataset)
    except ValueError as exc:
        assert "dataset is empty" in str(exc)
    else:
        raise AssertionError("expected empty dataset validation to fail")


def test_validate_grouped_dataset_rejects_missing_view() -> None:
    dataset = Dataset.from_list(_dataset_rows()[:-1])
    try:
        validate_grouped_dataset(dataset)
    except ValueError as exc:
        assert "expected 4 views" in str(exc)
    else:
        raise AssertionError("expected missing view validation to fail")


def test_packed_group_collator_never_cohabits_same_group() -> None:
    collator = PackedGroupCollator(
        pad_token_id=0,
        eos_token_id=3,
        max_length=16,
        pad_to_multiple_of=4,
        return_tensors="np",
    )
    batch = collator(
        [
            {"group_id": "g0", "view_type": "complete", "viewer_seat": -1, "input_ids": [10, 11, 12]},
            {"group_id": "g0", "view_type": "imperfect", "viewer_seat": 0, "input_ids": [13, 14, 15]},
            {"group_id": "g1", "view_type": "complete", "viewer_seat": -1, "input_ids": [21, 22]},
            {"group_id": "g1", "view_type": "imperfect", "viewer_seat": 0, "input_ids": [23, 24]},
        ]
    )

    assert batch.input_ids.shape[1] % 4 == 0
    for packed_group_ids in batch.packed_group_ids:
        assert len(packed_group_ids) == len(set(packed_group_ids))
    assert batch.stats.segment_count == 4
    assert batch.stats.packed_row_count >= 2


def test_packed_group_collator_rejects_sequences_longer_than_context() -> None:
    collator = PackedGroupCollator(
        pad_token_id=0,
        eos_token_id=3,
        max_length=4,
        return_tensors="np",
    )
    try:
        collator([{"group_id": "g0", "view_type": "complete", "viewer_seat": -1, "input_ids": [1, 2, 3, 4]}])
    except ValueError as exc:
        assert "exceeds max_length" in str(exc)
    else:
        raise AssertionError("expected overlong segment to fail")


def test_split_grouped_dataset_keeps_groups_intact() -> None:
    dataset = Dataset.from_list(_dataset_rows())
    train_dataset, eval_dataset = split_grouped_dataset(dataset, eval_ratio=0.5, seed=123)

    assert set(train_dataset["group_id"]).isdisjoint(set(eval_dataset["group_id"]))
