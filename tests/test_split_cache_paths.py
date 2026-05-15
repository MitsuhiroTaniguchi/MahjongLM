from pathlib import Path

from gpt2.data import build_split_cache_paths


def _write_dataset_dir(path: Path, *, arrow_size: int = 4) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataset_info.json").write_text('{"features": {}}\n', encoding="utf-8")
    (path / "state.json").write_text('{"_data_files": []}\n', encoding="utf-8")
    (path / "data-00000-of-00001.arrow").write_bytes(b"x" * arrow_size)


def test_split_cache_key_changes_when_dataset_dir_changes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_dataset_dir(dataset_dir, arrow_size=4)

    first = build_split_cache_paths([dataset_dir], eval_ratio=0.001, seed=1337, cache_dir=cache_dir)

    _write_dataset_dir(dataset_dir, arrow_size=8)
    second = build_split_cache_paths([dataset_dir], eval_ratio=0.001, seed=1337, cache_dir=cache_dir)

    assert first != second
