from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi


README_TEXT = """---
license: other
license_name: source-data-terms-apply
license_link: LICENSE
task_categories:
  - text-generation
language:
  - ja
pretty_name: MahjongLM Dataset
---

# MahjongLM Dataset

MahjongLM Dataset is a processed, pre-tokenized Japanese mahjong game-log corpus for causal language modeling.
It is intended for pretraining and evaluating MahjongLM-family models on long-form sequential prediction over riichi mahjong game records.

## Dataset Structure

- `2011/` ... `2024/`: yearly Hugging Face `Dataset` directories saved with `save_to_disk`
- `tokenizer/`: tokenizer assets used for MahjongLM training
- `README.md`: dataset card
- `LICENSE`: custom source-data license notice

## Features

- `game_id`
- `year`
- `seat_count`
- `view_type`
- `viewer_seat`
- `length`
- `input_ids`

## Splits

This repository stores the full yearly training corpora only.
Train/validation splits are created deterministically by the training pipeline and are not stored in the public dataset repository.

## Notes

- `group_id` is omitted from the public release.
- The dataset is pre-tokenized for MahjongLM training.
- `input_ids` are ready to use for causal language modeling without additional tokenization.

## Intended Use

- Pretraining and evaluating MahjongLM-family models
- Sequence modeling research on Japanese mahjong game logs
- Reproducible training runs across yearly subsets or the full 2011-2024 corpus

## License

`source-data-terms-apply`

This dataset is a processed derivative of third-party game log data. Use of this dataset is subject to the terms and restrictions of the original data source. Users are responsible for confirming that their usage complies with the source data terms. See `LICENSE` for the repository notice used for this release.

## Tokenizer

Tokenizer assets are included in `tokenizer/`:

- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`
"""

LICENSE_TEXT = """MahjongLM Dataset License Notice
=================================

Identifier: source-data-terms-apply

This repository contains a processed derivative of third-party game log data.

Use of this dataset is subject to the terms, restrictions, and any downstream
requirements imposed by the original data source. This repository does not
grant broader rights than those permitted by the source data terms.

By using, copying, redistributing, modifying, or training on this dataset, you
are responsible for ensuring that your use complies with the original source
terms and with any applicable laws, regulations, or platform policies.

No warranty is provided. The dataset is distributed on an "as is" basis.
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_export_dir() -> Path:
    return Path.home() / "hf_exports" / "mahjonglm-dataset"


def _is_within_repo(path: Path, repo_root: Path) -> bool:
    try:
        path.relative_to(repo_root)
        return True
    except ValueError:
        return False


def export_clean_dataset(source_dir: Path, export_dir: Path, tokenizer_dir: Path) -> None:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    year_dirs = sorted(
        path for path in source_dir.iterdir() if path.is_dir() and (path / "dataset_info.json").is_file()
    )
    if not year_dirs:
        raise RuntimeError(f"no yearly datasets found under {source_dir}")

    for year_dir in year_dirs:
        print(f"Exporting {year_dir.name} ...")
        dataset = load_from_disk(str(year_dir))
        removable = [column for column in ("group_id",) if column in dataset.column_names]
        if removable:
            dataset = dataset.remove_columns(removable)
        dataset.save_to_disk(str(export_dir / year_dir.name))

    tokenizer_export_dir = export_dir / "tokenizer"
    shutil.copytree(tokenizer_dir, tokenizer_export_dir, dirs_exist_ok=True)
    (export_dir / "README.md").write_text(README_TEXT, encoding="utf-8")
    (export_dir / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")


def clean_remote_repo(api: HfApi, repo_id: str, token: str) -> None:
    existing_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    year_dirs = sorted({path.split("/")[0] for path in existing_files if "/" in path and path.split("/")[0].isdigit()})
    for year in year_dirs:
        print(f"Deleting remote folder {year}/ ...")
        api.delete_folder(
            path_in_repo=year,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Remove old {year} dataset export",
        )
    if "tokenizer/tokenizer.json" in existing_files:
        print("Deleting remote folder tokenizer/ ...")
        api.delete_folder(
            path_in_repo="tokenizer",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Remove old tokenizer export",
        )
    if any(path.startswith(".ingested/") for path in existing_files):
        print("Deleting remote folder .ingested/ ...")
        api.delete_folder(
            path_in_repo=".ingested",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Remove internal ingestion markers",
        )
    if ".DS_Store" in existing_files:
        print("Deleting remote file .DS_Store ...")
        api.delete_file(
            path_in_repo=".DS_Store",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Remove stray macOS metadata file",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and upload the public MahjongLM dataset to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. mitsutani/mahjonglm-dataset")
    parser.add_argument("--token", required=True, help="HF write token")
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--export-dir", type=Path, default=_default_export_dir())
    args = parser.parse_args()

    repo_root = _repo_root()
    source_dir = args.source_dir.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    export_dir = args.export_dir.resolve()

    if _is_within_repo(export_dir, repo_root):
        raise RuntimeError(
            f"refusing to export inside the git repo: {export_dir}\n"
            f"choose a directory outside {repo_root}"
        )

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    export_clean_dataset(source_dir, export_dir, tokenizer_dir)
    clean_remote_repo(api, args.repo_id, args.token)

    print("Uploading cleaned dataset export ...")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(export_dir),
        ignore_patterns=[".DS_Store", "**/.DS_Store"],
    )


if __name__ == "__main__":
    main()
