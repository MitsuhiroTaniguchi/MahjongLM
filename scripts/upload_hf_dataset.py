from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token


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

MahjongLM Dataset is a processed, pre-tokenized riichi mahjong game-log corpus for causal language modeling.
It is built from Tenhou game logs and is intended for training MahjongLM-family models to predict long structured mahjong token streams.

The dataset is distributed as token IDs, not as raw Tenhou XML. The corresponding tokenizer assets are included in this repository.

## Dataset Structure

- `2011/` ... `2024/`: yearly Hugging Face `Dataset` directories saved with `datasets.Dataset.save_to_disk`
- `tokenizer/`: tokenizer assets used for MahjongLM training
- `README.md`: dataset card
- `LICENSE`: custom source-data license notice

## Rows and Features

Each row is one tokenized view of one game:

- `game_id`
- `year`
- `seat_count`
- `view_type`
- `viewer_seat`
- `length`
- `input_ids`

`group_id` is used internally during dataset construction and training, but is omitted from the public export.

## Views

`view_type` is one of:

- `complete`: complete public information for the game.
- `imperfect`: player-perspective information. `viewer_seat` identifies the visible player's seat.
- `omniscient`: the `complete` view plus a reconstructed full wall block for each hand.

The `omniscient` view inserts:

```text
round_start wall <wall tile tokens> ...
```

For four-player games, the wall block contains 136 physical tiles including red fives (`m0`, `p0`, `s0`).
For three-player games, the wall block contains the 108 tiles used by Tenhou sanma: `m1`, `m9`, all pinzu, all souzu, and honors, with red fives for pinzu and souzu.
Wall tokens are stored in actual consumption order, so the beginning of the block corresponds to the initial deal and subsequent live-wall draws.
Omniscient rows are emitted only when the Tenhou shuffle seed is available and the reconstructed wall order is consistent with the observed initial hands, live-wall draws, replacement draws, dora indicators, ura-dora indicators, and result data.

## Splits

This repository stores the full yearly training corpora only.
Train/validation splits are created deterministically by the training pipeline and are not stored in the public dataset repository.

## Token Stream Outline

Sequences are stored without `<bos>` and `<eos>` in `input_ids`; MahjongLM training code adds those boundary tokens at training time.

A game begins with rule and view tokens, then `game_start`, then one or more hand blocks:

```text
rule_* view_* game_start round_start ... round_end ... game_end
```

Important conventions:

- `round_start` / `round_end` delimit each hand; `game_start` / `game_end` delimit the game log.
- `round_end` and `game_end` are emitted after rank tokens and before final-score tokens at game end.
- `seat` numbers are table-relative seats in the tokenized game. `player` identifiers, when present in token names, are original player indices from the source game.
- Self decisions and reaction decisions are emitted as `opt_*` option tokens followed by matching `take_*` / `pass_*` resolution tokens in the same option order.
- Reaction options are ordered by `(priority, seat, tiebreak, option)`, with `ron` before `pon`/`minkan`, then `chi`.
- Self option priority is `tsumo > kyusyu > penuki > riichi > ankan > kakan`.
- `chi_pos_low` / `chi_pos_mid` / `chi_pos_high` describe the position of the consumed tiles, not the called discard.
- `red_used` / `red_not_used` is emitted whenever a call consumes a five tile; it records whether the consumed five was red.
- `take_self_*_tsumo` is followed by the winning tile token. The winning tile is excluded from the later `opened_hand_*` block.
- Win result blocks begin with one or more `hule_{seat}` markers.
- For multiple ron wins, all `hule_{seat}` markers are emitted first, shared `ura_dora` reveal tiles are emitted once, then each winner detail follows, and one combined score-delta block is emitted after all win details.
- `ura_dora` is emitted only when relevant to a riichi win.
- Kan dora timing follows Tenhou behavior: closed kan reveals immediately before the replacement draw; open kan / added kan reveal after the following discard or before a chained replacement draw. If multiple dora indicators are revealed together, they are emitted as one `dora` token followed by all revealed tile tokens.
- Kyushukyuhai result `opened_hand_*` contains exactly 14 tiles including the draw tile. North extraction (`penuki`) is treated like a call for first-turn kyushukyuhai suppression.

## Vocabulary Summary

The vocabulary contains:

- special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- view and rule tokens
- game and hand boundary tokens
- physical tile tokens, including red fives
- draw, discard, call, decision, and pass/take tokens
- dora and ura-dora reveal tokens
- opened-hand, hule, yaku, fu, han, score, score-delta, rank, and final-score tokens
- point-stick tokens from 100-point through 10,000-point units

## Intended Use

- Pretraining and evaluating MahjongLM-family models
- Sequence modeling research on Japanese mahjong game logs
- Reproducible training runs across yearly subsets or the full 2011-2024 corpus

## Source Code

The dataset construction code is maintained at:

https://github.com/MitsuhiroTaniguchi/MahjongLM

## License

`source-data-terms-apply`

This dataset is a processed derivative of third-party Tenhou game log data. Use of this dataset is subject to Tenhou's source-data terms and restrictions. See `LICENSE` for the repository notice used for this release.

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

This repository contains a processed derivative of Tenhou game log data.
The raw source logs are published by Tenhou / C-EGG Inc. and are subject to
Tenhou's source-data terms and use restrictions.

Use of this dataset is subject to the terms, restrictions, and any downstream
requirements imposed by the original data source. This repository does not
grant broader rights than those permitted by the source data terms.

By using, copying, redistributing, modifying, or training on this dataset, you
are responsible for ensuring that your use complies with the original source
terms and with any applicable laws, regulations, or platform policies.

Tenhou source-data restrictions
-------------------------------

The following restrictions are quoted from Tenhou's published notes for using
game logs:

※天鳳と競合する製品への開発・応用を目的として牌譜を使用していただくことはできません。
※天鳳の牌譜は、天鳳での対戦を公正に楽しんでいただく目的で公開されています。天鳳での対戦を必要としないサービスへの応用は無償有償ともに行えません。一般の麻雀への応用を目的に牌譜を使用する場合は support@c-egg.com までお問い合わせください。
※不特定多数が天鳳の牌譜をダウンロードするサービスは作成できません。
※企業として利用する場合には協賛イベントの開催をお願いします。

Reference:

- https://tenhou.net/sc/raw/?old=
- https://tenhou.net/man/

No warranty is provided. The dataset is distributed on an "as is" basis.
"""

LICENSE_TEXT = """MahjongLM Dataset License Notice
=================================

Identifier: source-data-terms-apply

This repository contains a processed derivative of Tenhou game log data.
The raw source logs are published by Tenhou / C-EGG Inc. and are subject to
Tenhou's source-data terms and use restrictions.

Use of this dataset is subject to the terms, restrictions, and any downstream
requirements imposed by the original data source. This repository does not
grant broader rights than those permitted by the source data terms.

By using, copying, redistributing, modifying, or training on this dataset, you
are responsible for ensuring that your use complies with the original source
terms and with any applicable laws, regulations, or platform policies.

Tenhou source-data restrictions
-------------------------------

The following restrictions are quoted from Tenhou's published notes for using
game logs:

※天鳳と競合する製品への開発・応用を目的として牌譜を使用していただくことはできません。
※天鳳の牌譜は、天鳳での対戦を正常に楽しんでいただく目的で公開されています。天鳳での対戦を必要としないサービスへの応用は無償有償ともに行えません。一般の麻雀への応用を目的に牌譜を使用する場合は support@c-egg.com までお問い合わせください。
※不特定多数が天鳳の牌譜をダウンロードするサービスは作成できません。
※企業として利用する場合には協賛イベントの開催をお願いします。

Reference:

- https://tenhou.net/sc/raw/?old=
- https://tenhou.net/man/

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
        print(f"Exporting {year_dir.name} ...", flush=True)
        dataset = load_from_disk(str(year_dir))
        removable = [column for column in ("group_id",) if column in dataset.column_names]
        if removable:
            dataset = dataset.remove_columns(removable)
        dataset.save_to_disk(str(export_dir / year_dir.name), max_shard_size="512MB")
        print(f"Exported {year_dir.name}: {dataset.num_rows} rows", flush=True)

    tokenizer_export_dir = export_dir / "tokenizer"
    shutil.copytree(tokenizer_dir, tokenizer_export_dir, dirs_exist_ok=True)
    (export_dir / "README.md").write_text(README_TEXT, encoding="utf-8")
    (export_dir / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")


def clean_remote_repo(api: HfApi, repo_id: str, token: str) -> None:
    existing_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    year_dirs = sorted({path.split("/")[0] for path in existing_files if "/" in path and path.split("/")[0].isdigit()})
    for year in year_dirs:
        print(f"Deleting remote folder {year}/ ...", flush=True)
        api.delete_folder(
            path_in_repo=year,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Remove old {year} dataset export",
        )
    if "tokenizer/tokenizer.json" in existing_files:
        print("Deleting remote folder tokenizer/ ...", flush=True)
        api.delete_folder(
            path_in_repo="tokenizer",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Remove old tokenizer export",
        )
    if any(path.startswith(".ingested/") for path in existing_files):
        print("Deleting remote folder .ingested/ ...", flush=True)
        api.delete_folder(
            path_in_repo=".ingested",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Remove internal ingestion markers",
        )
    if ".DS_Store" in existing_files:
        print("Deleting remote file .DS_Store ...", flush=True)
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
    parser.add_argument("--token", default=None, help="HF write token. Defaults to HF_TOKEN/cache login.")
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

    token = args.token or get_token()
    if not token:
        raise RuntimeError("HF token not found. Set HF_TOKEN or run huggingface-cli login.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    export_clean_dataset(source_dir, export_dir, tokenizer_dir)
    clean_remote_repo(api, args.repo_id, token)

    print("Uploading cleaned dataset export ...", flush=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(export_dir),
        ignore_patterns=[".DS_Store", "**/.DS_Store"],
    )


if __name__ == "__main__":
    main()
