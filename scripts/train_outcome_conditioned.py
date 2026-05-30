"""Full fine-tuning of MahjongLM with *improved* outcome conditioning.

This is the v2 trainer that addresses the failure diagnosed in
docs/research/outcome_conditioned_policy_analysis.md: the v1 LoRA adapter was
outcome-*insensitive* (conditioning on viewer=1st vs 4th produced near-identical
policies). The fixes here:

  1. Outcome condition is re-injected close to every decision
     (``--reinject turn`` injects the viewer's target rank before every viewer
     draw), so a 10M model can actually route the signal to each decision.
  2. Conditioning tokens are masked from the loss -> we train a conditional
     model ``P(trajectory | outcome)`` instead of also predicting the outcome.
  3. Full fine-tuning (not LoRA) for maximum capacity to absorb conditioning.

Plain Hugging Face ``Trainer`` (no Unsloth) for robustness on the tiny custom
Qwen3 config. bf16 + SDPA attention on a single GPU.

Example (smoke, uses cached arrow shards directly):

    PYTHONPATH=src python scripts/train_outcome_conditioned.py \
        --shard 2024/data-00000-of-00016.arrow --max-steps 60 \
        --output-dir outputs/oc-smoke --wandb-mode disabled

Example (real run):

    PYTHONPATH=src python scripts/train_outcome_conditioned.py \
        --dataset-year 2020 --dataset-year 2021 --dataset-year 2022 \
        --dataset-year 2023 --dataset-year 2024 \
        --reinject turn --finetune full --num-train-epochs 1 \
        --output-dir outputs/mahjonglm-10m-oc-v2 --require-wandb
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Imported at module level so DataLoader worker processes can pickle the collator.
from gpt2.outcome_conditioning import (
    build_outcome_conditioned_v2,
    build_outcome_conditioned_v3,
    extend_tokenizer_for_v3,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="mitsutani/mahjonglm-10m")
    p.add_argument("--dataset-repo", default="mitsutani/mahjonglm-dataset")
    p.add_argument("--dataset-year", action="append", type=int, default=[])
    p.add_argument("--dataset-dir", action="append", type=Path, default=[])
    p.add_argument("--shard", action="append", default=[],
                   help="Load specific arrow shard files directly (smoke/dev). Repeatable.")
    p.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mahjonglm-10m-oc-v2")
    p.add_argument("--scheme", choices=("v2", "v3"), default="v3",
                   help="v2 = terminal final-rank conditioning; v3 = proximal per-round "
                        "score-delta-bucket conditioning (higher action-influence).")
    p.add_argument("--reinject", choices=("none", "round", "turn"), default="turn")
    p.add_argument("--loss-scope", choices=("full", "actions"), default="actions",
                   help="'actions' computes loss only on the viewer's decision tokens "
                        "(discards, take/pass, chi_pos, red use, kan/penuki target); "
                        "'full' uses every game token (v1 behaviour).")
    p.add_argument("--finetune", choices=("full", "freeze-embed"), default="full")
    p.add_argument("--max-seq-length", type=int, default=3072)
    p.add_argument("--attn", default="sdpa")
    # data limits / split
    p.add_argument("--max-train-groups", type=int, default=0)
    p.add_argument("--max-eval-groups", type=int, default=512)
    p.add_argument("--eval-ratio", type=float, default=0.01)
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument("--max-source-rows-per-dir", type=int, default=0)
    # optim
    p.add_argument("--per-device-train-batch-size", type=int, default=12)
    p.add_argument("--per-device-eval-batch-size", type=int, default=12)
    p.add_argument("--gradient-accumulation-steps", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--dataloader-num-workers", type=int, default=2)
    # wandb / hub
    p.add_argument("--wandb-project", default="mahjongLM_outcome_conditioned")
    p.add_argument("--wandb-entity", default="a21-3jck-")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--require-wandb", action="store_true")
    p.add_argument("--push-to-hub-repo", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_wandb and args.wandb_mode != "online":
        raise RuntimeError("--require-wandb requires --wandb-mode online")
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from tenhou_tokenizer.huggingface import MahjongTokenizerFast

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = MahjongTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added = 0
    if args.scheme == "v3":
        n_added = extend_tokenizer_for_v3(tokenizer)
        print(f"[v3] added {n_added} round-outcome tokens; tokenizer len={len(tokenizer)}")

    dataset = load_dataset(args)
    group_col = "group_id" if "group_id" in dataset.column_names else "game_id"
    train_ds, eval_ds = split_by_group(dataset, group_col, args.eval_ratio, args.split_seed)
    train_ds = limit_groups(train_ds, args.max_train_groups, group_col)
    eval_ds = limit_groups(eval_ds, args.max_eval_groups, group_col)
    print(f"[data] train_rows={len(train_ds)} eval_rows={len(eval_ds)} reinject={args.reinject}")

    collator = V2Collator(tokenizer, args.reinject, args.max_seq_length, args.loss_scope, args.scheme)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    if args.scheme == "v3" and model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        old_vocab = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        # Warm-init the new round-outcome rows to the mean of existing embeddings
        # (+ tiny noise) so they start in-distribution rather than cold/random.
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean = emb[:old_vocab].mean(0)
            emb[old_vocab:] = mean.unsqueeze(0) + 0.01 * torch.randn_like(emb[old_vocab:])
        print(f"[v3] resized embeddings {old_vocab} -> {len(tokenizer)} (warm mean-init for {n_added} new rows)")
    if args.finetune == "freeze-embed":
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                param.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params = {n_trainable/1e6:.2f}M  attn={args.attn}")

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if len(eval_ds) else "no",
        save_total_limit=args.save_total_limit,
        bf16=True,
        tf32=True,
        optim="adamw_torch",
        prediction_loss_only=True,  # never gather full-vocab logits during eval (OOM guard)
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        report_to=[] if args.wandb_mode == "disabled" else ["wandb"],
        run_name=args.wandb_run_name or None,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) else None,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    if len(eval_ds):
        trainer.evaluate(metric_key_prefix="eval")
    final_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] saved to {final_dir}")
    if args.push_to_hub_repo:
        model.push_to_hub(args.push_to_hub_repo)
        tokenizer.push_to_hub(args.push_to_hub_repo)
        print(f"[done] pushed to {args.push_to_hub_repo}")


class V2Collator:
    def __init__(self, tokenizer, reinject: str, max_seq_length: int, loss_scope: str = "actions",
                 scheme: str = "v3") -> None:
        self.tok = tokenizer
        self.reinject = reinject
        self.max_seq_length = max_seq_length
        self.loss_scope = loss_scope
        self.scheme = scheme
        self.pad_id = tokenizer.pad_token_id

    def build(self, feature: dict):
        builder = build_outcome_conditioned_v3 if self.scheme == "v3" else build_outcome_conditioned_v2
        ids, mask = builder(
            [int(t) for t in feature["input_ids"]],
            tokenizer=self.tok,
            seat_count=int(feature["seat_count"]),
            viewer_seat=int(feature["viewer_seat"]),
            reinject=self.reinject,
            loss_scope=self.loss_scope,
        )
        return ids, mask

    def __call__(self, features: list[dict]) -> dict:
        import torch

        built = []
        for f in features:
            ids, mask = self.build(f)
            if len(ids) > self.max_seq_length:
                continue
            built.append((ids, mask))
        if not built:
            # fall back to truncating the longest so the batch is never empty
            ids, mask = self.build(features[0])
            ids, mask = ids[: self.max_seq_length], mask[: self.max_seq_length]
            built.append((ids, mask))
        max_len = max(len(ids) for ids, _ in built)
        max_len += (-max_len) % 8
        input_ids, attn, labels = [], [], []
        for ids, mask in built:
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append([(i if m else -100) for i, m in zip(ids, mask)] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_dataset(args):
    from datasets import Dataset, concatenate_datasets, load_from_disk

    parts = []
    if args.shard:
        for shard in args.shard:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(args.dataset_repo, shard, repo_type="dataset")
            ds = Dataset.from_file(path)
            parts.append(ds.filter(lambda v: v == "imperfect", input_columns=["view_type"]))
    else:
        dirs = list(args.dataset_dir) or download_years(args)
        for d in dirs:
            ds = load_from_disk(str(d))
            if args.max_source_rows_per_dir > 0:
                ds = ds.select(range(min(len(ds), args.max_source_rows_per_dir)))
            parts.append(ds.filter(lambda v: v == "imperfect", input_columns=["view_type"]))
    if not parts:
        raise ValueError("no data sources; pass --shard, --dataset-dir or --dataset-year")
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if len(ds) == 0:
        raise ValueError("filtered imperfect dataset is empty")
    return ds


def download_years(args):
    from huggingface_hub import snapshot_download

    years = args.dataset_year or list(range(2011, 2025))
    snapshot = Path(snapshot_download(
        repo_id=args.dataset_repo, repo_type="dataset",
        allow_patterns=[f"{y}/*" for y in years],
    ))
    return [snapshot / str(y) for y in years]


def split_by_group(dataset, group_col, eval_ratio, seed):
    if eval_ratio <= 0:
        return dataset, dataset.select([])
    threshold = max(1, min(999_999, math.floor(eval_ratio * 1_000_000)))

    def is_eval(gid: str) -> bool:
        payload = f"{seed}:{gid}".encode()
        v = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % 1_000_000
        return v < threshold

    train = dataset.filter(lambda g: not is_eval(g), input_columns=[group_col])
    ev = dataset.filter(is_eval, input_columns=[group_col])
    return train, ev


def limit_groups(dataset, max_groups, group_col):
    if max_groups <= 0:
        return dataset
    chosen: set[str] = set()
    for gid in dataset[group_col]:
        chosen.add(gid)
        if len(chosen) >= max_groups:
            break
    return dataset.filter(lambda g: g in chosen, input_columns=[group_col])


if __name__ == "__main__":
    main()

