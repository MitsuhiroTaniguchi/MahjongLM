from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .data import (
    PackedBatch,
    PackedGroupCollator,
    UnpackedBatch,
    UnpackedCollator,
    build_fixed_batch_size_sampler,
    build_group_batch_sampler,
    limit_groups,
    load_grouped_dataset,
)
from .model import load_saved_causal_lm
from tenhou_tokenizer import load_hf_tokenizer


def _require_eval_deps():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError("torch-backed transformers dependencies are required for evaluation") from exc
    return torch, DataLoader


def _select_device(torch, eval_device: str):
    if eval_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("eval_device=cuda was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _build_eval_loader(
    *,
    eval_dataset,
    packing_mode: str,
    eval_max_tokens_per_batch: int,
    per_device_eval_batch_size: int,
    collator,
    DataLoader,
):
    if packing_mode == "packed":
        batch_sampler = build_group_batch_sampler(
            eval_dataset,
            max_tokens_per_batch=eval_max_tokens_per_batch,
            shuffle=False,
            seed=1337,
        )
    else:
        batch_sampler = build_fixed_batch_size_sampler(
            eval_dataset,
            batch_size=per_device_eval_batch_size,
            shuffle=False,
            seed=1337,
        )
    return DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )


def evaluate_checkpoint(
    *,
    model_dir: Path,
    dataset_dirs: tuple[Path, ...],
    tokenizer_dir: Path,
    packing_mode: str,
    attn_implementation: str,
    max_seq_length: int,
    pad_to_multiple_of: int,
    eval_max_tokens_per_batch: int,
    per_device_eval_batch_size: int,
    max_eval_groups: int,
    eval_device: str,
) -> dict[str, float]:
    torch, DataLoader = _require_eval_deps()
    device = _select_device(torch, eval_device)
    tokenizer = load_hf_tokenizer(tokenizer_dir)
    eval_dataset = load_grouped_dataset(dataset_dirs)
    eval_dataset = limit_groups(eval_dataset, max_eval_groups)

    if packing_mode == "packed":
        collator = PackedGroupCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_seq_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
    else:
        collator = UnpackedCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_seq_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
    eval_loader = _build_eval_loader(
        eval_dataset=eval_dataset,
        packing_mode=packing_mode,
        eval_max_tokens_per_batch=eval_max_tokens_per_batch,
        per_device_eval_batch_size=per_device_eval_batch_size,
        collator=collator,
        DataLoader=DataLoader,
    )

    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    model = load_saved_causal_lm(
        model_dir=model_dir,
        attn_implementation=attn_implementation,
        dtype=autocast_dtype,
    )
    model.to(device)
    model.eval()

    losses: list[float] = []
    packed_tokens = 0
    padded_tokens = 0
    with torch.inference_mode():
        for batch in eval_loader:
            assert isinstance(batch, (PackedBatch, UnpackedBatch))
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                outputs = model(
                    input_ids=batch.input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                    labels=batch.labels.to(device),
                )
            losses.append(float(outputs.loss.detach().cpu()))
            packed_tokens += batch.stats.packed_tokens
            padded_tokens += batch.stats.padded_tokens
            del outputs, batch

    mean_loss = sum(losses) / len(losses)
    metrics = {
        "eval/loss": mean_loss,
        "eval/perplexity": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
        "eval/packing_efficiency": packed_tokens / padded_tokens if padded_tokens else 0.0,
    }

    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved GPT-2 or Qwen3 checkpoint on a MahjongLM dataset.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", action="append", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--packing-mode", type=str, default="packed", choices=["packed", "unpadded"])
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--eval-max-tokens-per-batch", type=int, default=131072)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--max-eval-groups", type=int, default=0)
    parser.add_argument("--eval-device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        model_dir=args.model_dir,
        dataset_dirs=tuple(args.dataset_dir),
        tokenizer_dir=args.tokenizer_dir,
        packing_mode=args.packing_mode,
        attn_implementation=args.attn_implementation,
        max_seq_length=args.max_seq_length,
        pad_to_multiple_of=args.pad_to_multiple_of,
        eval_max_tokens_per_batch=args.eval_max_tokens_per_batch,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_eval_groups=args.max_eval_groups,
        eval_device=args.eval_device,
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
