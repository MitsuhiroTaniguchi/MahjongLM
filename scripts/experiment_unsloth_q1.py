from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _add_src_to_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Q1 memory/runtime with and without Unsloth.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/qwen3-q1-init/model"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/unsloth-q1-probe/results.json"))
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hf_bf16", "unsloth_full16", "unsloth_4bit_lora"],
        choices=["hf_bf16", "unsloth_full16", "unsloth_4bit_lora"],
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--disable-unsloth-compile", action="store_true")
    return parser.parse_args()


def _release_cuda_memory(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _reset_peak_memory(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _cuda_memory_stats(torch) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "max_allocated_gib": 0.0,
            "max_reserved_gib": 0.0,
        }
    to_gib = 1024**3
    return {
        "allocated_gib": torch.cuda.memory_allocated() / to_gib,
        "reserved_gib": torch.cuda.memory_reserved() / to_gib,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / to_gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / to_gib,
    }


def _count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _make_batch(torch, *, batch_size: int, seq_length: int, vocab_size: int, device) -> dict[str, object]:
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch_size, seq_length), device=device, dtype=torch.long)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _run_single_step(torch, model, *, batch: dict[str, object], learning_rate: float) -> dict[str, float]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    start_time = time.perf_counter()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - start_time
    result = {
        "loss": float(loss.detach().cpu()),
        "step_time_sec": elapsed,
        "trainable_parameters": _count_trainable_parameters(model),
    }
    del outputs, loss, optimizer
    return result


def _load_hf_model(torch, checkpoint_dir: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()
    model.train()
    return model.cuda()


def _load_unsloth_full16(torch, checkpoint_dir: Path, *, seq_length: int):
    from unsloth import FastLanguageModel

    model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.train()
    return model


def _load_unsloth_4bit_lora(torch, checkpoint_dir: Path, *, seq_length: int):
    from unsloth import FastLanguageModel

    model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=seq_length,
        dtype=None,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.train()
    return model


def _run_mode(
    *,
    mode: str,
    checkpoint_dir: Path,
    seq_length: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    import torch

    _release_cuda_memory(torch)
    _reset_peak_memory(torch)
    config_payload = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(config_payload["vocab_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _make_batch(
        torch,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        device=device,
    )
    result: dict[str, object] = {
        "mode": mode,
        "seq_length": seq_length,
        "batch_size": batch_size,
    }
    model = None
    try:
        load_start = time.perf_counter()
        if mode == "hf_bf16":
            model = _load_hf_model(torch, checkpoint_dir)
        elif mode == "unsloth_full16":
            model = _load_unsloth_full16(torch, checkpoint_dir, seq_length=seq_length)
        elif mode == "unsloth_4bit_lora":
            model = _load_unsloth_4bit_lora(torch, checkpoint_dir, seq_length=seq_length)
        else:  # pragma: no cover - argparse should prevent this
            raise ValueError(f"unknown mode: {mode}")
        result["load_time_sec"] = time.perf_counter() - load_start
        result["load_memory"] = _cuda_memory_stats(torch)
        _reset_peak_memory(torch)
        step_metrics = _run_single_step(
            torch,
            model,
            batch=batch,
            learning_rate=learning_rate,
        )
        result.update(step_metrics)
        result["step_memory"] = _cuda_memory_stats(torch)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["step_memory"] = _cuda_memory_stats(torch)
    finally:
        del batch
        if model is not None:
            del model
        _release_cuda_memory(torch)
        result["memory_after_cleanup"] = _cuda_memory_stats(torch)
    return result


def main() -> None:
    args = _parse_args()
    if args.disable_unsloth_compile:
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    _add_src_to_path()
    results: dict[str, object] = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "disable_unsloth_compile": bool(args.disable_unsloth_compile),
        "results": [],
    }
    for mode in args.modes:
        results["results"].append(
            _run_mode(
                mode=mode,
                checkpoint_dir=args.checkpoint_dir,
                seq_length=args.seq_length,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
