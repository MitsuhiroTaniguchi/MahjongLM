from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from transformers import AutoModelForCausalLM

from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate same-seed MahjongLM rollouts for DPO data construction. "
            "Input JSONL rows must contain prompt_tokens with the omniscient wall prefix already inserted."
        )
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 1:
        raise ValueError("--batch-size must be at least 2 to create preferences")
    torch.manual_seed(args.seed)
    tokenizer = MahjongTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device.startswith("cuda"):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **model_kwargs)
    model.to(args.device)
    model.eval()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.prompts_jsonl.open("r", encoding="utf-8") as source, args.output_jsonl.open("w", encoding="utf-8") as sink:
        for prompt_index, line in enumerate(source):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_tokens = list(record["prompt_tokens"])
            if not prompt_tokens or prompt_tokens[0] != "<bos>":
                prompt_tokens.insert(0, "<bos>")
            input_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
            if any(token_id == tokenizer.unk_token_id for token_id in input_ids):
                raise ValueError(f"prompt {prompt_index} contains unknown tokens")
            batch = torch.tensor([input_ids] * args.batch_size, dtype=torch.long, device=args.device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=batch,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            generations = []
            for generation_index, ids in enumerate(generated.detach().cpu().tolist()):
                if tokenizer.eos_token_id in ids:
                    ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
                tokens = tokenizer.convert_ids_to_tokens(ids)
                generations.append({"generation_index": generation_index, "tokens": tokens})
            sink.write(
                json.dumps(
                    {
                        "seed_id": str(record.get("seed_id", record.get("game_id", prompt_index))),
                        "rule_key": str(record.get("rule_key", "")),
                        "generations": generations,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()


if __name__ == "__main__":
    # Keep this script opt-in: generation is intentionally never launched by the pipeline unless called directly.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
