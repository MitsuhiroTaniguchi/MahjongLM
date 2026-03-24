from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from gpt2.config import TinyGPT2Config, TinyQwen3Config
    from gpt2.model import build_tiny_gpt2_model, build_tiny_qwen3_model, count_parameters
    from tenhou_tokenizer import load_hf_tokenizer

    tokenizer = load_hf_tokenizer(Path("tokenizer"))
    common = dict(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa",
        dtype=None,
    )

    gpt2_config = TinyGPT2Config(
        n_layer=19,
        n_head=10,
        n_embd=640,
        n_inner=None,
        n_positions=8192,
    )
    qwen3_config = TinyQwen3Config(
        hidden_size=640,
        intermediate_size=1920,
        num_hidden_layers=20,
        num_attention_heads=10,
        num_key_value_heads=5,
        head_dim=64,
        max_position_embeddings=8192,
    )

    gpt2_model = build_tiny_gpt2_model(config=gpt2_config, **common)
    qwen3_model = build_tiny_qwen3_model(config=qwen3_config, **common)

    payload = {
        "vocab_size": tokenizer.vocab_size,
        "gpt2_100m": {
            "label": "G100",
            "parameter_count": count_parameters(gpt2_model),
            **asdict(gpt2_config),
        },
        "qwen3_100m": {
            "label": "Q100",
            "parameter_count": count_parameters(qwen3_model),
            **asdict(qwen3_config),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
