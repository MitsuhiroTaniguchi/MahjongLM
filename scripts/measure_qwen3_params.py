from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from gpt2.config import TinyQwen3Config
    from gpt2.model import build_tiny_qwen3_model, count_parameters
    from tenhou_tokenizer import load_hf_tokenizer

    tokenizer = load_hf_tokenizer(Path("tokenizer"))
    vocab_size = tokenizer.vocab_size

    presets = {
        "Q0": TinyQwen3Config(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=31,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=8192,
        ),
        "Q1": TinyQwen3Config(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=32,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=8192,
        ),
        "Q2": TinyQwen3Config(
            hidden_size=1088,
            intermediate_size=3264,
            num_hidden_layers=28,
            num_attention_heads=17,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=8192,
        ),
        "Q3": TinyQwen3Config(
            hidden_size=1056,
            intermediate_size=3168,
            num_hidden_layers=30,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=8192,
        ),
        "Q4": TinyQwen3Config(
            hidden_size=1152,
            intermediate_size=3456,
            num_hidden_layers=25,
            num_attention_heads=18,
            num_key_value_heads=9,
            head_dim=128,
            max_position_embeddings=8192,
        ),
    }

    rows: list[dict[str, object]] = []
    for name, config in presets.items():
        model = build_tiny_qwen3_model(
            vocab_size=vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation="sdpa",
            dtype=None,
            config=config,
        )
        rows.append(
            {
                "label": name,
                "parameter_count": count_parameters(model),
                **asdict(config),
            }
        )
        del model

    print(json.dumps({"vocab_size": vocab_size, "presets": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
