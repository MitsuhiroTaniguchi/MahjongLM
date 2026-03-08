from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tenhou_tokenizer import TenhouTokenizer
from tests.dataset_sample import DATASET_2023_CURATED_GAME_IDS
from tests.validation_helpers import trace_round_token_slices

ZIP_PATH = ROOT / "data" / "raw" / "tenhou" / "data2023.zip"
OUT_DIR = ROOT / "data" / "processed" / "tenhou" / "inspection"
GAME_IDS = list(DATASET_2023_CURATED_GAME_IDS)


def _distinct_action_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token.startswith("take_self_") or token.startswith("take_react_") or token.startswith("pingju_"):
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out


def _round_trace_payload(game: dict) -> list[dict]:
    traces: list[dict] = []
    for round_index, round_data in enumerate(game.get("log", [])):
        _tokenizer, round_traces = trace_round_token_slices(round_data)
        traces.append(
            {
                "round_index": round_index,
                "events": round_traces,
            }
        )
    return traces


def main() -> None:
    if not ZIP_PATH.exists():
        raise SystemExit(f"missing zip: {ZIP_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for game_id in GAME_IDS:
            game = json.load(zf.open(game_id))
            tokens = TenhouTokenizer().tokenize_game(game)
            slug = Path(game_id).stem
            json_path = OUT_DIR / f"{slug}.json"
            tokens_path = OUT_DIR / f"{slug}.tokens.txt"
            events_path = OUT_DIR / f"{slug}.events.json"

            json_payload = {
                "source_zip": str(ZIP_PATH),
                "game_id": game_id,
                "token_count": len(tokens),
                "distinct_action_tokens": _distinct_action_tokens(tokens),
                "tokens": tokens,
            }
            events_payload = {
                "source_zip": str(ZIP_PATH),
                "game_id": game_id,
                "rounds": _round_trace_payload(game),
            }

            json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tokens_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
            events_path.write_text(json.dumps(events_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            entries.append(
                {
                    "game_id": game_id,
                    "slug": slug,
                    "token_count": len(tokens),
                    "distinct_action_tokens": json_payload["distinct_action_tokens"],
                    "json_path": str(json_path),
                    "tokens_path": str(tokens_path),
                    "events_path": str(events_path),
                }
            )

    (OUT_DIR / "index.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "README.txt").write_text(
        "Selected curated real-game tokenization samples with varied and rare actions.\n"
        "Open index.json first, then inspect each *.tokens.txt, *.json, or *.events.json file.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
