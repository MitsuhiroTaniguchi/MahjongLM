from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tenhou_tokenizer import tokenize_game_views
from tests.dataset_sample import DATASET_2023_CURATED_GAME_IDS
from tests.validation_helpers import trace_round_token_slices

ZIP_PATH = ROOT / "data" / "raw" / "tenhou" / "data2023.zip"
OUT_DIR = ROOT / "data" / "processed" / "tenhou" / "inspection"
FOCUSED_DIR = OUT_DIR / "focused"
GAME_IDS = list(DATASET_2023_CURATED_GAME_IDS)
FOCUSED_SPECS = [
    {"label": "sukantsu_kaigang", "game_id": "2023/2023021117gm-00a9-0000-d0a6d793.txt", "round_index": 8},
    {"label": "sanchahou", "game_id": "2023/2023052310gm-00a9-0000-2faa0711.txt", "round_index": 6},
    {"label": "suuchariichi_kaigang", "game_id": "2023/2023081821gm-00a9-0000-7cf4f26e.txt", "round_index": 1},
    {"label": "multi_ron_kaigang", "game_id": "2023/2023073004gm-00a9-0000-ddb6dc93.txt", "round_index": 1},
    {"label": "multi_ron", "game_id": "2023/2023091810gm-00a9-0000-1428c108.txt", "round_index": 6},
]


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


def _view_slug(slug: str, view_type: str, viewer_seat: int | None) -> str:
    if view_type == "complete":
        return f"{slug}.complete"
    return f"{slug}.player_{viewer_seat}"


def _format_focused_text(payload: dict) -> str:
    lines = [
        f"game_id: {payload['source_game_id']}",
        f"round_index: {payload['round_index']}",
        f"label: {payload['label']}",
        "",
        "[events]",
    ]
    for event in payload["events"]:
        lines.append(
            f"- [{event['event_index']}] {event['event_key']}: "
            f"{json.dumps(event['event_value'], ensure_ascii=False)}"
        )
        lines.append(f"  tokens: {' '.join(event['tokens'])}")
    lines.append("")
    lines.append("[views]")
    for view in payload["views"]:
        seat_label = "complete" if view["viewer_seat"] is None else f"seat_{view['viewer_seat']}"
        lines.append(f"- {view['view_type']} ({seat_label}) token_count={view['token_count']}")
        lines.append(f"  tokens: {' '.join(view['tokens'])}")
    lines.append("")
    return "\n".join(lines)


def _focused_payload(game_id: str, game: dict, round_index: int) -> dict:
    round_data = game["log"][round_index]
    _tokenizer, round_traces = trace_round_token_slices(round_data)
    views = tokenize_game_views({"log": [round_data], **{k: v for k, v in game.items() if k != "log"}})
    return {
        "source_game_id": game_id,
        "round_index": round_index,
        "events": round_traces,
        "views": [
            {
                "view_type": view.view_type,
                "viewer_seat": view.viewer_seat,
                "token_count": len(view.tokens),
                "tokens": view.tokens,
            }
            for view in views
        ],
    }


def main() -> None:
    if not ZIP_PATH.exists():
        raise SystemExit(f"missing zip: {ZIP_PATH}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FOCUSED_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    focused_entries: list[dict] = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for game_id in GAME_IDS:
            game = json.load(zf.open(game_id))
            views = tokenize_game_views(game)
            complete_view = views[0]
            tokens = complete_view.tokens
            slug = Path(game_id).stem
            json_path = OUT_DIR / f"{slug}.json"
            tokens_path = OUT_DIR / f"{slug}.complete.tokens.txt"
            events_path = OUT_DIR / f"{slug}.events.json"
            view_entries: list[dict] = []

            json_payload = {
                "source_zip": str(ZIP_PATH),
                "game_id": game_id,
                "token_count": len(tokens),
                "distinct_action_tokens": _distinct_action_tokens(tokens),
                "views": [],
            }
            events_payload = {
                "source_zip": str(ZIP_PATH),
                "game_id": game_id,
                "rounds": _round_trace_payload(game),
            }

            for view in views:
                view_slug = _view_slug(slug, view.view_type, view.viewer_seat)
                view_path = OUT_DIR / f"{view_slug}.tokens.txt"
                view_path.write_text("\n".join(view.tokens) + "\n", encoding="utf-8")
                view_entry = {
                    "view_type": view.view_type,
                    "viewer_seat": view.viewer_seat,
                    "token_count": len(view.tokens),
                    "distinct_action_tokens": _distinct_action_tokens(view.tokens),
                    "tokens_path": str(view_path),
                }
                view_entries.append(view_entry)
                json_payload["views"].append(view_entry)

            json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
                    "views": view_entries,
                }
            )

        for spec in FOCUSED_SPECS:
            game = json.load(zf.open(spec["game_id"]))
            slug = Path(spec["game_id"]).stem
            focused_json_path = FOCUSED_DIR / f"{slug}.round{spec['round_index']}.{spec['label']}.json"
            focused_text_path = FOCUSED_DIR / f"{slug}.round{spec['round_index']}.{spec['label']}.txt"
            payload = {
                "label": spec["label"],
                **_focused_payload(spec["game_id"], game, spec["round_index"]),
            }
            focused_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            focused_text_path.write_text(_format_focused_text(payload), encoding="utf-8")
            focused_entries.append(
                {
                    "label": spec["label"],
                    "game_id": spec["game_id"],
                    "round_index": spec["round_index"],
                    "json_path": str(focused_json_path),
                    "text_path": str(focused_text_path),
                }
            )

    (OUT_DIR / "index.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (FOCUSED_DIR / "index.json").write_text(json.dumps(focused_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "README.txt").write_text(
        "Selected curated real-game tokenization samples.\n"
        "Each game now includes complete and imperfect-information token views.\n"
        "Open index.json first, then inspect *.json, *.events.json, and *.tokens.txt files.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
