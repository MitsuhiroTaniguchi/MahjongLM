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
from tests.dataset_sample import DATASET_2023_CURATED_GAME_IDS, get_dataset_2023_curated_zip
from tests.validation_helpers import trace_round_token_slices

OUT_DIR = ROOT / "data" / "processed" / "tenhou" / "inspection"
FOCUSED_DIR = OUT_DIR / "focused"
GAME_IDS = list(DATASET_2023_CURATED_GAME_IDS)
EXTRA_GAME_JSONS = [
    ROOT / "tests" / "fixtures" / "tenhou" / "2023010100gm-00f1-0000-caffca62.json",
    ROOT / "tests" / "fixtures" / "tenhou" / "2023012721gm-00e1-0000-72b6b9d6.json",
]
FOCUSED_SPECS = [
    {"label": "sukantsu_kaigang", "game_id": "2023/2023021117gm-00a9-0000-d0a6d793.txt", "round_index": 8},
    {"label": "sanchahou", "game_id": "2023/2023052310gm-00a9-0000-2faa0711.txt", "round_index": 6},
    {"label": "suuchariichi_kaigang", "game_id": "2023/2023081821gm-00a9-0000-7cf4f26e.txt", "round_index": 1},
    {"label": "multi_ron_kaigang", "game_id": "2023/2023073004gm-00a9-0000-ddb6dc93.txt", "round_index": 1},
    {"label": "multi_ron", "game_id": "2023/2023091810gm-00a9-0000-1428c108.txt", "round_index": 6},
]
EXTRA_FOCUSED_SPECS = [
    {
        "label": "sanma_tonpu",
        "json_path": ROOT / "tests" / "fixtures" / "tenhou" / "2023010100gm-00f1-0000-caffca62.json",
        "round_index": 0,
    },
    {
        "label": "yonma_tonpu",
        "json_path": ROOT / "tests" / "fixtures" / "tenhou" / "2023012721gm-00e1-0000-72b6b9d6.json",
        "round_index": 0,
    },
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
        lines.append(f"- [{event['event_index']}] {event['event_key']}:")
        lines.append(f"  value: {json.dumps(event['event_value'], ensure_ascii=False)}")
        lines.append("  tokens:")
        lines.extend(f"    {token}" for token in event["tokens"])
    lines.append("")
    lines.append("[views]")
    for view in payload["views"]:
        seat_label = "complete" if view["viewer_seat"] is None else f"seat_{view['viewer_seat']}"
        lines.append(f"- {view['view_type']} ({seat_label}) token_count={view['token_count']}")
        lines.append("  tokens:")
        lines.extend(f"    {token}" for token in view["tokens"])
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


def _write_game_views(source_label: str, game_id: str, game: dict) -> dict:
    views = tokenize_game_views(game)
    slug = Path(game_id).stem
    view_entries: list[dict] = []
    for view in views:
        view_slug = _view_slug(slug, view.view_type, view.viewer_seat)
        view_path = OUT_DIR / f"{view_slug}.tokens.txt"
        view_path.write_text("\n".join(view.tokens) + "\n", encoding="utf-8")
        view_entries.append(
            {
                "view_type": view.view_type,
                "viewer_seat": view.viewer_seat,
                "token_count": len(view.tokens),
                "distinct_action_tokens": _distinct_action_tokens(view.tokens),
                "tokens_path": str(view_path.relative_to(OUT_DIR)),
            }
        )
    return {
        "source": source_label,
        "game_id": game_id,
        "title": game.get("title", ""),
        "slug": slug,
        "views": view_entries,
    }


def _format_index(entries: list[dict], focused_entries: list[dict]) -> str:
    lines = [
        "Selected real-game inspection samples.",
        "Files are emitted as plain text only.",
        "",
        "[games]",
    ]
    for entry in entries:
        lines.append(f"- {entry['game_id']} ({entry['title']}) source={entry['source']}")
        for view in entry["views"]:
            seat = "complete" if view["viewer_seat"] is None else f"seat_{view['viewer_seat']}"
            lines.append(
                f"  {view['view_type']} ({seat}) token_count={view['token_count']} file={view['tokens_path']}"
            )
    lines.append("")
    lines.append("[focused]")
    for entry in focused_entries:
        lines.append(
            f"- {entry['label']} game_id={entry['game_id']} round_index={entry['round_index']} "
            f"file=focused/{entry['text_path']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ZIP_PATH = get_dataset_2023_curated_zip()
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
            entries.append(_write_game_views(str(ZIP_PATH.name), game_id, game))

        for spec in FOCUSED_SPECS:
            game = json.load(zf.open(spec["game_id"]))
            slug = Path(spec["game_id"]).stem
            focused_text_path = FOCUSED_DIR / f"{slug}.round{spec['round_index']}.{spec['label']}.txt"
            payload = {
                "label": spec["label"],
                **_focused_payload(spec["game_id"], game, spec["round_index"]),
            }
            focused_text_path.write_text(_format_focused_text(payload), encoding="utf-8")
            focused_entries.append(
                {
                    "label": spec["label"],
                    "game_id": spec["game_id"],
                    "round_index": spec["round_index"],
                    "text_path": focused_text_path.name,
                }
            )

    for json_path in EXTRA_GAME_JSONS:
        game = json.loads(json_path.read_text(encoding="utf-8"))
        entries.append(_write_game_views(json_path.name, json_path.name, game))

    for spec in EXTRA_FOCUSED_SPECS:
        game = json.loads(spec["json_path"].read_text(encoding="utf-8"))
        slug = spec["json_path"].stem
        focused_text_path = FOCUSED_DIR / f"{slug}.round{spec['round_index']}.{spec['label']}.txt"
        payload = {
            "label": spec["label"],
            **_focused_payload(spec["json_path"].name, game, spec["round_index"]),
        }
        focused_text_path.write_text(_format_focused_text(payload), encoding="utf-8")
        focused_entries.append(
            {
                "label": spec["label"],
                "game_id": spec["json_path"].name,
                "round_index": spec["round_index"],
                "text_path": focused_text_path.name,
            }
        )

    (OUT_DIR / "INDEX.txt").write_text(_format_index(entries, focused_entries), encoding="utf-8")
    (OUT_DIR / "README.txt").write_text(
        "Selected curated real-game tokenization samples.\n"
        "Each game includes complete and imperfect-information token views.\n"
        "Inspection output is plain text only. Start with INDEX.txt.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
