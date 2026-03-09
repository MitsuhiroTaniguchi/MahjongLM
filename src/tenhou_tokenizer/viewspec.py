from __future__ import annotations

from dataclasses import dataclass


VIEW_COMPLETE = "complete"
VIEW_IMPERFECT = "imperfect"
TOKEN_VIEW_COMPLETE = "view_complete"
TOKEN_VIEW_IMPERFECT = "view_imperfect"


@dataclass(frozen=True)
class ViewArtifactSpec:
    game_id: str
    view_type: str
    viewer_seat: int | None


def view_artifact_name(game_id: str, view_type: str, viewer_seat: int | None = None) -> str:
    if view_type == VIEW_COMPLETE:
        return f"{game_id}__complete.ids.bin"
    if view_type == VIEW_IMPERFECT and viewer_seat is not None:
        return f"{game_id}__player_{viewer_seat}.ids.bin"
    raise ValueError(f"invalid view artifact spec: view_type={view_type} viewer_seat={viewer_seat}")


def parse_view_artifact_name(filename: str) -> ViewArtifactSpec:
    if not filename.endswith(".ids.bin") or "__" not in filename:
        raise ValueError(f"invalid tokenized view filename: {filename}")
    stem = filename.removesuffix(".ids.bin")
    game_id, suffix = stem.rsplit("__", 1)
    if suffix == "complete":
        return ViewArtifactSpec(game_id=game_id, view_type=VIEW_COMPLETE, viewer_seat=None)
    if suffix.startswith("player_"):
        viewer_seat = int(suffix.removeprefix("player_"))
        return ViewArtifactSpec(game_id=game_id, view_type=VIEW_IMPERFECT, viewer_seat=viewer_seat)
    raise ValueError(f"invalid tokenized view suffix: {filename}")
