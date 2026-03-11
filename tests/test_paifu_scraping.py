from __future__ import annotations

from pathlib import Path

from scripts.paifu_scraping.scraping import (
    clear_not_found_marker,
    has_any_tokenized_views,
    has_not_found_marker,
    not_found_marker_path,
    write_not_found_marker,
)


def test_not_found_marker_round_trip(tmp_path: Path) -> None:
    raw_path = tmp_path / "2011010100gm-00a9-0000-aaaaaaaa.txt"

    assert has_not_found_marker(raw_path) is False
    assert not_found_marker_path(raw_path).name.endswith(".txt.404")

    write_not_found_marker(raw_path)
    assert has_not_found_marker(raw_path) is True

    clear_not_found_marker(raw_path)
    assert has_not_found_marker(raw_path) is False


def test_has_any_tokenized_views_detects_ids_bin_files(tmp_path: Path) -> None:
    assert has_any_tokenized_views(tmp_path) is False

    (tmp_path / "game__complete.ids.bin").write_bytes(b"\x00\x01")

    assert has_any_tokenized_views(tmp_path) is True
