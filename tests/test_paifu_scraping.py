from __future__ import annotations

from pathlib import Path

from scripts.paifu_scraping.scraping import (
    FetchTimingStats,
    clear_not_found_marker,
    clear_timeout_marker,
    has_any_tokenized_views,
    has_not_found_marker,
    has_timeout_marker,
    not_found_marker_path,
    timeout_marker_path,
    write_not_found_marker,
    write_timeout_marker,
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


def test_timeout_marker_round_trip(tmp_path: Path) -> None:
    raw_path = tmp_path / "2011010100gm-00a9-0000-bbbbbbbb.txt"

    assert has_timeout_marker(raw_path) is False
    assert timeout_marker_path(raw_path).name.endswith(".txt.timeout")

    write_timeout_marker(raw_path, TimeoutError("read timed out"))
    assert has_timeout_marker(raw_path) is True

    clear_timeout_marker(raw_path)
    assert has_timeout_marker(raw_path) is False


def test_fetch_timing_stats_recommend_adaptive_timeout() -> None:
    stats = FetchTimingStats(window_size=5)

    assert stats.recommended_read_timeout_seconds() == 10.0

    for elapsed in (0.8, 0.9, 1.0, 1.1, 1.2):
        stats.record_success(elapsed)

    recommended = stats.recommended_read_timeout_seconds()
    assert 3.0 <= recommended <= 4.0
