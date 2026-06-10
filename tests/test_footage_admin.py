"""List/delete-by-day footage admin used by the web UI.

Builds a small `<root>/<camera>/<day>/` tree on disk and checks the listing
shape/ordering and that deletion frees the right bytes. Deletion is
irreversible, so the traversal-rejection cases (escaping `camera_id`/`day`)
are asserted to raise rather than touch anything outside the root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evio.recording import delete_footage_day, list_footage_days


def _day(root: Path, camera: str, day: str, files: dict[str, int]) -> None:
    day_dir = root / camera / day
    day_dir.mkdir(parents=True, exist_ok=True)
    for name, size in files.items():
        (day_dir / name).write_bytes(b"\0" * size)


def test_list_days_reports_size_and_newest_first(tmp_path: Path) -> None:
    _day(tmp_path, "event-0", "2026-06-04", {"a.dat": 100})
    _day(tmp_path, "event-0", "2026-06-05", {"a.dat": 10, "b.dat": 20})
    _day(tmp_path, "event-1", "2026-06-05", {"a.dat": 5})

    days = list_footage_days({"event": tmp_path})["event"]

    assert [(d["day"], d["camera_id"]) for d in days] == [
        ("2026-06-05", "event-1"),
        ("2026-06-05", "event-0"),
        ("2026-06-04", "event-0"),
    ]
    newest_event0 = next(
        d for d in days
        if d["camera_id"] == "event-0" and d["day"] == "2026-06-05"
    )
    assert newest_event0 == {
        "camera_id": "event-0", "day": "2026-06-05", "bytes": 30, "files": 2,
    }


def test_list_skips_non_day_dirs_and_missing_root(tmp_path: Path) -> None:
    _day(tmp_path, "event-0", "2026-06-05", {"a.dat": 1})
    (tmp_path / "event-0" / "notaday").mkdir()

    listed = list_footage_days({"event": tmp_path, "mono": tmp_path / "nope"})

    assert [d["day"] for d in listed["event"]] == ["2026-06-05"]
    assert listed["mono"] == []


def test_delete_frees_bytes_and_removes_only_that_day(tmp_path: Path) -> None:
    _day(tmp_path, "event-0", "2026-06-04", {"a.dat": 100})
    _day(tmp_path, "event-0", "2026-06-05", {"a.dat": 10, "b.dat": 20})

    freed = delete_footage_day(tmp_path, "event-0", "2026-06-05")

    assert freed == 30
    assert not (tmp_path / "event-0" / "2026-06-05").exists()
    assert (tmp_path / "event-0" / "2026-06-04").exists()


def test_delete_missing_day_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        delete_footage_day(tmp_path, "event-0", "2026-06-05")


@pytest.mark.parametrize(
    ("camera_id", "day"),
    [
        ("event-0", "2026-6-5"),       # not the YYYY-MM-DD shape
        ("event-0", "../2026-06-05"),  # day-side traversal
        ("..", "2026-06-05"),          # camera-side traversal
        ("a/b", "2026-06-05"),         # nested camera id
    ],
)
def test_delete_rejects_traversal(
    tmp_path: Path, camera_id: str, day: str,
) -> None:
    outside = tmp_path.parent / "victim"
    outside.mkdir(exist_ok=True)
    (outside / "keep.txt").write_text("do not delete")

    with pytest.raises(ValueError, match=r"day|camera"):
        delete_footage_day(tmp_path, camera_id, day)

    assert (outside / "keep.txt").exists()
