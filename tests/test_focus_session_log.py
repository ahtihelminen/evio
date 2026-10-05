"""The focus session's JSONL stream and summary table (SN-607)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evio.focus.metric import FocusResult
from evio.focus.session_log import SessionLog
from evio.focus.sources import PaneSnapshot

pytestmark = pytest.mark.t1


def _snap(serial: str, hf: float) -> PaneSnapshot:
    return PaneSnapshot(
        serial=serial,
        status="synthetic",
        view=None,
        roi=(448, 168, 384, 384),
        raw=np.zeros(4, dtype=np.float32),
        ema=np.zeros(4, dtype=np.float32),
        last=FocusResult(hf=hf, roi_rate=25_000.0, frame_rate=40_000.0),
        ema_now=hf,
        peak_ema=0.2,
        die_temp_c=41.0,
        windows=12,
    )


def test_session_log_writes_jsonl_in_order(tmp_path: Path) -> None:
    log = SessionLog(tmp_path, {"operator": "test"})
    log.scores([_snap("cam-a", 0.11)])
    log.mark("cam-a", "ring locked")
    log.scores([_snap("cam-b", 0.12)])
    log.mark("cam-b", "iris open")
    log.close()

    lines = (log.dir / "session.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [(r["type"], r.get("serial"), r.get("label")) for r in records] == [
        ("meta", None, None),
        ("score", "cam-a", None),
        ("mark", "cam-a", "ring locked"),
        ("score", "cam-b", None),
        ("mark", "cam-b", "iris open"),
    ]
    assert records[0]["operator"] == "test"
    assert records[1]["hf"] == 0.11


def test_session_log_summary_lists_every_serial(tmp_path: Path) -> None:
    log = SessionLog(tmp_path, {})
    snaps = [_snap("cam-a", 0.11), _snap("cam-b", 0.2), _snap("cam-c", 0.05)]

    summary = log.summary(snaps).read_text()
    log.close()

    for serial in ("cam-a", "cam-b", "cam-c"):
        assert f"| {serial} |" in summary
