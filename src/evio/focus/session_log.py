"""Session logging: JSONL stream + end-of-session summary table.

Focus audits are provenance events — per-serial peak scores, final
scores, operator marks, iris state and temperatures all need a durable
record (the audit table in docs/focus_audit.md is built from these).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TextIO

from evio.focus.sources import PaneSnapshot


class SessionLog:
    def __init__(self, root: Path, meta: dict[str, object]) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = root / f"focus-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = (self.dir / "session.jsonl").open(
            "w", encoding="utf-8",
        )
        self._write({"type": "meta", **meta})

    def _write(self, record: dict[str, object]) -> None:
        record.setdefault("t", round(time.time(), 3))
        self._file.write(json.dumps(record) + "\n")

    def scores(self, snaps: list[PaneSnapshot]) -> None:
        for snap in snaps:
            if snap.last is None:
                continue
            self._write({
                "type": "score",
                "serial": snap.serial,
                "hf": round(snap.last.hf, 6),
                "ema": round(snap.ema_now, 6),
                "peak_ema": round(snap.peak_ema, 6),
                "roi_rate": round(snap.last.roi_rate, 1),
                "frame_rate": round(snap.last.frame_rate, 1),
                "die_temp_c": snap.die_temp_c,
                "windows": snap.windows,
            })
        self._file.flush()

    def mark(self, serial: str, label: str) -> None:
        self._write({"type": "mark", "serial": serial, "label": label})
        self._file.flush()

    def summary(self, snaps: list[PaneSnapshot]) -> Path:
        lines = [
            "# Focus session summary",
            "",
            f"Ended: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| serial | status | peak EMA | final EMA | % of peak |"
            " windows | die C |",
            "|---|---|---|---|---|---|---|",
        ]
        for snap in snaps:
            pct = (
                100.0 * snap.ema_now / snap.peak_ema
                if snap.peak_ema > 0
                else 0.0
            )
            temp = (
                f"{snap.die_temp_c:.0f}"
                if snap.die_temp_c is not None
                else "-"
            )
            lines.append(
                f"| {snap.serial} | {snap.status} |"
                f" {snap.peak_ema:.4f} | {snap.ema_now:.4f} |"
                f" {pct:.0f}% | {snap.windows} | {temp} |",
            )
        lines += [
            "",
            "Final EMA at ~100% of peak on every camera = rings locked"
            " at best focus.",
            "",
        ]
        path = self.dir / "summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def close(self) -> None:
        self._file.close()
