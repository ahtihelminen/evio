"""ids_peak must stay confined to its backend module.

The soak run measures the U3V driver against ids_peak as a baseline, not
as evio's own dependency. The `ids_peak` SDK is imported at module scope
only in `evio.io.mono_backends.ids_peak` (selected lazily at runtime by
`evio.io.mono_camera` via `EVIO_MONO_BACKEND`), so `src/` must stay
importable with no IDS Peak SDK installed. Mirrors
`tests/test_repo_boundary.py`. A mention of the backend's *name* (e.g. the
`EVIO_MONO_BACKEND` default) is not an import and is not flagged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_FILE = "src/evio/io/mono_backends/ids_peak.py"
_IMPORT_RE = re.compile(r"^\s*(import ids_peak\b|from ids_peak(_\w+)?\b)")


def test_ids_peak_imported_nowhere_under_src() -> None:
    offenders = [
        f"{path.relative_to(_ROOT)}:{n}"
        for path in (_ROOT / "src").rglob("*.py")
        if str(path.relative_to(_ROOT)) != _BACKEND_FILE
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if _IMPORT_RE.match(line)
    ]
    assert not offenders, f"ids_peak imported outside its backend: {offenders}"
