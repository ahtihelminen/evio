"""Ensures sibling test-support modules (e.g. `_cdp_client`) import cleanly
regardless of pytest's rootdir/import-mode resolution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
