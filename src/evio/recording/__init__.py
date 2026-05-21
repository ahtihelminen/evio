"""Always-on footage capture for training data.

Every flight's raw event stream is teed into hour-bucketed `.dat` files
keyed by camera serial and day, with a JSONL manifest. The intent is
that *every* flight grows the training corpus passively, even on
batches the detector ignored. The transformer (or any future head) can
then mine the on-disk footage at will.

Why a separate module rather than reusing `evio.media`: the media
pipeline is a per-detection clip ring — bounded to the moments the
tracker considered interesting. Footage is broader; the recorder
captures everything the camera produced regardless of whether
detection fired. Same physical bytes, different lifecycle.

Retention + remote pull live in follow-ups; this first commit lands the
write side.
"""

from evio.recording.footage_recorder import (
    DAT_FILE_HEADER,
    FootageRecorder,
)
from evio.recording.manifest import (
    ManifestEntry,
    iter_manifest,
)

__all__ = [
    "DAT_FILE_HEADER",
    "FootageRecorder",
    "ManifestEntry",
    "iter_manifest",
]
