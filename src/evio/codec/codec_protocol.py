from collections.abc import Iterable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch

CodecInput = NDArray[np.uint16]


class EventCodec(Protocol):
    """Streaming decoder from raw words to EventsBatch objects."""

    def reset(self) -> None:
        """Clear any internal state."""
        ...

    def decode_chunk(
        self,
        words: CodecInput,
        *,
        final: bool = False,
    ) -> Iterable[EventsBatch]:
        """Decode a chunk of words into zero or more EventsBatch objects."""
        ...
