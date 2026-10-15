"""Record tap values by step, so a stage can be re-scored without upstream."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from evio.dtc.taps import TapBus


class TapRecorder:
    """Keeps `(step_index, value)` for each named tap, in publish order.

    A stage tap publishes per stream, so a step may appear several times.
    """

    def __init__(self, bus: TapBus, names: Sequence[str]) -> None:
        self.recorded: dict[str, list[tuple[int, object]]] = {
            name: [] for name in names
        }
        self._unsubscribers = [
            bus.subscribe(name, self._keeper(name)) for name in names
        ]

    def _keeper(self, name: str) -> Callable[[int, object], None]:
        values = self.recorded[name]

        def keep(step_index: int, value: object) -> None:
            values.append((step_index, value))

        return keep

    def close(self) -> None:
        """Stop recording; what was recorded stays."""
        for unsubscribe in self._unsubscribers:
            unsubscribe()
