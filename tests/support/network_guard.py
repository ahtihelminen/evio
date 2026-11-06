"""Refusal of socket creation, for the suite's no-network guard.

Differences from the real device:
    none known: it replaces `socket.socket.__init__` only, so `_socket`
    called directly, C extensions and subprocesses are not blocked.
"""

from __future__ import annotations


class SocketBlocked(AssertionError):
    """A test opened a socket without declaring `pytest.mark.sockets`."""


def refuse(*_args: object, **_kwargs: object) -> None:
    message = (
        "a test opened a socket; the laptop suite has no network. Mark it "
        "`pytest.mark.sockets` (t2 only) if it needs a local one"
    )
    raise SocketBlocked(message)
