"""The laptop suite has no network: opening a socket is refused.

`conftest._no_sockets` enforces it; a test that really needs a local socket
says so with `pytest.mark.sockets` and is left alone.
"""

from __future__ import annotations

import socket

import pytest
from support.network_guard import SocketBlocked

pytestmark = pytest.mark.t2


@pytest.mark.xfail(raises=SocketBlocked, strict=True)
def test_opening_a_socket_is_refused() -> None:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)


@pytest.mark.sockets
def test_marked_test_may_open_a_socket() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
        pass
