"""Shared test doubles, each with a test pinning the property it models.

A double belongs here, not inline in a test file, once it stands in for an
external boundary. The rule exists because SN-541's `_FakeCam` echoed back
whatever `set_biases()` was handed, so an offset-versus-absolute scale
mismatch was invisible: the round-trip proved arithmetic, not agreement.

`tests/test_no_adhoc_fakes.py` holds the line. Its allowlist is the set of
doubles that predate this package; it may only shrink.
"""
