"""Shared test doubles, each with a test pinning the property it models.

A double belongs here, not inline in a test file, once it stands in for an
external boundary. The rule exists because SN-541's `_FakeCam` echoed back
whatever `set_biases()` was handed, so an offset-versus-absolute scale
mismatch was invisible: the round-trip proved arithmetic, not agreement.

`tests/test_no_adhoc_fakes.py` holds the line. Its allowlist is the set of
doubles that predate this package; it may only shrink.

`docs/test-doubles.md` lists every module here and under `tests/support/`
with its conformance test, or `GAP` where it has none.
`tests/test_double_inventory.py` fails when it drifts from the tree. Each
module's docstring has a `Differences from the real device:` section, the
text `none known` if there is nothing to say; add one to a new double.
"""
