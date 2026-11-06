"""Find the test doubles in this tree, by what they do rather than by name.

SN-541 shipped because a double could not express the property its test
claimed to check. The rule that stops the next one has to find the doubles
first, and neither obvious predicate finds them on its own:

* by name (`Fake*`, `Stub*`, `Dummy*`, `Mock*`) finds 52 classes;
* by injection (handed to `monkeypatch.setattr/setitem`, assigned into
  `sys.modules`, passed as a `*_factory`/`*_source` argument) finds 24;
* they overlap in only 7.

Neither alone would have caught `_FakeCam`, the SN-541 double, which
reaches the code under test as a plain `self.cam = _FakeCam()`. So the
predicate is the union of all three, including attribute assignment.

Differences from the real device:
- none known: it is not a stand-in for a device, it reads source files.
"""

from __future__ import annotations

import ast
from pathlib import Path

_NAME_PREFIXES = ("Fake", "Stub", "Dummy", "Mock")
_INJECTORS = {"setattr", "setitem", "patch", "object"}
_FACTORY_SUFFIXES = ("_factory", "_source", "_cls", "_class", "_type")


def _names_used(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def doubles_in(path: Path) -> set[str]:
    """Names of classes defined in `path` that stand in for something."""
    tree = ast.parse(path.read_text())
    local = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    found = {
        name for name in local
        if name.lstrip("_").startswith(_NAME_PREFIXES)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Attribute) and callee.attr in _INJECTORS:
                for arg in node.args[1:]:
                    found |= _names_used(arg) & local
            for keyword in node.keywords:
                if keyword.arg and keyword.arg.endswith(_FACTORY_SUFFIXES):
                    found |= _names_used(keyword.value) & local
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and (
                    "sys.modules" in ast.unparse(target)
                )) or isinstance(target, ast.Attribute):
                    found |= _names_used(node.value) & local
    return found


def scan(tests_dir: Path) -> dict[str, set[str]]:
    """`{relative path: {class names}}` for every test module with doubles."""
    return {
        path.name: names
        for path in sorted(tests_dir.glob("test_*.py"))
        if (names := doubles_in(path))
    }
