"""The library must not import the paper."""
from __future__ import annotations

import ast
import pathlib

import pytest

LIB = pathlib.Path(__file__).resolve().parent.parent / "mechinfo"
MODULES = sorted(p for p in LIB.glob("*.py"))


def _imported_names(path: pathlib.Path):
    """Every module named by an import anywhere in the file, including inside functions."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_the_library_has_modules_to_check():
    """Guards against this file quietly passing because the package moved or emptied."""
    assert {p.stem for p in MODULES} >= {"mechanics", "estimators", "critic", "gpu_estimator"}


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_no_library_module_imports_the_paper_package(path):
    offenders = [n for n in _imported_names(path) if n.split(".")[0] == "passive_info"]
    assert not offenders, (
        f"{path.name} imports {offenders} -- the library may not depend on the paper. Move "
        "what it needs down into the library, or pass it in as a parameter.")


def test_mechanics_still_depends_on_nothing_in_either_package():
    """It is the root of the graph: the one file both halves may rely on absolutely."""
    local = [n for n in _imported_names(LIB / "mechanics.py")
             if n.split(".")[0] in {"passive_info", "mechinfo"}]
    assert not local, f"mechanics.py grew a local dependency: {local}"


def test_the_adapter_names_the_robot_in_data_rather_than_in_code():
    """`mechinfo.mjcf` must contain no Go2 string. The whole point of `RobotSpec`."""
    src = (LIB / "mjcf.py").read_text()
    for name in ('"FL"', '"FR"', '"RL"', '"RR"', "'FL'", "'FR'"):
        assert name not in src, f"{name} is a Go2 foot name; it belongs in a RobotSpec"
    # `unitree_go2` is allowed ONLY as a default argument, never as a built-in path.
    assert "unitree_go2" in src, "the Menagerie default is expected; if it went, say why"
    assert src.count("unitree_go2") <= 3, "more Go2 mentions than the documented defaults"
