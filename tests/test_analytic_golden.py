"""The analytic arm pinned scalar by scalar, so a refactor is provable rather than trusted."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

GOLDEN = pathlib.Path(__file__).parent / "golden_analytic.json"

pytest.importorskip("mujoco")          # the Go2 rows need Menagerie's mass matrix


@pytest.fixture(scope="module")
def tables():
    import golden
    return json.loads(GOLDEN.read_text()), golden.collect()


def test_the_golden_table_is_complete(tables):
    """Keys appearing or disappearing is itself a change worth failing on."""
    want, got = tables
    assert set(got) - set(want) == set(), "new scalars: regenerate the golden file"
    assert set(want) - set(got) == set(), "scalars vanished: a path stopped being computed"


def test_every_analytic_scalar_is_unchanged(tables):
    want, got = tables
    bad = []
    for k in sorted(want):
        a, b = want[k], got[k]
        if a != b and abs(a - b) > 1e-9 * max(1.0, abs(a)):
            bad.append(f"  {k}: {a!r} -> {b!r}  ({b - a:+.3e})")
    assert not bad, f"{len(bad)} analytic scalars moved:\n" + "\n".join(bad[:40])
