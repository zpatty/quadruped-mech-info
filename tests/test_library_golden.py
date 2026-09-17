"""`mechinfo` pinned scalar by scalar, on mechanisms the paper never uses."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

GOLDEN = pathlib.Path(__file__).parent / "golden_library.json"


@pytest.fixture(scope="module")
def tables():
    import golden_library
    return json.loads(GOLDEN.read_text()), golden_library.collect()


def test_the_library_table_is_complete(tables):
    want, got = tables
    assert set(got) - set(want) == set(), "new scalars: regenerate with -m tests.golden_library"
    assert set(want) - set(got) == set(), "scalars vanished: a path stopped being computed"


def test_every_library_scalar_is_unchanged(tables):
    want, got = tables
    bad = []
    for k in sorted(want):
        a, b = want[k], got[k]
        if a != b and abs(a - b) > 1e-9 * max(1.0, abs(a)):
            bad.append(f"  {k}: {a!r} -> {b!r}  ({b - a:+.3e})")
    assert not bad, f"{len(bad)} library scalars moved:\n" + "\n".join(bad[:40])


def test_the_locked_reference_is_what_separates_the_two_designs(tables):
    """The claim `locked_effective_mass` exists to make, asserted rather than only recorded."""
    want, _ = tables
    rigid, sea, ref = (want["plant/rigid_c/m_e"], want["plant/sea_c/m_e"],
                       want["plant/locked_reference"])
    assert sea < rigid, "the SEA's own effective mass must be the lighter, defective one"
    assert ref == rigid, "the locked reference IS the rigid design's effective mass"
    assert ref != sea, "if these ever coincide the whole distinction has been lost"
