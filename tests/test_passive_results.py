"""The numbers the paper reports, pinned against the cached runs they were read from."""
import glob
import json
import pathlib

import numpy as np
import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "passive_info" / "results"


def _passive(band_hz: float) -> dict:
    rows = [json.load(open(f)) for f in sorted(glob.glob(str(RESULTS / "grid_*.json")))]
    rows = [r for r in rows if float(r.get("band_hz", 100.0)) == band_hz]
    if len(rows) < 5:
        pytest.skip(f"only {len(rows)} of 5 passive runs at {band_hz:g} Hz")
    return {r["ks"]: r for r in rows}


def _controlled(band_hz: float = 100.0) -> dict:
    """The controlled runs at ONE band, keyed by stiffness."""
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "ctrl_*.json"))):
        if pathlib.Path(f).stem.endswith(".1"):
            continue                       # the replicate has its own test
        r = json.load(open(f))
        # pre-band-argument runs are all 100 Hz and are backfilled; the default is the
        # last-resort case of a file written before either happened
        if float(r.get("band_hz", 100.0)) != band_hz:
            continue
        rows.append(r)
    if len(rows) < 5:
        pytest.skip(f"only {len(rows)} of 5 controlled runs at {band_hz:g} Hz")
    return {r["ks"]: r for r in rows}


@pytest.mark.parametrize("band_hz", [100.0, 200.0])
def test_the_estimator_offset_is_flat_in_frequency(band_hz):
    """The sampled route is displaced in level, not in shape."""
    for ks, row in _passive(band_hz).items():
        ratio = np.array(row["bands"]) / np.array(row["reference_bands"])
        assert ratio.std() < 0.07, f"{row['label']}: offset varies with frequency"
        assert 0.60 < ratio.mean() < 0.85, f"{row['label']}: offset outside the known range"


@pytest.mark.parametrize("band_hz", [100.0, 200.0])
def test_the_offset_does_not_separate_by_observation_dimension(band_hz):
    """The retracted claim, pinned as retracted."""
    rows = _passive(band_hz)
    ratio = {ks: r["total"] / r["linearised"] for ks, r in rows.items()}
    rigid = ratio.pop(None)
    sea = np.array(list(ratio.values()))
    nearest = float(np.min(np.abs(sea - rigid)))
    spread = float(sea.max() - sea.min())
    assert nearest < 0.01, "a dimensional split would separate the rigid leg from all of them"
    assert spread > 10.0 * nearest, "the within-group spread must swamp the between-group gap"


def test_the_passive_effect_clears_its_floor_only_at_the_higher_band():
    """Why the passive result is reported at 200 Hz."""
    lo, hi = _passive(100.0), _passive(200.0)
    floor_lo = abs(lo[1e6]["total"] - lo[None]["total"])
    floor_hi = abs(hi[1e6]["total"] - hi[None]["total"])
    soft_hi = hi[70.0]["total"] - hi[None]["total"]

    assert floor_lo == pytest.approx(1.74, abs=0.3)
    assert floor_hi == pytest.approx(6.33, abs=0.6)
    assert soft_hi == pytest.approx(-14.98, abs=1.0)
    assert abs(soft_hi) > 2.0 * floor_hi, "the reported effect must clear its own control"


def test_the_controlled_conditional_is_monotone_and_reaches_the_rigid_limit():
    """The headline of the controlled experiment."""
    c = _controlled()
    cmi = {ks: r["joint_total"] - r["action_total"] for ks, r in c.items()}
    seq = [cmi[k] for k in (70.0, 1e3, 1e4, 1e6)]
    assert seq == sorted(seq), "the conditional must rise with stiffness"
    assert abs(cmi[1e6] - cmi[None]) < 1.5, "the stiff limit must reach the rigid leg"
    assert cmi[None] == pytest.approx(29.52, abs=0.5)


def test_the_soft_conditionals_are_zero_within_resolution():
    """The two softest designs are reported as zero, and an identity says why."""
    c = _controlled()
    for ks, row in c.items():
        joint, action = np.array(row["joint"]), np.array(row["action"])
        excess = float(np.maximum(0.0, action - joint).sum())
        cmi = row["joint_total"] - row["action_total"]
        resolvable = excess < abs(cmi)
        assert resolvable == (ks in (None, 1e4, 1e6)), f"{row['label']} changed resolvability"


def test_the_replicate_corroborates_the_scale_of_that_error():
    """Two runs of k_s = 1e3 differing only in GPU reduction order."""
    a = RESULTS / "ctrl_ks1000.json"
    b = RESULTS / "ctrl_ks1000.1.json"
    if not (a.exists() and b.exists()):
        pytest.skip("the replicate is not on disk")
    x, y = json.load(open(a)), json.load(open(b))
    for key in ("joint_total", "action_total"):
        rel = abs(x[key] - y[key]) / x[key]
        assert rel < 0.03, f"{key} should reproduce to a few percent"
    cmi_x = x["joint_total"] - x["action_total"]
    cmi_y = y["joint_total"] - y["action_total"]
    assert abs(cmi_x - cmi_y) > 2.0, "the difference inherits both terms' absolute scatter"
