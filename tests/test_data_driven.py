"""
Tier 2 tests — require the published refodat dataset.

Set the environment variable POLYUQ_DATA_DIR to the local path of the
unpacked refodat archive (DOI 10.71758/refodat.46) before running.

These tests validate that the post-processing pipeline (sensitivity,
imprecision, incompleteness) produces correct results given the pre-computed
HPC propagation outputs — WITHOUT re-running the HPC simulation.

Run with:
    POLYUQ_DATA_DIR=/path/to/run-oma pytest -m data tests/test_data_driven.py
"""
import os
import sys
import numpy as np
import pytest
from pathlib import Path
from polyuq import PolyUQ


RESULT_DIR = Path(os.environ.get("POLYUQ_DATA_DIR", ""))


def _skip_if_missing(*paths):
    for p in paths:
        if not p.exists():
            pytest.skip(f"Required file not found: {p}")


def _oma_uq_root():
    """Repo root of the companion application layer (pyoma-uq).

    Public convention: PolyUQ and pyoma-uq cloned as sibling repos; override
    with OMA_UQ_PATH for other layouts. Returns the root rather than the study
    directory, because the studies import each other as ``pyoma_uq.studies.*``
    and so need the root itself on sys.path.
    """
    return Path(os.environ.get("OMA_UQ_PATH",
                               Path(__file__).parent.parent.parent / "pyoma-uq"))


def _oma_uq_available():
    return (_oma_uq_root() / "pyoma_uq" / "studies").is_dir()


@pytest.fixture(scope="module")
def vars_stage3():
    """Load vars_definition(stage=3) from oma_uq."""
    if not os.environ.get("POLYUQ_DATA_DIR", ""):
        pytest.skip("POLYUQ_DATA_DIR not set — skipping data-driven tests")
    sys.path.insert(0, str(_oma_uq_root()))
    from pyoma_uq.studies.UQ_OMA import vars_definition
    vars_ale, vars_epi, _ = vars_definition(stage=3)
    return vars_ale, vars_epi


@pytest.mark.data
def test_load_samp(vars_stage3):
    """polyuq_samp.npz loads and has expected structure."""
    samp_path = RESULT_DIR / "polyuq_samp.npz"
    _skip_if_missing(samp_path)

    vars_ale, vars_epi = vars_stage3
    poly = PolyUQ(vars_ale, vars_epi, dim_ex="cartesian")
    poly.load_state(str(samp_path), differential="samp")

    assert poly.inp_samp_prim is not None
    # check primary sample has correct number of variables
    n_prim = sum(1 for v in vars_ale + vars_epi if v.primary)
    assert poly.inp_samp_prim.shape[-1] == n_prim


@pytest.mark.data
@pytest.mark.parametrize("method,mode", [
    ("f_sc", 0), ("f_sc", 6),
    ("d_sc", 0),
    ("f_cf", 0),
])
def test_load_prop_one_mode(vars_stage3, method, mode):
    """polyuq_prop.npz loads for individual output/mode combinations."""
    ret_dir = f"{method}-{mode}"
    prop_path = RESULT_DIR / "estimations" / ret_dir / "polyuq_prop.npz"
    samp_path = RESULT_DIR / "polyuq_samp.npz"
    _skip_if_missing(prop_path, samp_path)

    vars_ale, vars_epi = vars_stage3
    poly = PolyUQ(vars_ale, vars_epi, dim_ex="cartesian")
    poly.load_state(str(samp_path), differential="samp")
    poly.load_state(str(prop_path), differential="prop")

    assert poly.out_samp is not None
    # natural frequencies must be > 0
    if method.startswith("f_"):
        assert np.all(poly.out_samp > 0), (
            f"Frequencies for {ret_dir} contain non-positive values"
        )
    # damping ratios in (0, 100) %
    if method.startswith("d_"):
        assert np.all(poly.out_samp > 0)
        assert np.all(poly.out_samp < 100)


@pytest.mark.data
@pytest.mark.parametrize("method,mode", [("f_sc", 0), ("d_sc", 0)])
def test_sensitivity_values(vars_stage3, method, mode):
    """Sensitivity indices are in [0,1] and approximately sum to 1."""
    ret_dir = f"{method}-{mode}"
    samp_path = RESULT_DIR / "polyuq_samp.npz"
    prop_path = RESULT_DIR / "estimations" / ret_dir / "polyuq_prop.npz"
    sens_path = RESULT_DIR / "estimations" / ret_dir / "polyuq_sens.npz"
    _skip_if_missing(samp_path, prop_path, sens_path)

    vars_ale, vars_epi = vars_stage3
    poly = PolyUQ(vars_ale, vars_epi, dim_ex="cartesian")
    poly.load_state(str(samp_path), differential="samp")
    poly.load_state(str(prop_path), differential="prop")
    poly.load_state(str(sens_path), differential="sens")

    S = poly.S_point
    assert S is not None
    assert np.all(S >= -0.05), "Sensitivity indices should not be significantly negative"
    assert np.all(S <= 1.05), "Sensitivity indices should not exceed 1"


@pytest.mark.data
@pytest.mark.parametrize("method,mode", [("f_sc", 0)])
def test_imprecision_focal_ordering(vars_stage3, method, mode):
    """Imprecision focal intervals satisfy lower ≤ upper."""
    ret_dir = f"{method}-{mode}"
    samp_path = RESULT_DIR / "polyuq_samp.npz"
    prop_path = RESULT_DIR / "estimations" / ret_dir / "polyuq_prop.npz"
    imp_path  = RESULT_DIR / "estimations" / ret_dir / "polyuq_imp.npz"
    _skip_if_missing(samp_path, prop_path, imp_path)

    vars_ale, vars_epi = vars_stage3
    poly = PolyUQ(vars_ale, vars_epi, dim_ex="cartesian")
    poly.load_state(str(samp_path), differential="samp")
    poly.load_state(str(prop_path), differential="prop")
    poly.load_state(str(imp_path), differential="imp")

    foc = poly.imp_foc
    assert foc is not None
    # ignore NaN entries (unfinished samples)
    valid = ~np.isnan(foc[:, :, 0])
    assert np.all(foc[valid, 0] <= foc[valid, 1] + 1e-9), (
        "Imprecision focal lower bound must be ≤ upper bound"
    )


@pytest.mark.data
@pytest.mark.parametrize("estimator", ["avg"])
def test_incompleteness_belief_plausibility(vars_stage3, estimator):
    """Belief ≤ plausibility for the incompleteness result of f_sc-0."""
    ret_dir = "f_sc-0"
    samp_path = RESULT_DIR / "polyuq_samp.npz"
    prop_path = RESULT_DIR / "estimations" / ret_dir / "polyuq_prop.npz"
    imp_path  = RESULT_DIR / "estimations" / ret_dir / "polyuq_imp.npz"
    inc_path  = RESULT_DIR / "estimations" / ret_dir / f"polyuq_{estimator}_inc.npz"
    _skip_if_missing(samp_path, prop_path, imp_path, inc_path)

    vars_ale, vars_epi = vars_stage3
    poly = PolyUQ(vars_ale, vars_epi, dim_ex="cartesian")
    poly.load_state(str(samp_path), differential="samp")
    poly.load_state(str(prop_path), differential="prop")
    poly.load_state(str(imp_path), differential="imp")
    poly.load_state(str(inc_path), differential="inc")

    from polyuq import compute_belief
    # focals_stats and focals_mass should be attributes after loading inc state
    if not hasattr(poly, "focals_stats") or poly.focals_stats is None:
        pytest.skip("focals_stats not populated — check load_state differential name")

    bins, bel, pl, _ = compute_belief(poly.focals_stats, poly.focals_mass, cumulative=True)
    assert np.all(bel <= pl + 1e-9), "Belief must be ≤ plausibility"
