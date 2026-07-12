"""
Tier 1 tests — self-contained, no external data, always run in CI.

All tests use only numpy/scipy/matplotlib (core deps).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from polyuq import (
    RandomVariable, MassFunction, PolyUQ,
    compute_belief, generate_histogram_bins,
)


def _oma_uq_examples_dir():
    # Public convention: PolyUQ and oma_uq cloned as sibling repos.
    # Override with OMA_UQ_PATH for other layouts.
    root = Path(os.environ.get("OMA_UQ_PATH", Path(__file__).parent.parent.parent / "oma_uq"))
    return root / "examples"


class TestRandomVariable:
    def test_normal_construction(self):
        q = RandomVariable(name="q1", dist="norm", params=[15.0, 4.0], primary=True)
        assert q.name == "q1"

    def test_normal_sample_shape(self):
        q = RandomVariable(name="q1", dist="norm", params=[15.0, 4.0], primary=True)
        s = q.rvs(size=50)
        assert s.shape == (50,)

    def test_normal_support_finite(self):
        q = RandomVariable(name="q1", dist="norm", params=[15.0, 4.0], primary=True)
        lo, hi = q.support()
        assert np.isfinite(lo) or np.isfinite(hi)   # at least one finite

    def test_weibull_min_positive(self):
        # used for wind speed in vars_definition
        v_b = RandomVariable("weibull_min", "v_b", [2.267, 5.618], primary=True)
        s = v_b.rvs(size=100)
        assert np.all(s > 0)

    def test_bernoulli_binary(self):
        # ice_occ in UQ_Modal_Analytical
        ice_days_param = 28.2 / 365
        ice_occ = RandomVariable("bernoulli", "ice_occ",
                                 [ice_days_param], primary=True)
        s = ice_occ.rvs(size=500)
        assert set(np.unique(s)).issubset({0, 1})

    def test_lognorm_construction(self):
        E = RandomVariable("lognorm", "E", [0.074, 1.093e11, 9.58e10], primary=False)
        s = E.rvs(size=20)
        assert np.all(s > 0)


class TestMassFunction:
    def test_point_interval_construction(self):
        mf = MassFunction("x", [(5.0,), (10.0,)], [0.6, 0.4], primary=True)
        assert abs(sum([0.6, 0.4]) - 1.0) < 1e-12

    def test_interval_support(self):
        mf = MassFunction("x", [(1.0, 3.0), (2.0, 5.0)], [0.5, 0.5], primary=True)
        lo, hi = mf.support()
        assert lo <= 1.0
        assert hi >= 5.0

    def test_nested_random_variable(self):
        # replicate the analytical example: MassFunction containing RV as focal
        E_norm = RandomVariable("norm", "En", [2.0576e11, 6.9e9], primary=False)
        E = MassFunction("E", [(2.1e11,), (E_norm,)], [0.8, 0.2], primary=True)
        lo, hi = E.support()
        assert lo > 0

    def test_masses_stored_correctly(self):
        mf = MassFunction("y", [(0.0, 1.0), (2.0, 3.0)], [0.3, 0.7], primary=True)
        stored = np.array(mf.masses)
        np.testing.assert_allclose(stored.sum(), 1.0)

    def test_n_locations_discrete(self):
        # as in vars_definition
        n_loc = MassFunction("n_locations", [(4,), (8,), (12,)],
                             [0.2, 0.5, 0.3], primary=True)
        lo, hi = n_loc.support()
        # integer-dtype MassFunctions pad support by +/-0.5 so that
        # uniform sampling + rounding recovers the discrete values
        assert 4 - 0.5 - 1e-9 <= lo
        assert hi <= 12 + 0.5 + 1e-9


class TestPolyUQ:
    """Tests using the example problems from polymorphic_uncertainty.py."""

    def _example_a_vars(self):
        q1 = RandomVariable(name="q1", dist="norm", params=[15.0, 4.0], primary=True)
        q2 = RandomVariable(name="q2", dist="norm", params=[8.0, 2.0], primary=True)
        return [q1, q2], []

    def _example_b_vars(self):
        inc_q1a1 = (RandomVariable("norm", "q1", [15.0, 4.0], primary=False),)
        inc_q1a0 = (
            inc_q1a1[0],
            RandomVariable("norm", "q1a0l", [-0.4, 0.1], primary=False),
            RandomVariable("norm", "dq1a0r", [0.5, 0.06], primary=False),
        )
        q1 = MassFunction("q1", [inc_q1a1, inc_q1a0], [0.5, 0.5],
                          primary=True, incremental=True)
        q2_base = RandomVariable("norm", "q2", [8.0, 2.0], primary=False)
        q2 = MassFunction("q2", [(q2_base,)], [1.0], primary=True)
        return [q1, q2], []

    def test_sample_qmc_shape_aleatory_only(self):
        vars_ale, vars_epi = self._example_a_vars()
        poly = PolyUQ(vars_ale, vars_epi, dim_ex="hadamard")
        poly.sample_qmc(N_mcs_ale=20, N_mcs_epi=1, check_discr=False)
        prim_ale = [v for v in vars_ale if v.primary]
        assert poly.inp_samp_prim.shape[0] == 20
        assert poly.inp_samp_prim.shape[-1] == len(prim_ale)

    def test_sample_qmc_shape_mixed(self):
        vars_ale, vars_epi = self._example_b_vars()
        poly = PolyUQ(vars_ale, vars_epi, dim_ex="cartesian")
        poly.sample_qmc(N_mcs_ale=8, N_mcs_epi=4, check_discr=False)
        # outer loop is epi, inner is ale
        assert poly.inp_samp_prim.shape[0] >= 8

    def test_propagate_with_scalar_mapping(self):
        """Propagate through a trivial identity function and check output shape."""
        vars_ale, vars_epi = self._example_a_vars()
        poly = PolyUQ(vars_ale, vars_epi, dim_ex="hadamard")
        poly.sample_qmc(N_mcs_ale=10, N_mcs_epi=1, check_discr=False)

        def mapping(q1, q2, jid=None, result_dir=None, working_dir=None):
            return float(q1 + q2)

        poly.propagate(mapping, {"q1": "q1", "q2": "q2"})
        assert poly.out_samp is not None
        assert poly.out_samp.shape[0] == 10

    def test_sample_within_bounds(self):
        q = RandomVariable("uniform", "q", [0.0, 1.0], primary=True)
        poly = PolyUQ([q], [], dim_ex="hadamard")
        poly.sample_qmc(N_mcs_ale=50, N_mcs_epi=1, check_discr=False)
        vals = poly.inp_samp_prim.values.ravel()
        assert np.all(vals >= 0.0 - 1e-9)
        assert np.all(vals <= 1.0 + 1e-9)


class TestComputeBelief:
    def test_belief_le_plausibility(self):
        rng = np.random.default_rng(42)
        n = 8
        masses = rng.dirichlet(np.ones(n))
        focals = np.sort(rng.random((n, 2)), axis=1)
        focals[:, 1] += 0.05   # ensure non-degenerate width

        bins, bel, pl, q = compute_belief(focals, masses, cumulative=False)
        assert np.all(bel <= pl + 1e-9), "belief must be ≤ plausibility everywhere"

    def test_belief_monotone_cumulative(self):
        masses = np.array([0.3, 0.4, 0.3])
        focals = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])
        bins, bel, pl, q = compute_belief(focals, masses, cumulative=True)
        # cumulative belief should be non-decreasing
        assert np.all(np.diff(bel) >= -1e-9)

    def test_total_mass_consistency(self):
        masses = np.array([0.25, 0.75])
        focals = np.array([[0.0, 2.0], [1.0, 3.0]])
        bins, bel, pl, _ = compute_belief(focals, masses, cumulative=False)
        # plausibility should not exceed 1
        assert np.all(pl <= 1.0 + 1e-9)


class TestEvidentialBeam:
    """Integration test from example_evidential_beam.py."""

    def test_beam_deflection_bounds(self):
        """Evidential beam: imprecise interval must be wider than point estimate."""
        from examples.example_evidential_beam import (
            get_deflection_curve, get_deflection_point
        )
        npoints = 25
        F = 1000.0
        l = 10000.0
        E_nom = 11000.0
        I = 80e6

        # Nominal point result
        w_nom = get_deflection_curve(npoints, 0.25, F, l, E_nom, I)
        w_max_nom = np.max(w_nom)

        # With E uncertainty: compute at E boundaries
        w_lo = get_deflection_curve(npoints, 0.25, F, l, E_nom * 1.1, I)
        w_hi = get_deflection_curve(npoints, 0.25, F, l, E_nom * 0.9, I)
        w_max_lo = np.max(w_lo)
        w_max_hi = np.max(w_hi)

        assert w_max_hi > w_max_nom > w_max_lo, (
            "Deflection should increase as E decreases"
        )


class TestAnalyticalMapping:
    """Test the UQ_Modal_Analytical mapping function in isolation (no HPC)."""

    def test_frequencies_positive(self):
        sys.path.insert(0, str(_oma_uq_examples_dir()))
        from UQ_Modal_Analytical import mapping_function

        fd, zetas, frf = mapping_function(
            E=2.1e11, A=0.0343, rho=7850.0, L=200.0,
            omega_u=440.0, zeta=0.047,
            add_mass=60.0, ice_occ=0, ice_mass=0.0
        )
        assert np.all(fd > 0), "Natural frequencies must be positive"
        assert np.all(zetas > 0), "Damping ratios must be positive"
        assert np.all(zetas < 1), "Damping ratios must be < 1 (underdamped)"

    def test_ice_mass_lowers_frequencies(self):
        sys.path.insert(0, str(_oma_uq_examples_dir()))
        from UQ_Modal_Analytical import mapping_function

        fd_no_ice, _, _ = mapping_function(
            E=2.1e11, A=0.0343, rho=7850.0, L=200.0,
            omega_u=440.0, zeta=0.047,
            add_mass=60.0, ice_occ=0, ice_mass=75.0
        )
        fd_ice, _, _ = mapping_function(
            E=2.1e11, A=0.0343, rho=7850.0, L=200.0,
            omega_u=440.0, zeta=0.047,
            add_mass=60.0, ice_occ=1, ice_mass=75.0
        )
        # Ice increases mass → frequencies decrease
        assert np.all(fd_ice <= fd_no_ice + 1e-6)


class TestDataManagerImportOrder:
    """Guards the static-TLS import-order constraint (see oma_uq history ee3f8db):
    ray must never be imported before numpy/scipy/matplotlib in polyuq.data_manager.
    Run in a subprocess so the assertion reflects a genuinely fresh interpreter.
    """

    def test_import_in_clean_interpreter(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", "import polyuq.data_manager"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"import polyuq.data_manager failed in a clean interpreter:\n{result.stderr}"
        )


class TestFromPropagatedSamples:
    """from_propagated_samples + estimate_imp on a closed-form toy problem.

    y = x1 + 2 * x2 with Imprecision focals
    x1: (0, 1) mass 0.6 | (0.5, 2) mass 0.4 ; x2: (1, 3) mass 1.0
    gives the exact output intervals
    H0 = [2, 7] (mass 0.6), H1 = [2.5, 8] (mass 0.4).
    """

    EXPECTED = np.array([[[2.0, 7.0], [2.5, 8.0]]])

    @staticmethod
    def _toy_polyuq(N=200, nan_frac=0.0, seed=0):
        x1 = MassFunction("x1", [(0.0, 1.0), (0.5, 2.0)], [0.6, 0.4],
                          primary=True)
        x2 = MassFunction("x2", [(1.0, 3.0)], [1.0], primary=True)
        rng = np.random.default_rng(seed)
        s1 = rng.uniform(0.0, 2.0, N)
        s2 = rng.uniform(1.0, 3.0, N)
        y = s1 + 2.0 * s2
        if nan_frac:
            y[rng.random(N) < nan_frac] = np.nan
        return PolyUQ.from_propagated_samples(
            [x1, x2], {"x1": s1, "x2": s2}, y[None, :], out_name="y")

    def test_bookkeeping(self):
        pq = self._toy_polyuq()
        assert pq.N_mcs_ale == 1
        assert pq.N_mcs_epi == 200
        assert not pq.loop_ale
        assert pq.loop_epi
        assert np.allclose(pq.var_supp["x1"], [0.0, 2.0])
        assert np.allclose(pq.var_supp["x2"], [1.0, 3.0])
        assert len(pq.imp_hyc_foc_inds) == 2
        assert np.allclose(pq.imp_hyc_mass, [0.6, 0.4])
        assert pq.out_samp.shape == (1, 200)
        assert pq.out_samp.dtype == np.float64

    def test_estimate_imp_closed_form(self):
        pq = self._toy_polyuq()
        imp_foc, _, _, _, _ = pq.estimate_imp(interp_fun="rbf",
                                              opt_meth="genetic")
        assert imp_foc.shape == (1, 2, 2)
        # RBF over 200 samples of a linear function is near-exact, but the
        # genetic optimizer (polish=False) stops within ~5 % of the output
        # range of the exact corner optima
        assert np.allclose(imp_foc, self.EXPECTED, atol=0.3)
        # the hypercube-specific bounds must be distinguishable:
        # H1 is shifted right against H0 by (0.5, 1.0)
        assert imp_foc[0, 1, 0] > imp_foc[0, 0, 0]
        assert imp_foc[0, 1, 1] > imp_foc[0, 0, 1]

    def test_nan_outputs_tolerated(self):
        pq = self._toy_polyuq(nan_frac=0.15, seed=1)
        assert np.isnan(pq.out_samp).sum() > 0
        imp_foc, _, _, _, _ = pq.estimate_imp(interp_fun="rbf",
                                              opt_meth="genetic")
        assert np.allclose(imp_foc, self.EXPECTED, atol=0.35)

    def test_with_primary_aleatory_variable(self):
        x1 = MassFunction("x1", [(0.0, 1.0), (0.5, 2.0)], [0.6, 0.4],
                          primary=True)
        x2 = MassFunction("x2", [(1.0, 3.0)], [1.0], primary=True)
        a = RandomVariable("uniform", "a", [0.0, 1.0], primary=True)
        rng = np.random.default_rng(2)
        N = 200
        s1 = rng.uniform(0.0, 2.0, N)
        s2 = rng.uniform(1.0, 3.0, N)
        a_samp = rng.uniform(0.0, 1.0, N)
        # output shifts by the aleatory value of each row
        out = np.vstack([s1 + 2.0 * s2 + a_samp[n] for n in range(2)])
        pq = PolyUQ.from_propagated_samples(
            [x1, x2], {"x1": s1, "x2": s2, "a": a_samp}, out,
            vars_ale=[a], out_name="y")
        assert pq.N_mcs_ale == 2
        assert pq.loop_ale
        imp_foc, _, _, _, _ = pq.estimate_imp(interp_fun="rbf",
                                              opt_meth="genetic")
        assert imp_foc.shape == (2, 2, 2)
        for n in range(2):
            assert np.allclose(imp_foc[n], self.EXPECTED[0] + a_samp[n],
                               atol=0.35)

    def test_missing_column_raises(self):
        x1 = MassFunction("x1", [(0.0, 1.0)], [1.0], primary=True)
        x2 = MassFunction("x2", [(1.0, 3.0)], [1.0], primary=True)
        with pytest.raises(ValueError, match="missing columns"):
            PolyUQ.from_propagated_samples(
                [x1, x2], {"x1": np.zeros(10)}, np.zeros((1, 10)))

    def test_out_rows_without_aleatory_raises(self):
        x1 = MassFunction("x1", [(0.0, 1.0)], [1.0], primary=True)
        with pytest.raises(ValueError, match="no primary aleatory"):
            PolyUQ.from_propagated_samples(
                [x1], {"x1": np.zeros(10)}, np.zeros((3, 10)))

    def test_epi_sample_count_mismatch_raises(self):
        x1 = MassFunction("x1", [(0.0, 1.0)], [1.0], primary=True)
        with pytest.raises(ValueError, match="epistemic samples"):
            PolyUQ.from_propagated_samples(
                [x1], {"x1": np.zeros(10)}, np.zeros((1, 8)))

    def test_secondary_epistemic_requires_suppl(self):
        x1 = MassFunction("x1", [(0.0, 1.0)], [1.0], primary=True)
        c = MassFunction("c", [(2.0, 3.0)], [1.0], primary=False)
        with pytest.raises(ValueError, match="inp_suppl_epi"):
            PolyUQ.from_propagated_samples(
                [x1, c], {"x1": np.zeros(10)}, np.zeros((1, 10)))
