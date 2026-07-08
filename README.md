# PolyUQ

Polymorphic Uncertainty Quantification for Operational Modal Analysis.

PolyUQ propagates mixed aleatory/epistemic uncertainty — random variables,
intervals, and Dempster-Shafer evidence — through arbitrary mapping
functions, and post-processes the results into imprecision (belief /
plausibility) and incompleteness bounds.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[viz]   # seaborn-based plotting helpers
pip install -e .[data]  # xarray-backed result loading
pip install -e .[hpc]   # ray/simpleflock/psutil for distributed evaluation
pip install -e .[dev]   # test suite
```

## Quick start

```python
from polyuq import RandomVariable, MassFunction, PolyUQ

q1 = RandomVariable(name="q1", dist="norm", params=[15.0, 4.0], primary=True)
q2 = RandomVariable(name="q2", dist="norm", params=[8.0, 2.0], primary=True)

poly = PolyUQ([q1, q2], [], dim_ex="hadamard")
poly.sample_qmc(N_mcs_ale=100, N_mcs_epi=1)
poly.propagate(lambda q1, q2, **kwargs: q1 + q2, {"q1": "q1", "q2": "q2"})
```

See `examples/` for worked case studies and `notebooks/` for interactive
walkthroughs. The companion repository
[oma_uq](https://github.com/simonmarwitz/oma_uq) applies PolyUQ to an
operational modal analysis case study, using data published at
[DOI 10.71758/refodat.46](https://doi.org/10.71758/refodat.46).

## License

GPL v3, see [LICENSE](LICENSE).
