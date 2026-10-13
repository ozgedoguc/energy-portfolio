# ODD energy portfolio experiment

ODD regenerated synthetic computational illustration

This is a fresh synthetic scenario generated for the manuscript revision.
It is not recovered original data, a reconstruction of prior results, or real
enterprise observations. The generator does not target previous table totals.

Reproduce with Python 3 and NumPy (the verified run used Python 3.12.14 and
NumPy 2.3.5):
    python synthetic_experiment.py

Outputs are written to synthetic_reproduction/. Generation seed: 20261013;
random-weight seed: 20261014. Inputs and all full-precision computed results
are included. The code enumerates all 32,768 binary portfolios, verifies
static Pareto efficiency, and computes the independent shocks, weight
searches and institutional constraint revision. There is no agent execution.

All units and generator ranges are documented in synthetic_projects.json.
Satisfaction, tie-breaking, regret and penalties are documented in code and
computed_results.json. Delays are hypothetical assumptions. Score units are
unvalidated; projects are additive and independent. Results are illustrative
and should not be interpreted as organizational or empirical validation.

This package replaces the numerical illustration accompanying the earlier
manuscript. In this run the initial weighted and max-min portfolios coincide;
the institutionally constrained portfolios differ. No seed search was used
to preserve the previous conclusions.

Repository: https://github.com/ozgedoguc/energy-portfolio
