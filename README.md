# Target-Feasibility Branch-and-Prune

Reference implementation and computational experiments accompanying:

Preprint: William Boyle, A Target-Feasibility Branch-and-Prune Algorithm for Nonconvex Constrained Optimization (2026). DOI: 10.5281/zenodo.21886107

This repository contains the numerical solver and the 30-problem computational demonstration suite reported in the paper.

## Files

- `maximin_solver.py` — implementation of the target-feasibility branch-and-prune solver.
- `computational_demonstration_suite.py` — the 30 numerical test problems reported in the paper.
- `results/paper_results.csv` — machine-readable results from the full computational demonstration.

## Requirements

- Python 3
- NumPy
- SymPy

Install the required packages with:

    pip install -r requirements.txt

## Reproducing the computational results

Run all 30 problems with:

    python computational_demonstration_suite.py

The complete run writes the results to:

    results/paper_results.csv

Individual problems may also be run. For example:

    python computational_demonstration_suite.py 26

A range of problems can be run with:

    python computational_demonstration_suite.py 21-30

Partial runs do not overwrite the canonical `paper_results.csv`.

## Method

The method introduces the incumbent objective value U as the target constraint

    f0(x) <= U

with an associated nonnegative multiplier. Within each box, the implementation alternates between projected primal descent and dual ascent. Spatial branching is used to explore the nonconvex domain.

Finite-threshold dual divergence is used by the implementation as a numerical pruning signal. As discussed in the accompanying paper, this should not be interpreted as a general infeasibility certificate for arbitrary nonconvex problems.

## Reproducibility

The individual test definitions, domains, solver parameters, and reference objective values used for the computational study are specified directly in `computational_demonstration_suite.py`.

`results/paper_results.csv` records the numerical output corresponding to the computational demonstration reported in the paper.

## Citation

Citation information will be added upon publication of the accompanying paper.

## License

This project is released under the MIT License.
