# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reimplementation and acceleration of the Quantum Spectral Curve (QSC) numerical solver for planar N=4 Super-Yang-Mills theory. The existing C++/Mathematica solver lives in `reference/qsc/` as a read-only baseline. New optimized code targets JAX/Python with 50-100x speedup.

Full specification: @QSC_SOLVER_IMPLEMENTATION_GUIDE.md

## Architecture

- `reference/qsc/` — **read-only** reference implementation (C++ cores + Mathematica/Jupyter orchestration). Do not modify.
- New code follows the structure in Section 4 of the implementation guide (`qsc/` Python package).
- Reference C++ uses the CLN library for arbitrary-precision arithmetic. New code uses JAX (`float64`) with optional `mpmath` for high-precision refinement.

## Key Constraints

- **Validate against reference data**: every new module must be cross-checked against the existing C++/Mathematica outputs before being considered correct. Reference numerical data is in `reference/qsc/local operators N4 SYM/data/numerical/` (.mx Mathematica files).
- **JAX-traceable forward map**: the core computation `(coefficients, Δ) → residual` must be written in pure functional style (no in-place mutation) for automatic differentiation to work.
- **Branch cut handling**: the Zhukovsky map has a cut on `[-2g, 2g]`. Use `jnp.where` for sheet selection, never rely on default `jnp.sqrt` branch.

## Build & Run

```bash
# Python (new solver)
pip install -e ".[dev]"
pytest                        # run all tests
pytest -k test_name           # run single test
ruff check .                  # lint
ruff format .                 # format

# Reference C++ (for validation only)
# Requires: g++, libcln-dev
g++ reference/qsc/local\ operators\ N4\ SYM/core/TypeI_core.cpp -lm -lcln -o TypeI_exec.out
```

## Operator Types

The reference code has four type classifications based on symmetry:
- **Type I**: Left-right + parity symmetric (simplest, start here — e.g., Konishi operator)
- **Type II**: Left-right symmetric, general parity
- **Type III**: General, parity symmetric
- **Type IV**: General (most complex)

## Validation Checkpoints

- Konishi Δ(g=1) ≈ 4.189
- Konishi at weak coupling must match 8-loop perturbation theory
- AD Jacobian vs finite-difference Jacobian: agree to ~10⁻⁷ (float64)
- All 219 states with Δ₀ ≤ 6 have reference data available

## State Identification

States are identified by: `_Δ₀[nb1 nb2 nf1 nf2 nf3 nf4 na1 na2]_sol`
Example: Konishi = `_2[0 0 1 1 1 1 0 0]_1`
