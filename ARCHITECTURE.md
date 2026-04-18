# Architecture

## Module Map (`qsc/`)

### Core Forward Maps
- `forward_map.py` — JAX float64 residual: (params, g) -> residual via Chebyshev grid, P-contraction, gluing. AD-traceable.
- `forward_map_mp.py` — mpmath arbitrary-precision forward map with FD Jacobian. Breaks the g~0.157 float64 barrier.
- `forward_map_flint.py` — FLINT/Arb C-library drop-in for `forward_map_mp`. 10-50x faster, same interface.

### Solvers
- `newton.py` — Newton with JAX AD Jacobian, backtracking line search, LM fallback.
- `newton_mp.py` — Newton with mpmath FD Jacobian + Broyden rank-1 updates.
- `hybrid_solve.py` — FLINT residual (50-digit) + JAX AD Jacobian (~10-digit Newton step).

### Continuation
- `continuation.py` — Predictor-corrector in coupling g with adaptive step size.
- `arclength.py` — Pseudo-arc-length continuation via tangent prediction + augmented Newton.

### Spectral / Precision
- `chebyshev.py` — Chebyshev-Gauss grid and transform matrices (CT, CU) on [-2g, 2g].
- `zhukovsky.py` — Zhukovsky map, branch-cut utilities, complex binomial coefficients.
- `pulldown_mp.py` — Mixed-precision pulldown: Q-functions from large |u| to the cut via sequential matrix products (mpmath).
- `spectral_q.py` — Direct Q evaluation at probe points via basis conversion (experimental, currently non-functional).

### Initial Guesses & ML
- `perturbative.py` — Weak-coupling Marboe-Volin expansion for initial Konishi guess.
- `ml_predictor.py` — ML predictor with positional encoding of g for bootstrap scanning.

### Utilities
- `quantum_numbers.py` — Oscillator quantum numbers (nb, nf, na), derived quantities (L, Delta_0, Mt).
- `io_utils.py` — Mathematica <-> internal format conversion, denormalization, gauge transforms.

## Scripts (`scripts/`)

- `scan_konishi.py` — JAX float64 continuation scan from g=0.1 upward.
- `scan_konishi_mp.py` — Two-phase scan: JAX (g<0.15) then mpmath+Broyden past the float64 barrier.
- `bootstrap_scan.py` — ML-predicted initial guesses, retrain every 10 points.
- `dense_scan_and_train.py` — Dense JAX scan with 4-pt interpolation + GD warmup fallback.
- `dense_scan_mpmath.py` — Dense scan with tight convergence (||E|| < 1e-5), float64 only.
- `generate_training_data.py` — Perturbative + C++ fixture data for ML training.
- `test_basin.py` — Newton basin-of-attraction diagnostic at g=0.1, 0.2.

## Data (`data/`)

All `.npz` files contain scan results: arrays of `g` values, converged parameters, and Delta(g).

- `konishi_solutions.npz`, `konishi_all_solutions.npz` — JAX float64 scan results.
- `konishi_dense_scan.npz`, `konishi_dense_v2.npz` — Dense scan outputs.
- `konishi_mp_scan.npz`, `konishi_tight_scan.npz` — mpmath high-precision scan results.
- `konishi_from_g02.npz` — Scan starting from g=0.2.
- `delta_predictor.npz`, `delta_predictor_v2.npz` — Trained ML predictor weights.

## Three-Tier Precision Strategy

1. **JAX float64** — fast (~68ms/eval), AD Jacobian, works to g ~ 0.157.
2. **mpmath** — arbitrary precision, FD Jacobian + Broyden, extends to g ~ 0.183. Slow (~30s/eval).
3. **FLINT/Arb** — C-library speed with mpmath precision, 3x over mpmath. Extends to g ~ 0.172 (hybrid Jacobian issues limit further).

## Current Frontier

The solver reliably reaches g ~ 0.183 (Konishi) using mpmath + Broyden continuation. Beyond this, the pulldown step (bringing Q from large |u| to the cut) becomes ill-conditioned. Three alternative approaches (spectral Q-solver) have been attempted and failed — see `discussion_AI.md` Implementation-22.
