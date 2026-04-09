"""ML predictor for QSC initial guesses — Delta(g) and full parameters."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


def g_features(g: np.ndarray) -> np.ndarray:
    """Positional encoding of coupling g for ML input.

    Captures both weak-coupling (g², g⁴) and strong-coupling (√g, g^{1/4}) regimes.
    """
    g = np.atleast_1d(g)
    return np.stack([
        g,
        g ** 2,
        g ** 4,
        np.log(g + 1e-6),
        np.sqrt(g),
        g ** 0.25,
    ], axis=-1)


def delta_perturbative(g: np.ndarray) -> np.ndarray:
    """Perturbative Delta(g) = Delta0 + sum gamma_k * g^k."""
    g = np.atleast_1d(g)
    # Konishi perturbative coefficients from sbWeak
    gamma = {2: 12.0, 4: -48.0, 6: 336.0,
             8: -3296.791, 10: 37519.471, 12: -457695.399}
    result = np.full_like(g, 2.0)  # Delta0 = 2
    for k, val in gamma.items():
        result = result + val * g ** k
    return result


class DeltaPredictor:
    """Simple MLP to predict Delta(g) as residual on top of perturbative."""

    def __init__(self, hidden_sizes=(64, 64), seed=42):
        self.hidden_sizes = hidden_sizes
        self.seed = seed
        self.params = None

    def init_params(self, key):
        """Initialize MLP parameters."""
        sizes = [6] + list(self.hidden_sizes) + [1]  # 6 input features → 1 output
        params = []
        for i in range(len(sizes) - 1):
            key, k1, k2 = random.split(key, 3)
            scale = np.sqrt(2.0 / sizes[i])
            W = scale * random.normal(k1, (sizes[i], sizes[i + 1]))
            b = jnp.zeros(sizes[i + 1])
            params.append((W, b))
        self.params = params
        return params

    @staticmethod
    def apply(params, x):
        """Forward pass: x → Delta_residual."""
        for i, (W, b) in enumerate(params):
            x = x @ W + b
            if i < len(params) - 1:
                x = jnp.tanh(x)  # tanh activation for bounded outputs
        return x.squeeze(-1)

    def predict(self, g: np.ndarray) -> np.ndarray:
        """Predict Delta(g) directly from MLP."""
        g = np.atleast_1d(g)
        X_raw = g_features(g)
        X = jnp.array((X_raw - self.X_mean) / self.X_std)
        y_norm = self.apply(self.params, X)
        return y_norm * self.y_std + self.y_mean

    def train(self, g_train: np.ndarray, Delta_train: np.ndarray,
              lr: float = 1e-3, epochs: int = 5000, verbose: bool = True):
        """Train the MLP on (g, Delta) data."""
        key = random.PRNGKey(self.seed)
        self.init_params(key)

        X_raw = g_features(g_train)
        # Predict Delta directly (perturbative diverges at large g)
        y_raw = Delta_train

        # Normalize inputs and outputs for stable training
        self.X_mean = np.mean(X_raw, axis=0)
        self.X_std = np.std(X_raw, axis=0) + 1e-8
        self.y_mean = np.mean(y_raw)
        self.y_std = np.std(y_raw) + 1e-8

        X = jnp.array((X_raw - self.X_mean) / self.X_std)
        y = jnp.array((y_raw - self.y_mean) / self.y_std)

        @jax.jit
        def loss_fn(params):
            pred = self.apply(params, X)
            return jnp.mean((pred - y) ** 2)

        @jax.jit
        def update(params, lr):
            grads = jax.grad(loss_fn)(params)
            return [(W - lr * dW, b - lr * db)
                    for (W, b), (dW, db) in zip(params, grads)]

        for epoch in range(epochs):
            self.params = update(self.params, lr)
            if verbose and (epoch % 2000 == 0 or epoch == epochs - 1):
                loss = float(loss_fn(self.params))
                pred = self.predict(g_train)
                max_err = float(jnp.max(jnp.abs(pred - Delta_train)))
                mean_err = float(jnp.mean(jnp.abs(pred - Delta_train)))
                print(f"  epoch {epoch:5d}: loss={loss:.6e}, max|err|={max_err:.4e}, "
                      f"mean|err|={mean_err:.4e}")

    def save(self, path: str):
        """Save parameters and normalization stats to npz."""
        flat = {}
        for i, (W, b) in enumerate(self.params):
            flat[f"W{i}"] = np.array(W)
            flat[f"b{i}"] = np.array(b)
        flat["hidden_sizes"] = np.array(self.hidden_sizes)
        flat["X_mean"] = self.X_mean
        flat["X_std"] = self.X_std
        flat["y_mean"] = np.array([self.y_mean])
        flat["y_std"] = np.array([self.y_std])
        np.savez(path, **flat)

    def load(self, path: str):
        """Load parameters and normalization stats from npz."""
        data = np.load(path)
        self.hidden_sizes = tuple(data["hidden_sizes"])
        n_layers = len([k for k in data.files if k.startswith("W")])
        self.params = [(jnp.array(data[f"W{i}"]), jnp.array(data[f"b{i}"]))
                       for i in range(n_layers)]
        self.X_mean = data["X_mean"]
        self.X_std = data["X_std"]
        self.y_mean = float(data["y_mean"][0])
        self.y_std = float(data["y_std"][0])


class FullParamPredictor:
    """Predict full QSC parameters (Delta + c-coefficients) from g.

    Uses Delta predictor + perturbative c-coefficients + interpolation
    from nearby converged solutions.
    """

    def __init__(self, delta_predictor: DeltaPredictor):
        self.delta_pred = delta_predictor
        self.solved_g = []
        self.solved_phys = []

    def add_solution(self, g: float, phys: np.ndarray):
        """Add a converged solution for interpolation."""
        self.solved_g.append(g)
        self.solved_phys.append(phys.copy())

    def predict(self, g: float, N0: int = 8) -> np.ndarray:
        """Predict full parameter vector in physical convention."""
        # Delta from ML
        Delta = float(self.delta_pred.predict(np.array([g]))[0])

        if len(self.solved_g) >= 2:
            # Interpolate c-coefficients from nearest solved points
            dists = [abs(gs - g) for gs in self.solved_g]
            n_use = min(4, len(self.solved_g))
            indices = sorted(range(len(self.solved_g)),
                           key=lambda i: dists[i])[:n_use]

            gs = np.array([self.solved_g[i] for i in indices])
            phys_arr = np.array([self.solved_phys[i] for i in indices])

            # Polynomial interpolation for each parameter
            phys = np.zeros(1 + 4 * N0)
            phys[0] = Delta - 2  # anomalous dimension
            deg = min(n_use - 1, 3)
            for j in range(1, 1 + 4 * N0):
                coeffs = np.polyfit(gs, phys_arr[:, j], deg)
                phys[j] = np.polyval(coeffs, g)
        else:
            # Fall back to perturbative c-coefficients
            from qsc.perturbative import perturbative_params
            Mt = np.array([2., 1., 0., -1.])
            params_pert = perturbative_params(g, N0)
            # Convert to physical
            phys = np.zeros(1 + 4 * N0)
            phys[0] = Delta - 2
            for a in range(4):
                s = 1 + a * N0
                block = params_pert[s:s + N0]
                if a in (0, 2):
                    phys[s:s + N0] = np.array(jnp.imag(block)) * g ** Mt[a]
                else:
                    phys[s:s + N0] = np.array(jnp.real(block)) * g ** Mt[a]

        return phys

    def predict_internal(self, g: float, N0: int = 8) -> jnp.ndarray:
        """Predict full parameter vector in internal (C++) convention."""
        Mt = np.array([2., 1., 0., -1.])
        phys = self.predict(g, N0)
        internal = jnp.zeros(1 + 4 * N0, dtype=complex)
        internal = internal.at[0].set(phys[0] + 0j)
        for a in range(4):
            s = 1 + a * N0
            block = phys[s:s + N0] / g ** Mt[a]
            if a in (0, 2):
                internal = internal.at[s:s + N0].set(1j * block)
            else:
                internal = internal.at[s:s + N0].set(block + 0j)
        return internal
