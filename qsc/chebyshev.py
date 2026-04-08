"""Chebyshev grid and transform matrices for the QSC cut [-2g, 2g]."""

import jax.numpy as jnp


def chebyshev_grid(g: float, lc: int) -> jnp.ndarray:
    """Chebyshev-Gauss quadrature points on [-2|g|, 2|g|].

    u_k = -2*Re(g)*cos(pi*(2k+1)/(2*lc)), k = 0,...,lc-1
    """
    k = jnp.arange(lc)
    return -2 * jnp.real(g) * jnp.cos(jnp.pi * (2 * k + 1) / (2 * lc))


def chebyshev_CT(lc: int) -> jnp.ndarray:
    """Chebyshev-T transform matrix.

    CT[n, k] = cos(pi*(2*n+1)*k / (2*lc))
    """
    n = jnp.arange(lc)
    k = jnp.arange(lc)
    return jnp.cos(jnp.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * lc))


def chebyshev_CU(CT: jnp.ndarray, lc: int) -> jnp.ndarray:
    """Chebyshev-U transform matrix derived from CT.

    CU[n, k] = (CT[n, k] - CT[n, k+2]) / 2  for k <= lc-3
    CU[n, k] = CT[n, k] / 2                  for k >= lc-2
    """
    CU = jnp.zeros_like(CT)
    CU = CU.at[:, :lc - 2].set((CT[:, :lc - 2] - CT[:, 2:lc]) / 2)
    CU = CU.at[:, lc - 2:].set(CT[:, lc - 2:] / 2)
    return CU


def sqrt_weight(g: complex, uA: jnp.ndarray) -> jnp.ndarray:
    """sqrt(4g^2 - u_k^2) weight for Fourier inversion."""
    return jnp.sqrt(4 * g * g - uA**2)


def ensure_min_lc(lc: int, Nas: list[list[int]], N0: int) -> int:
    """Ensure lc is large enough for Fourier transformation.

    Matches the C++ logic that adjusts lc based on Nas values.
    """
    min_vals = []
    for a in range(4):
        min_vals.append(3 + Nas[a][0])
        min_vals.append(2 * N0 + 1 - Nas[a][0])
    return max(lc, max(min_vals))
