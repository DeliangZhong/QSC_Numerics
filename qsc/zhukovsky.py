"""Zhukovsky variable, branch-cut utilities, and sigma coefficients."""

import jax.numpy as jnp
from jax.scipy.special import gammaln


def x_of_u_long(u: jnp.ndarray, g: complex) -> jnp.ndarray:
    """Zhukovsky variable with long cut: x = u/2 - (i/2)*sqrt(4 - u^2).

    Here u is already rescaled by 1/g: the C++ code calls X(uA[k]/g).
    For the QSC, the actual Zhukovsky map on physical u is X(u/g).
    """
    return u / 2 - 0.5j * jnp.sqrt(4 - u**2)


def x_of_u_short(u: jnp.ndarray) -> jnp.ndarray:
    """Zhukovsky variable with short cut: x = u*(1 + sqrt(1-4/u^2))/2.

    Used for |u| > 2 (away from the cut).
    """
    return u * (1 + jnp.sqrt(1 - 4 / u**2)) / 2


def cbinomial(z: complex, n: int) -> complex:
    """Complex binomial coefficient C(z, n) = z*(z-1)*...*(z-n+1) / n!

    Works for complex z and integer n >= 0.
    """
    if n < 0:
        return 0.0 + 0j
    if n == 0:
        return 1.0 + 0j
    result = 1.0 + 0j
    for k in range(n):
        result = result * (z - k) / (k + 1)
    return result


def kappa(n: int, s: int) -> float:
    """Auxiliary function for 1/u expansion of integer powers of x(u).

    kappa(n, s) = n/s * C(n + 2s - 1, s - 1) for s > 0, n > 0.
    kappa(n, 0) = 1, kappa(0, s) = 0 for s > 0.
    """
    if s == 0:
        return 1.0
    if n == 0:
        return 0.0
    return float(n) / s * float(cbinomial(n + 2 * s - 1, s - 1).real)


def kappabar(twicen: int, s: int) -> float:
    """Auxiliary for half-integer case: kappabar(2n, s) = kappa(n, s).

    kappabar(twicen, s) where twicen = 2*n (can be odd for half-integer n).
    """
    if s == 0:
        return 1.0
    if twicen == 0:
        return 0.0
    # kappabar(twicen, s) = (twicen/(2*s)) * C(twicen/2 + 2s - 1, s - 1)
    n_half = twicen / 2.0
    return n_half / s * float(cbinomial(n_half + 2 * s - 1, s - 1).real)


def fsigma(twiceMt: int, n: int, r: int, g: complex) -> complex:
    """Sigma coefficient for the 1/u expansion of P_a(u).

    sigma(twiceMt, n, r, g) = sum_{s=0}^{k-r} kappabar(twiceMt, s) * kappa(2r+q0, k-r-s)
                               * (sqrt(g))^{twiceMt + 2n}

    where k = n // 2, q0 = n % 2.
    """
    k = n // 2
    q0 = n % 2
    total = 0.0
    for s in range(k - r + 1):
        total += kappabar(twiceMt, s) * kappa(2 * r + q0, k - r - s)
    total *= jnp.sqrt(g) ** (twiceMt + 2 * n)
    return total


def build_sigma_table(twiceMt: jnp.ndarray, N0: int, NQ: int,
                      g: complex) -> list[jnp.ndarray]:
    """Build sigma tables for all 4 P-functions.

    sigma[a][r, n] = fsigma(twiceMt[a], 2*n, r, g)

    Note: the C++ sigmasubfunc2 uses n -> 2*n (even indices only for TypeI).

    Returns list of 4 arrays, each of shape (N0+1, NQ+1).
    """
    tables = []
    for a in range(4):
        tMt = int(twiceMt[a])
        table = jnp.zeros((N0 + 1, NQ + 1), dtype=complex)
        for r in range(N0 + 1):
            for n in range(NQ + 1):
                table = table.at[r, n].set(fsigma(tMt, 2 * n, r, g))
        tables.append(table)
    return tables
