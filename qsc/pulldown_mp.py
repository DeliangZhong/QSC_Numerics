"""Mixed-precision pulldown using mpmath for arbitrary-precision arithmetic.

The pulldown brings Q-functions from large imaginary u down to the real cut
via NI sequential matrix multiplications. At float64, this is limited to
NI ≈ 4 before roundoff destroys the result. With mpmath at 50+ digits,
NI = 30-60 works correctly.
"""

import mpmath
import numpy as np


def pulldown_Q_mp(Q_init: np.ndarray, Puj: np.ndarray,
                  NI: int, dps: int = 50) -> np.ndarray:
    """Perform the Q-function pulldown at arbitrary precision.

    Args:
        Q_init: Q_{a|i}(u_k) at large u, shape (4, 4, lc), complex128
        Puj: P_a at shifted points, shape (4, NI, lc), complex128
        NI: number of pulldown steps
        dps: decimal digits of precision for mpmath

    Returns:
        Q after pulldown, shape (4, 4, lc), complex128

    The recurrence (from C++ lines 1510-1522):
        Q_new[a, i, k] = Q_old[a, i, k]
            + Puj[a, n, k] * Σ_b (-1)^{b+1} * Puj[3-b, n, k] * Q_old[b, i, k]
    iterated for n = NI-1 down to 0.
    """
    mpmath.mp.dps = dps
    lc = Q_init.shape[2]
    m1_signs = [mpmath.mpf(-1), mpmath.mpf(1), mpmath.mpf(-1), mpmath.mpf(1)]

    # Convert Q to mpmath: work column-by-column (each k independently)
    Q_mp = _to_mp_3d(Q_init)  # [a][i][k] -> mpmath mpc
    Puj_mp = _to_mp_3d(Puj)   # [a][n][k] -> mpmath mpc

    # Pulldown: n from NI-1 down to 0
    for n in range(NI - 1, -1, -1):
        for k in range(lc):
            for i in range(4):
                # Save Q_old for this (i, k)
                Q_old = [Q_mp[a][i][k] for a in range(4)]

                # contrib = Σ_b (-1)^{b+1} * Puj[3-b, n, k] * Q_old[b]
                contrib = mpmath.mpc(0)
                for b in range(4):
                    contrib += m1_signs[b] * Puj_mp[3 - b][n][k] * Q_old[b]

                # Q_new[a] = Q_old[a] + Puj[a, n, k] * contrib
                for a in range(4):
                    Q_mp[a][i][k] = Q_old[a] + Puj_mp[a][n][k] * contrib

    return _from_mp_3d(Q_mp, Q_init.shape)


def _to_mp_3d(arr: np.ndarray) -> list:
    """Convert numpy complex128 3D array to nested lists of mpmath mpc."""
    d0, d1, d2 = arr.shape
    result = []
    for i in range(d0):
        layer1 = []
        for j in range(d1):
            layer2 = []
            for k in range(d2):
                z = arr[i, j, k]
                layer2.append(mpmath.mpc(float(z.real), float(z.imag)))
            layer1.append(layer2)
        result.append(layer1)
    return result


def _from_mp_3d(mp_arr: list, shape: tuple) -> np.ndarray:
    """Convert nested lists of mpmath mpc to numpy complex128 3D array."""
    result = np.empty(shape, dtype=np.complex128)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                z = mp_arr[i][j][k]
                result[i, j, k] = complex(z.real, z.imag)
    return result
