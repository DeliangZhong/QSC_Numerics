"""Full forward map in mpmath: arbitrary precision, no JAX dependency.

Mirrors forward_map.py but uses mpmath for ALL operations, enabling
QaiShift >> 4 without float64 precision loss. The Jacobian must be
computed via finite differences (no AD through mpmath).

Usage:
    from qsc.forward_map_mp import forward_map_mp
    E = forward_map_mp(params_np, KONISHI, g, config, dps=50)
"""

# Enable float64 for quantum_numbers.py which uses JAX internally
import jax
jax.config.update("jax_enable_x64", True)

import mpmath
import numpy as np

from qsc.quantum_numbers import (
    QuantumNumbers,
    compute_A,
    compute_alfa,
    compute_B,
    compute_BB,
    compute_gauge_info,
    compute_kettoLAMBDA,
    compute_Mhat,
    compute_Mhat0,
    compute_Mt,
    compute_Mtint,
    compute_Nas,
)
from qsc.zhukovsky import cbinomial, kappa, kappabar


# ---------------------------------------------------------------------------
# Mpmath utility helpers
# ---------------------------------------------------------------------------

def _mpc(z):
    """Convert any number to mpmath complex."""
    if isinstance(z, mpmath.mpc):
        return z
    if isinstance(z, mpmath.mpf):
        return mpmath.mpc(z, 0)
    return mpmath.mpc(float(np.real(z)), float(np.imag(z)))


def _to_mp_1d(arr):
    """Convert 1D numpy array to list of mpmath mpc."""
    return [_mpc(arr[i]) for i in range(len(arr))]


def _to_mp_2d(arr):
    """Convert 2D numpy array to nested list of mpmath mpc."""
    return [[_mpc(arr[i, j]) for j in range(arr.shape[1])]
            for i in range(arr.shape[0])]


def _from_mp_1d(lst):
    """Convert list of mpmath mpc to numpy complex128."""
    return np.array([complex(z) for z in lst], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Zhukovsky map (mpmath versions)
# ---------------------------------------------------------------------------

def _x_of_u_long_mp(u, g):
    """Zhukovsky variable (long cut) at arbitrary precision."""
    return u / 2 - mpmath.mpc(0, 0.5) * mpmath.sqrt(4 - u**2)


def _x_of_u_short_mp(u):
    """Zhukovsky variable (short cut) at arbitrary precision."""
    return u * (1 + mpmath.sqrt(1 - 4 / u**2)) / 2


# ---------------------------------------------------------------------------
# Sigma table (mpmath version)
# ---------------------------------------------------------------------------

def _fsigma_mp(twiceMt, n, r, sqrt_g):
    """Sigma coefficient at arbitrary precision."""
    k = n // 2
    q0 = n % 2
    total = mpmath.mpf(0)
    for s in range(k - r + 1):
        total += kappabar(twiceMt, s) * kappa(2 * r + q0, k - r - s)
    total *= sqrt_g ** (twiceMt + 2 * n)
    return total


def _build_sigma_mp(twiceMt_arr, N0, NQ, g):
    """Build sigma[a][r][n] as nested lists of mpmath values."""
    sqrt_g = mpmath.sqrt(_mpc(g))
    sigma = [[[mpmath.mpc(0) for _ in range(NQ + 1)]
              for _ in range(N0 + 1)]
             for _ in range(4)]
    for a in range(4):
        tMt = int(twiceMt_arr[a])
        for r in range(N0 + 1):
            for n in range(NQ + 1):
                sigma[a][r][n] = _fsigma_mp(tMt, 2 * n, r, sqrt_g)
    return sigma


# ---------------------------------------------------------------------------
# Chebyshev grid (numpy, precomputed once)
# ---------------------------------------------------------------------------

def _chebyshev_grid(g, lc):
    """Chebyshev-Gauss points on [-2|g|, 2|g|]. Matches qsc.chebyshev."""
    k = np.arange(lc, dtype=np.float64)
    return -2 * np.real(g) * np.cos(np.pi * (2 * k + 1) / (2 * lc))


def _chebyshev_CT(lc):
    """Chebyshev-T transform matrix. Matches qsc.chebyshev."""
    n = np.arange(lc)
    k = np.arange(lc)
    return np.cos(np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * lc))


def _chebyshev_CU(CT, lc):
    """Chebyshev-U transform matrix. Matches qsc.chebyshev."""
    CU = np.zeros_like(CT)
    CU[:, :lc - 2] = (CT[:, :lc - 2] - CT[:, 2:lc]) / 2
    CU[:, lc - 2:] = CT[:, lc - 2:] / 2
    return CU


def _sqrt_weight(g, uA):
    """sqrt(4g^2 - u_k^2) weights. Matches qsc.chebyshev."""
    return np.sqrt(4 * g * g - uA**2)


# ---------------------------------------------------------------------------
# Core forward map functions (mpmath)
# ---------------------------------------------------------------------------

def _build_c_full_mp(Delta, c_n1_to_N0, A, Mt, g, gauge_indices):
    """Build c[a][0..N0] with c[a][0] = A[a] / g^Mt[a]."""
    N0 = len(c_n1_to_N0[0])
    g_mp = _mpc(g)
    c = [[mpmath.mpc(0) for _ in range(N0 + 1)] for _ in range(4)]
    for a in range(4):
        c[a][0] = _mpc(A[a]) / g_mp ** _mpc(Mt[a])
        for n in range(N0):
            c[a][n + 1] = _mpc(c_n1_to_N0[a][n])
    # Gauge fixing
    for a_g, n_g in gauge_indices:
        c[a_g][n_g] = mpmath.mpc(0)
    return c


def _compute_ksub_mp(c, sigma, NQ, N0):
    """ksub[a][n] = sum_r c[a][r] * sigma[a][r][n]."""
    ksub = [[mpmath.mpc(0) for _ in range(NQ + 1)] for _ in range(4)]
    for a in range(4):
        for n in range(NQ + 1):
            s = mpmath.mpc(0)
            for r in range(N0 + 1):
                s += c[a][r] * sigma[a][r][n]
            ksub[a][n] = s
    return ksub


def _compute_q_array_mp(ksub, AA, NQ):
    """q[n][a][b] via convolution of ksub."""
    m1_signs = [mpmath.mpf(-1), mpmath.mpf(1), mpmath.mpf(-1), mpmath.mpf(1)]
    q = [[[mpmath.mpc(0) for _ in range(4)] for _ in range(4)]
         for _ in range(NQ + 1)]
    for n in range(NQ + 1):
        for a in range(4):
            for b in range(4):
                conv = mpmath.mpc(0)
                for m in range(n + 1):
                    conv += ksub[a][m] * m1_signs[b] * ksub[3 - b][n - m]
                q[n][a][b] = conv / _mpc(AA[a, b])
    return q


def _build_scT_mp(AA, BB, alfa, NQ):
    """scT[i][m][a][b] matrices for the b-coefficient recursion."""
    II = mpmath.mpc(0, 1)
    scT = [[[[mpmath.mpc(0) for _ in range(4)] for _ in range(4)]
             for _ in range(NQ + 1)]
            for _ in range(4)]
    for i in range(4):
        for m in range(1, NQ + 1):
            for a in range(4):
                for b0 in range(4):
                    val = _mpc(AA[a, b0]) * _mpc(BB[b0, i])
                    if a == b0:
                        val -= II * _mpc(BB[a, i]) * (2 * m - _mpc(alfa[a, i]))
                    scT[i][m][a][b0] = val
    return scT


def _build_aux_tables_mp(alfa, NQ):
    """Build T1, T2, T41, S1, S31 tables. Matches JAX _build_auxiliary_tables."""
    lmax = NQ
    m1p4k = [mpmath.mpf(-0.25) ** j for j in range(lmax + 2)]
    m4k = [mpmath.mpf(-4) ** j for j in range(lmax + 2)]

    T1 = [[mpmath.mpc(0) for _ in range(max(lmax - 1, 1))]
          for _ in range(lmax + 1)]
    T2 = [[mpmath.mpc(0) for _ in range(max(lmax - 1, 1))]
          for _ in range(lmax + 1)]
    for l in range(2, lmax + 1):
        for k in range(min(l - 1, lmax - 1)):
            T1[l][k] = cbinomial(-2 * (k + 1), 2 * (l - k) - 1) * m1p4k[l - k - 1]
            T2[l][k] = cbinomial(-2 * (k + 1), 2 * (l - k - 1)) * m1p4k[l - k - 1]

    T41 = [[mpmath.mpc(0) for _ in range(lmax)]
           for _ in range(max(2 * lmax - 2, 1))]
    for m in range(2 * lmax - 2):
        ref = m // 2
        for k in range(min(ref + 1, lmax)):
            T41[m][k] = cbinomial(-2 * (k + 1), m - 2 * k) * m4k[k + 1]

    # S1[n][j] — matches JAX: cbinomial(-2*(j+1), 2*(n-j-1))
    S1 = [[mpmath.mpc(0) for _ in range(max(lmax - 1, 1))]
          for _ in range(lmax + 1)]
    for n in range(lmax + 1):
        for j in range(min(n - 1, lmax - 1)):
            if j >= 0:
                S1[n][j] = cbinomial(-2 * (j + 1), 2 * (n - j - 1)) * m1p4k[n - j - 1]

    # S31[k][j] — matches JAX: range 0..2*NQ-2, ref=(k+1)//2-1
    S31 = [[mpmath.mpc(0) for _ in range(max(lmax - 1, 1))]
           for _ in range(max(2 * lmax - 1, 1))]
    for k in range(2 * lmax - 1):
        ref = (k + 1) // 2 - 1
        for j in range(min(ref + 1, lmax - 1)):
            if j >= 0:
                S31[k][j] = cbinomial(-2 * (j + 1), k - 2 * j - 1) * m4k[j + 1]

    return {"T1": T1, "T2": T2, "T41": T41, "S1": S1, "S31": S31,
            "m1p4k": m1p4k, "m4k": m4k}


def _build_alfa_tables_mp(alfa, NQ, m1p4k):
    """Build T3, T5, S1n, S32 tables. Matches JAX _build_alfa_tables exactly.

    alfaais[m][a] = C(alfa[a,i], m) for m=0..2*NQ+1
    T3[l][a]      = alfaais[2*l+1][a] * m1p4k[l]
    T5[mm][l][a]   = alfaais[2*l-mm-1][a] * m1p4k[l]  (mm <= 2*l-3)
    S1n[n][a]     = alfaais[2*n][a] * m1p4k[n]
    S32[n][k][a]  = alfaais[2*n-k-1][a] * m1p4k[n]    (k <= 2*n-2)
    """
    lmax = NQ
    mm_dim = max(2 * lmax - 2, 1)
    kNQm1 = max(2 * lmax - 1, 1)
    max_m = 2 * lmax + 2

    T3 = []
    T5 = []
    S1n = []
    S32 = []

    for i in range(4):
        # Precompute alfaais[m][a] = C(alfa[a,i], m)
        alfaais = [[mpmath.mpc(0) for _ in range(4)] for _ in range(max_m)]
        for a in range(4):
            alfa_ai = complex(alfa[a, i])
            for m in range(max_m):
                alfaais[m][a] = _mpc(cbinomial(alfa_ai, m))

        # T3[l][a] = alfaais[2*l+1][a] * m1p4k[l]
        T3_i = [[mpmath.mpc(0) for _ in range(4)] for _ in range(lmax + 1)]
        for l in range(lmax + 1):
            idx = 2 * l + 1
            if idx < max_m:
                for a in range(4):
                    T3_i[l][a] = alfaais[idx][a] * m1p4k[l]
        T3.append(T3_i)

        # T5[mm][l][a] = alfaais[2*l - mm - 1][a] * m1p4k[l]
        T5_i = [[[mpmath.mpc(0) for _ in range(4)]
                  for _ in range(lmax + 1)]
                 for _ in range(mm_dim)]
        for mm in range(mm_dim):
            for l in range(lmax + 1):
                idx = 2 * l - mm - 1
                if mm <= 2 * l - 3 and 0 <= idx < max_m:
                    for a in range(4):
                        T5_i[mm][l][a] = alfaais[idx][a] * m1p4k[l]
        T5.append(T5_i)

        # S1n[n][a] = alfaais[2*n][a] * m1p4k[n]
        S1n_i = [[mpmath.mpc(0) for _ in range(4)] for _ in range(lmax + 1)]
        for n in range(lmax + 1):
            idx = 2 * n
            if idx < max_m:
                for a in range(4):
                    S1n_i[n][a] = alfaais[idx][a] * m1p4k[n]
        S1n.append(S1n_i)

        # S32[n][k][a] = alfaais[2*n - k - 1][a] * m1p4k[n]
        S32_i = [[[mpmath.mpc(0) for _ in range(4)]
                   for _ in range(kNQm1)]
                  for _ in range(lmax + 1)]
        for n in range(lmax + 1):
            for k in range(kNQm1):
                idx = 2 * n - k - 1
                if k <= 2 * n - 2 and 0 <= idx < max_m:
                    for a in range(4):
                        S32_i[n][k][a] = alfaais[idx][a] * m1p4k[n]
        S32.append(S32_i)

    return {"T3": T3, "T5": T5, "S1n": S1n, "S32": S32}


def _compute_F1_mp(i_idx, m, b, BB, alfa, T1, T2, T3_i, T41, T5_i, NQ):
    """F1 source for b-coefficient recursion (mpmath)."""
    mI = mpmath.mpc(0, -1)

    # T1s[a] = sum_{k=0}^{m-2} b[k+1][a] * T1[m][k]
    T1s = [mpmath.mpc(0)] * 4
    for a in range(4):
        for k in range(max(m - 1, 0)):
            if k < len(T1[m]):
                T1s[a] += b[k + 1][a] * T1[m][k]

    # T2s[a] = alfa[a,i] * sum_{k=0}^{m-2} b[k+1][a] * T2[m][k]
    T2s = [mpmath.mpc(0)] * 4
    for a in range(4):
        for k in range(max(m - 1, 0)):
            if k < len(T2[m]):
                T2s[a] += b[k + 1][a] * T2[m][k]
        T2s[a] *= _mpc(alfa[a, i_idx])

    # T3s[a] = b[0][a] * T3_i[m][a]
    T3s = [b[0][a] * T3_i[m][a] for a in range(4)]

    # T4s: sum over mm
    T4s = [mpmath.mpc(0)] * 4
    for mm in range(max(2 * m - 2, 0)):
        if mm < len(T41) and mm < len(T5_i):
            for a in range(4):
                t4part = mpmath.mpc(0)
                for k in range(min(mm // 2 + 1, len(T41[mm]))):
                    if k + 1 < len(b):
                        t4part += T41[mm][k] * b[k + 1][a]
                T4s[a] += T5_i[mm][m][a] * t4part

    F1 = [mI * _mpc(BB[a, i_idx]) * (T1s[a] + T2s[a] + T3s[a] + T4s[a])
          for a in range(4)]
    return F1


def _compute_F2_mp(i_idx, m, b, q, AA, BB, S1n_i, S1, S31, S32_i, NQ):
    """F2 source for b-coefficient recursion (mpmath)."""
    # S0s[a][b0] = sum_{n=1}^{m-1} b[n][b0] * q[m-n][a][b0]
    S0s = [[mpmath.mpc(0) for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b0 in range(4):
            for n in range(1, m):
                S0s[a][b0] += b[n][b0] * q[m - n][a][b0]

    # S_total[n][b0] = S1n + S2s + S3s for Sq computation
    # Matches JAX: S2s = S1 @ b, S3s = einsum(S32_i, S31 @ b)

    NQ1 = NQ + 1
    n_b_s1 = min(NQ1 - 1, len(S1[0]))  # columns of S1
    n_b_s31 = min(NQ1 - 1, len(S31[0]) if S31 else 0)
    k_max_s32 = len(S32_i[0]) if S32_i else 0

    # Precompute T4spart_s3[k][b0] = Σ_j S31[k][j] * b[j+1][b0]
    T4spart_s3 = [[mpmath.mpc(0) for _ in range(4)]
                   for _ in range(len(S31))]
    for k in range(len(S31)):
        for b0 in range(4):
            for j in range(n_b_s31):
                if j + 1 < len(b):
                    T4spart_s3[k][b0] += S31[k][j] * b[j + 1][b0]

    S_total = [[mpmath.mpc(0) for _ in range(4)] for _ in range(NQ1)]
    for n in range(NQ1):
        for b0 in range(4):
            # S1n
            val = S1n_i[n][b0]
            # S2s = Σ_j S1[n][j] * b[j+1][b0]
            for j in range(n_b_s1):
                if j + 1 < len(b):
                    val += b[j + 1][b0] * S1[n][j]
            # S3s = Σ_k S32_i[n][k][b0] * T4spart_s3[k][b0]
            for k in range(min(k_max_s32, len(T4spart_s3))):
                val += S32_i[n][k][b0] * T4spart_s3[k][b0]
            S_total[n][b0] = val

    # Sq[a][b0] = sum_{n=0}^{m} q[m-n][a][b0] * S_total[n][b0]
    Sq = [[mpmath.mpc(0) for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b0 in range(4):
            for n in range(m + 1):
                Sq[a][b0] += q[m - n][a][b0] * S_total[n][b0]

    # F2[a] = sum_b AA[a,b]*BB[b,i] * (S0s[a,b] + Sq[a,b])
    F2 = [mpmath.mpc(0)] * 4
    for a in range(4):
        for b0 in range(4):
            coeff = _mpc(AA[a, b0]) * _mpc(BB[b0, i_idx])
            F2[a] += coeff * (S0s[a][b0] + Sq[a][b0])
    return F2


def _solve_b_coefficients_mp(q, AA, BB, alfa, scT, aux_tables, alfa_tables, NQ):
    """Solve b[i][m][a] for all i via 4x4 linear systems."""
    T1, T2, T41 = aux_tables["T1"], aux_tables["T2"], aux_tables["T41"]
    S1, S31 = aux_tables["S1"], aux_tables["S31"]

    b_all = []
    for i in range(4):
        T3_i = alfa_tables["T3"][i]
        T5_i = alfa_tables["T5"][i]
        S1n_i = alfa_tables["S1n"][i]
        S32_i = alfa_tables["S32"][i]

        # b[m][a]: list of lists
        b = [[mpmath.mpc(0) for _ in range(4)] for _ in range(NQ + 1)]
        for a in range(4):
            b[0][a] = mpmath.mpc(1)

        for m in range(1, NQ + 1):
            F1 = _compute_F1_mp(i, m, b, BB, alfa, T1, T2, T3_i, T41, T5_i, NQ)
            F2 = _compute_F2_mp(i, m, b, q, AA, BB, S1n_i, S1, S31, S32_i, NQ)
            rhs = [F1[a] - F2[a] for a in range(4)]

            # Solve 4x4: scT[i][m] @ x = rhs
            mat = mpmath.matrix(4, 4)
            for a in range(4):
                for b0 in range(4):
                    mat[a, b0] = scT[i][m][a][b0]
            rhs_mp = mpmath.matrix(rhs)
            x = mpmath.lu_solve(mat, rhs_mp)
            for a in range(4):
                b[m][a] = x[a]

        b_all.append(b)
    return b_all


def _evaluate_Q_and_pulldown_mp(b_all, BB, Mt, Mhat, c, sigma,
                                 uA, g, NQ, NI, N0, lc):
    """Evaluate Q at large u and pull down to the cut. All in mpmath."""
    g_mp = _mpc(g)
    II = mpmath.mpc(0, 1)

    # 1. Q_upper[a][i][k] at u_shifted = uA[k] + i*(NI + 0.5)
    u_shifted = [_mpc(uA[k]) + II * (NI + mpmath.mpf(0.5)) for k in range(lc)]

    Q_upper = [[[mpmath.mpc(0) for _ in range(lc)]
                for _ in range(4)] for _ in range(4)]

    for a in range(4):
        for i in range(4):
            for k in range(lc):
                u_inv_sq = mpmath.mpf(1) / (u_shifted[k] ** 2)
                q_sum = mpmath.mpc(0)
                for n in range(NQ + 1):
                    q_sum += b_all[i][n][a] * (u_inv_sq ** n)
                exponent = _mpc(Mhat[i]) - _mpc(Mt[a])
                Q_upper[a][i][k] = _mpc(BB[a, i]) * (u_shifted[k] ** exponent) * q_sum

    # 2. Compute Puj[a][n_shift][k] for pulldown (n_shift = 0..NI-1)
    Puj = [[[mpmath.mpc(0) for _ in range(lc)]
            for _ in range(NI)] for _ in range(4)]

    for n_shift in range(NI):
        for k in range(lc):
            u_imag = (_mpc(uA[k]) + II * (n_shift + 1)) / g_mp
            x_val = _x_of_u_short_mp(u_imag)
            x_inv_sq = mpmath.mpf(1) / (x_val ** 2)
            for a in range(4):
                p_val = mpmath.mpc(0)
                x_inv_sq_n = mpmath.mpc(1)
                for m_idx in range(N0 + 1):
                    p_val += c[a][m_idx] * x_inv_sq_n
                    x_inv_sq_n *= x_inv_sq
                Puj[a][n_shift][k] = p_val / (x_val ** _mpc(Mt[a]))

    # 3. Pulldown: n from NI-1 down to 0
    m1_signs = [mpmath.mpf(-1), mpmath.mpf(1), mpmath.mpf(-1), mpmath.mpf(1)]

    for n in range(NI - 1, -1, -1):
        for k in range(lc):
            for i in range(4):
                Q_old = [Q_upper[a][i][k] for a in range(4)]
                contrib = mpmath.mpc(0)
                for b in range(4):
                    contrib += m1_signs[b] * Puj[3 - b][n][k] * Q_old[b]
                for a in range(4):
                    Q_upper[a][i][k] = Q_old[a] + Puj[a][n][k] * contrib

    # 4. P, Pt on the cut (using long-cut Zhukovsky)
    P = [[mpmath.mpc(0) for _ in range(lc)] for _ in range(4)]
    Pt = [[mpmath.mpc(0) for _ in range(lc)] for _ in range(4)]

    for k in range(lc):
        x_cut = _x_of_u_long_mp(_mpc(uA[k]) / g_mp, mpmath.mpf(1))
        x2 = x_cut ** 2
        for a in range(4):
            p_sum = mpmath.mpc(0)
            pt_sum = mpmath.mpc(0)
            x2n = mpmath.mpc(1)
            for m_idx in range(N0 + 1):
                p_sum += c[a][m_idx] * x2n
                pt_sum += c[a][m_idx] / x2n
                x2n *= x2
            xMt = x_cut ** _mpc(Mt[a])
            P[a][k] = xMt * p_sum
            Pt[a][k] = pt_sum / xMt

    # 5. Q_lower, Qt_lower
    Qlower = [[mpmath.mpc(0) for _ in range(4)] for _ in range(lc)]
    Qtlower = [[mpmath.mpc(0) for _ in range(4)] for _ in range(lc)]

    for k in range(lc):
        for i in range(4):
            ql = mpmath.mpc(0)
            qtl = mpmath.mpc(0)
            for a in range(4):
                signed = -m1_signs[a] * P[3 - a][k]
                signedt = -m1_signs[a] * Pt[3 - a][k]
                ql += signed * Q_upper[a][i][k]
                qtl += signedt * Q_upper[a][i][k]
            Qlower[k][i] = ql
            Qtlower[k][i] = qtl

    return Qlower, Qtlower, Q_upper, P, Pt


def _compute_gluing_mp(Q_upper, Qlower, Qtlower, lc):
    """Gluing constant and deltaP residual."""
    # alpha_Q
    alfaQ_sum = mpmath.mpc(0)
    for k in range(lc):
        alfaQ_sum += (Qlower[k][0] / mpmath.conj(Qlower[k][2])
                      + Qtlower[k][0] / mpmath.conj(Qtlower[k][2])
                      - Qlower[k][1] / mpmath.conj(Qlower[k][3])
                      - Qtlower[k][1] / mpmath.conj(Qtlower[k][3]))
    alfaQ = mpmath.re(alfaQ_sum / (4 * lc))

    # Gluing vectors G, Gt
    deltaP = [[mpmath.mpc(0) for _ in range(4)] for _ in range(lc)]
    deltaPt = [[mpmath.mpc(0) for _ in range(4)] for _ in range(lc)]

    for k in range(lc):
        G = [
            Qlower[k][3] + mpmath.conj(Qlower[k][1]) / alfaQ,
            -(Qlower[k][2] - mpmath.conj(Qlower[k][0]) / alfaQ),
            Qlower[k][1] + mpmath.conj(Qlower[k][3]) * alfaQ,
            -(Qlower[k][0] - mpmath.conj(Qlower[k][2]) * alfaQ),
        ]
        Gt = [
            Qtlower[k][3] + mpmath.conj(Qtlower[k][1]) / alfaQ,
            -(Qtlower[k][2] - mpmath.conj(Qtlower[k][0]) / alfaQ),
            Qtlower[k][1] + mpmath.conj(Qtlower[k][3]) * alfaQ,
            -(Qtlower[k][0] - mpmath.conj(Qtlower[k][2]) * alfaQ),
        ]
        for a in range(4):
            dp = mpmath.mpc(0)
            dpt = mpmath.mpc(0)
            for i in range(4):
                dp += Q_upper[a][i][k] * G[i]
                dpt += Q_upper[a][i][k] * Gt[i]
            deltaP[k][a] = dp
            deltaPt[k][a] = dpt

    return deltaP, deltaPt, alfaQ


def _fourier_inversion_mp(deltaP, deltaPt, CT, CU, suA,
                           Nas, Mtint_arr, Nch, N0, lc, g,
                           kettoLAMBDA):
    """Fourier inversion: deltaP → residual E. Returns numpy array."""
    # Convert deltaP/deltaPt to numpy for matrix ops
    dP = np.array([[complex(deltaP[k][a]) for a in range(4)]
                    for k in range(lc)], dtype=np.complex128)
    dPt = np.array([[complex(deltaPt[k][a]) for a in range(4)]
                     for k in range(lc)], dtype=np.complex128)

    # Half-integer correction
    if kettoLAMBDA % 2 != 0:
        x_cut_half = np.array([
            complex(_x_of_u_long_mp(_mpc(suA_orig) / _mpc(g), mpmath.mpf(1)))
            for suA_orig in _chebyshev_grid(g, lc)
        ])
        # This needs the actual x values — simplified for now
        pass  # TODO: implement if needed for non-Konishi states

    fS = (dP + dPt) / 2
    fA = np.zeros_like(dP)
    for k in range(lc):
        for a in range(4):
            fA[k, a] = (dP[k, a] - dPt[k, a]) / (2 * suA[k])

    CT_rev = CT[::-1, :]
    cS = CT_rev.T @ fS / lc
    CU_rev = CU[::-1, :]
    cA_inner = CU_rev.T @ fA / lc
    cA = np.zeros((lc, 4), dtype=np.complex128)
    cA[1:, :] = cA_inner[:lc - 1, :] * 2j * g

    # Assemble E
    Mtint_np = [int(Mtint_arr[i]) for i in range(4)]
    dimV = 1 + N0 + Nch[1] + Nch[2] + Nch[3]
    E = np.zeros(dimV, dtype=np.complex128)

    # a=0
    a = 0
    for n in range(N0 + 1):
        idx = abs(-Nas[a][0] + 2 * n)
        if idx < lc:
            if 2 * n >= Nas[a][0]:
                E[n] = cS[idx, a] + cA[idx, a]
            else:
                E[n] = cS[idx, a] - cA[idx, a]

    # a=1
    offset = N0 + 1
    a = 1
    k_skip = 0
    for n in range(1, N0 + 1):
        if 2 * n == Mtint_np[0] - Mtint_np[1]:
            k_skip += 1
            continue
        idx = abs(-Nas[a][0] + 2 * n)
        if idx < lc:
            e_idx = offset + n - 1 - k_skip
            if 2 * n >= Nas[a][0]:
                E[e_idx] = cS[idx, a] + cA[idx, a]
            else:
                E[e_idx] = cS[idx, a] - cA[idx, a]

    # a=2
    offset = N0 + 1 + Nch[1]
    a = 2
    k_skip = 0
    for n in range(1, N0 + 1):
        if (2 * n == Mtint_np[0] - Mtint_np[2] or
                2 * n == Mtint_np[1] - Mtint_np[2]):
            k_skip += 1
            continue
        idx = abs(-Nas[a][0] + 2 * n)
        if idx < lc:
            e_idx = offset + n - 1 - k_skip
            if 2 * n >= Nas[a][0]:
                E[e_idx] = cS[idx, a] + cA[idx, a]
            else:
                E[e_idx] = cS[idx, a] - cA[idx, a]

    # a=3
    offset = N0 + 1 + Nch[1] + Nch[2]
    a = 3
    k_skip = 0
    for n in range(1, N0 + 1):
        if 2 * n == Mtint_np[0] - Mtint_np[3]:
            k_skip += 1
            continue
        idx = abs(-Nas[a][0] + 2 * n)
        if idx < lc:
            e_idx = offset + n - 1 - k_skip
            if 2 * n >= Nas[a][0]:
                E[e_idx] = cS[idx, a] + cA[idx, a]
            else:
                E[e_idx] = cS[idx, a] - cA[idx, a]

    return E


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def forward_map_mp(params: np.ndarray, qn: QuantumNumbers,
                   g: float, cutP: int = 16, nPoints: int = 18,
                   cutQai: int = 30, QaiShift: int = 50,
                   dps: int = 50) -> np.ndarray:
    """Full forward map at arbitrary precision.

    Args:
        params: [Delta, c[0][1],...,c[3][N0]] as numpy complex128
        qn: quantum numbers
        g: coupling constant
        cutP, nPoints, cutQai, QaiShift: solver parameters
        dps: mpmath decimal digits of precision

    Returns:
        Residual E as numpy complex128 array (dimV elements)
    """
    mpmath.mp.dps = dps

    N0 = cutP // 2
    NQ = cutQai // 2
    NI = QaiShift
    lc = nPoints

    # Quantum number derived quantities (these are small arrays, numpy is fine)
    Mtint = compute_Mtint(qn)
    kettoLAMBDA = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kettoLAMBDA)
    twiceMt = 2 * Mtint + kettoLAMBDA
    Mhat0 = compute_Mhat0(qn, kettoLAMBDA)

    Delta = complex(params[0])
    Mhat = compute_Mhat(Mhat0, Delta)
    A_arr, Af, AA = compute_A(Mt, Mhat)
    B = compute_B(Mt, Mhat)
    BB = compute_BB(A_arr, B, Mt, Mhat)
    alfa = compute_alfa(Mt, Mhat)
    Nas = compute_Nas(Mtint, kettoLAMBDA)
    gauge_info = compute_gauge_info(Mtint, N0)
    Nch = gauge_info["Nch"]
    gauge_indices = gauge_info["gauge_indices"]

    # Ensure lc is large enough
    from qsc.chebyshev import ensure_min_lc
    lc = ensure_min_lc(lc, Nas, N0)

    # Unpack params
    c_n1_to_N0 = params[1:].reshape(4, N0)

    # Build full c (mpmath)
    c = _build_c_full_mp(Delta, c_n1_to_N0, A_arr, Mt, g, gauge_indices)

    # Chebyshev grid (numpy, precomputed)
    uA = _chebyshev_grid(g, lc)
    CT = _chebyshev_CT(lc)
    CU = _chebyshev_CU(CT, lc)
    suA = _sqrt_weight(g, uA)

    # Sigma table (mpmath)
    sigma = _build_sigma_mp(twiceMt, N0, NQ, g)

    # ksub (mpmath)
    ksub = _compute_ksub_mp(c, sigma, NQ, N0)

    # q-array (mpmath)
    q = _compute_q_array_mp(ksub, AA, NQ)

    # scT matrices (mpmath)
    scT = _build_scT_mp(AA, BB, alfa, NQ)

    # Auxiliary tables (mpmath)
    aux_tables = _build_aux_tables_mp(alfa, NQ)
    alfa_tables = _build_alfa_tables_mp(alfa, NQ, aux_tables["m1p4k"])

    # Solve b-coefficients (mpmath)
    b_all = _solve_b_coefficients_mp(q, AA, BB, alfa, scT,
                                      aux_tables, alfa_tables, NQ)

    # Q evaluation + pulldown (mpmath)
    Qlower, Qtlower, Q_upper, P, Pt = _evaluate_Q_and_pulldown_mp(
        b_all, BB, Mt, Mhat, c, sigma, uA, g, NQ, NI, N0, lc
    )

    # Gluing (mpmath)
    deltaP, deltaPt, alfaQ = _compute_gluing_mp(Q_upper, Qlower, Qtlower, lc)

    # Fourier inversion (numpy — converts from mpmath at entry)
    E = _fourier_inversion_mp(deltaP, deltaPt, CT, CU, suA,
                               Nas, Mtint, Nch, N0, lc, g,
                               kettoLAMBDA)

    return E
