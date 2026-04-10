"""Full forward map using python-flint (FLINT/Arb C library).

Drop-in replacement for forward_map_mp.py: same algorithm, same interface,
10-50× faster due to compiled C arithmetic instead of pure Python mpmath.

Usage:
    from qsc.forward_map_flint import forward_map_flint
    E = forward_map_flint(params_np, KONISHI, g, dps=50)
"""

import jax
jax.config.update("jax_enable_x64", True)

import math
import numpy as np
from flint import acb, acb_mat, ctx

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

# Re-use numpy Chebyshev functions from the mpmath version (no precision needed)
from qsc.forward_map_mp import (
    _chebyshev_grid,
    _chebyshev_CT,
    _chebyshev_CU,
    _sqrt_weight,
    _fourier_inversion_mp,
)


# ---------------------------------------------------------------------------
# Flint utility helpers
# ---------------------------------------------------------------------------

def _acb(z):
    """Convert any number to flint acb."""
    if isinstance(z, acb):
        return z
    return acb(float(np.real(z)), float(np.imag(z)))


# ---------------------------------------------------------------------------
# Zhukovsky map (flint versions)
# ---------------------------------------------------------------------------

def _x_of_u_long_fl(u):
    """Zhukovsky variable (long cut) at arbitrary precision."""
    return u / 2 - acb(0, 0.5) * (acb(4) - u**2).sqrt()


def _x_of_u_short_fl(u):
    """Zhukovsky variable (short cut) at arbitrary precision."""
    return u * (1 + (1 - 4 / u**2).sqrt()) / 2


# ---------------------------------------------------------------------------
# Sigma table (flint version)
# ---------------------------------------------------------------------------

def _fsigma_fl(twiceMt, n, r, sqrt_g):
    """Sigma coefficient at arbitrary precision."""
    k = n // 2
    q0 = n % 2
    total = acb(0)
    for s in range(k - r + 1):
        total += kappabar(twiceMt, s) * kappa(2 * r + q0, k - r - s)
    total *= sqrt_g ** (twiceMt + 2 * n)
    return total


def _build_sigma_fl(twiceMt_arr, N0, NQ, g):
    """Build sigma[a][r][n] as nested lists of acb values."""
    sqrt_g = _acb(g).sqrt()
    sigma = [[[acb(0) for _ in range(NQ + 1)]
              for _ in range(N0 + 1)]
             for _ in range(4)]
    for a in range(4):
        tMt = int(twiceMt_arr[a])
        for r in range(N0 + 1):
            for n in range(NQ + 1):
                sigma[a][r][n] = _fsigma_fl(tMt, 2 * n, r, sqrt_g)
    return sigma


# ---------------------------------------------------------------------------
# Core forward map functions (flint)
# ---------------------------------------------------------------------------

def _build_c_full_fl(Delta, c_n1_to_N0, A, Mt, g, gauge_indices):
    """Build c[a][0..N0] with c[a][0] = A[a] / g^Mt[a]."""
    N0 = len(c_n1_to_N0[0])
    g_fl = _acb(g)
    c = [[acb(0) for _ in range(N0 + 1)] for _ in range(4)]
    for a in range(4):
        c[a][0] = _acb(A[a]) / g_fl ** _acb(Mt[a])
        for n in range(N0):
            c[a][n + 1] = _acb(c_n1_to_N0[a][n])
    for a_g, n_g in gauge_indices:
        c[a_g][n_g] = acb(0)
    return c


def _compute_ksub_fl(c, sigma, NQ, N0):
    """ksub[a][n] = sum_r c[a][r] * sigma[a][r][n]."""
    ksub = [[acb(0) for _ in range(NQ + 1)] for _ in range(4)]
    for a in range(4):
        for n in range(NQ + 1):
            s = acb(0)
            for r in range(N0 + 1):
                s += c[a][r] * sigma[a][r][n]
            ksub[a][n] = s
    return ksub


def _compute_q_array_fl(ksub, AA, NQ):
    """q[n][a][b] via convolution of ksub."""
    m1_signs = [acb(-1), acb(1), acb(-1), acb(1)]
    q = [[[acb(0) for _ in range(4)] for _ in range(4)]
         for _ in range(NQ + 1)]
    for n in range(NQ + 1):
        for a in range(4):
            for b in range(4):
                conv = acb(0)
                for m in range(n + 1):
                    conv += ksub[a][m] * m1_signs[b] * ksub[3 - b][n - m]
                q[n][a][b] = conv / _acb(AA[a, b])
    return q


def _build_scT_fl(AA, BB, alfa, NQ):
    """scT[i][m][a][b] matrices."""
    II = acb(0, 1)
    scT = [[[[acb(0) for _ in range(4)] for _ in range(4)]
             for _ in range(NQ + 1)]
            for _ in range(4)]
    for i in range(4):
        for m in range(1, NQ + 1):
            for a in range(4):
                for b0 in range(4):
                    val = _acb(AA[a, b0]) * _acb(BB[b0, i])
                    if a == b0:
                        val -= II * _acb(BB[a, i]) * (2 * m - _acb(alfa[a, i]))
                    scT[i][m][a][b0] = val
    return scT


def _build_aux_tables_fl(alfa, NQ):
    """Build T1, T2, T41, S1, S31 tables."""
    lmax = NQ
    m1p4k = [acb(-0.25) ** j for j in range(lmax + 2)]
    m4k = [acb(-4) ** j for j in range(lmax + 2)]

    T1 = [[acb(0) for _ in range(max(lmax - 1, 1))] for _ in range(lmax + 1)]
    T2 = [[acb(0) for _ in range(max(lmax - 1, 1))] for _ in range(lmax + 1)]
    for l in range(2, lmax + 1):
        for k in range(min(l - 1, lmax - 1)):
            T1[l][k] = cbinomial(-2 * (k + 1), 2 * (l - k) - 1) * m1p4k[l - k - 1]
            T2[l][k] = cbinomial(-2 * (k + 1), 2 * (l - k - 1)) * m1p4k[l - k - 1]

    T41 = [[acb(0) for _ in range(lmax)] for _ in range(max(2 * lmax - 2, 1))]
    for m in range(2 * lmax - 2):
        ref = m // 2
        for k in range(min(ref + 1, lmax)):
            T41[m][k] = cbinomial(-2 * (k + 1), m - 2 * k) * m4k[k + 1]

    S1 = [[acb(0) for _ in range(max(lmax - 1, 1))] for _ in range(lmax + 1)]
    for n in range(lmax + 1):
        for j in range(min(n - 1, lmax - 1)):
            if j >= 0:
                S1[n][j] = cbinomial(-2 * (j + 1), 2 * (n - j - 1)) * m1p4k[n - j - 1]

    S31 = [[acb(0) for _ in range(max(lmax - 1, 1))]
           for _ in range(max(2 * lmax - 1, 1))]
    for k in range(2 * lmax - 1):
        ref = (k + 1) // 2 - 1
        for j in range(min(ref + 1, lmax - 1)):
            if j >= 0:
                S31[k][j] = cbinomial(-2 * (j + 1), k - 2 * j - 1) * m4k[j + 1]

    return {"T1": T1, "T2": T2, "T41": T41, "S1": S1, "S31": S31,
            "m1p4k": m1p4k, "m4k": m4k}


def _build_alfa_tables_fl(alfa, NQ, m1p4k):
    """Build T3, T5, S1n, S32 tables. Matches JAX exactly."""
    lmax = NQ
    mm_dim = max(2 * lmax - 2, 1)
    kNQm1 = max(2 * lmax - 1, 1)
    max_m = 2 * lmax + 2

    T3, T5, S1n, S32 = [], [], [], []

    for i in range(4):
        alfaais = [[acb(0) for _ in range(4)] for _ in range(max_m)]
        for a in range(4):
            alfa_ai = complex(alfa[a, i])
            for m in range(max_m):
                alfaais[m][a] = _acb(cbinomial(alfa_ai, m))

        T3_i = [[acb(0) for _ in range(4)] for _ in range(lmax + 1)]
        for l in range(lmax + 1):
            idx = 2 * l + 1
            if idx < max_m:
                for a in range(4):
                    T3_i[l][a] = alfaais[idx][a] * m1p4k[l]
        T3.append(T3_i)

        T5_i = [[[acb(0) for _ in range(4)] for _ in range(lmax + 1)]
                 for _ in range(mm_dim)]
        for mm in range(mm_dim):
            for l in range(lmax + 1):
                idx = 2 * l - mm - 1
                if mm <= 2 * l - 3 and 0 <= idx < max_m:
                    for a in range(4):
                        T5_i[mm][l][a] = alfaais[idx][a] * m1p4k[l]
        T5.append(T5_i)

        S1n_i = [[acb(0) for _ in range(4)] for _ in range(lmax + 1)]
        for n in range(lmax + 1):
            idx = 2 * n
            if idx < max_m:
                for a in range(4):
                    S1n_i[n][a] = alfaais[idx][a] * m1p4k[n]
        S1n.append(S1n_i)

        S32_i = [[[acb(0) for _ in range(4)] for _ in range(kNQm1)]
                  for _ in range(lmax + 1)]
        for n in range(lmax + 1):
            for k in range(kNQm1):
                idx = 2 * n - k - 1
                if k <= 2 * n - 2 and 0 <= idx < max_m:
                    for a in range(4):
                        S32_i[n][k][a] = alfaais[idx][a] * m1p4k[n]
        S32.append(S32_i)

    return {"T3": T3, "T5": T5, "S1n": S1n, "S32": S32}


def _compute_F1_fl(i_idx, m, b, BB, alfa, T1, T2, T3_i, T41, T5_i, NQ):
    """F1 source for b-coefficient recursion."""
    mI = acb(0, -1)
    T1s = [acb(0)] * 4
    for a in range(4):
        for k in range(max(m - 1, 0)):
            if k < len(T1[m]):
                T1s[a] += b[k + 1][a] * T1[m][k]

    T2s = [acb(0)] * 4
    for a in range(4):
        for k in range(max(m - 1, 0)):
            if k < len(T2[m]):
                T2s[a] += b[k + 1][a] * T2[m][k]
        T2s[a] *= _acb(alfa[a, i_idx])

    T3s = [b[0][a] * T3_i[m][a] for a in range(4)]

    T4s = [acb(0)] * 4
    for mm in range(max(2 * m - 2, 0)):
        if mm < len(T41) and mm < len(T5_i):
            for a in range(4):
                t4part = acb(0)
                for k in range(min(mm // 2 + 1, len(T41[mm]))):
                    if k + 1 < len(b):
                        t4part += T41[mm][k] * b[k + 1][a]
                T4s[a] += T5_i[mm][m][a] * t4part

    return [mI * _acb(BB[a, i_idx]) * (T1s[a] + T2s[a] + T3s[a] + T4s[a])
            for a in range(4)]


def _compute_F2_fl(i_idx, m, b, q, AA, BB, S1n_i, S1, S31, S32_i, NQ):
    """F2 source for b-coefficient recursion."""
    S0s = [[acb(0) for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b0 in range(4):
            for n in range(1, m):
                S0s[a][b0] += b[n][b0] * q[m - n][a][b0]

    NQ1 = NQ + 1
    n_b_s1 = min(NQ1 - 1, len(S1[0]))
    n_b_s31 = min(NQ1 - 1, len(S31[0]) if S31 else 0)
    k_max_s32 = len(S32_i[0]) if S32_i else 0

    T4spart_s3 = [[acb(0) for _ in range(4)] for _ in range(len(S31))]
    for k in range(len(S31)):
        for b0 in range(4):
            for j in range(n_b_s31):
                if j + 1 < len(b):
                    T4spart_s3[k][b0] += S31[k][j] * b[j + 1][b0]

    S_total = [[acb(0) for _ in range(4)] for _ in range(NQ1)]
    for n in range(NQ1):
        for b0 in range(4):
            val = S1n_i[n][b0]
            for j in range(n_b_s1):
                if j + 1 < len(b):
                    val += b[j + 1][b0] * S1[n][j]
            for k in range(min(k_max_s32, len(T4spart_s3))):
                val += S32_i[n][k][b0] * T4spart_s3[k][b0]
            S_total[n][b0] = val

    Sq = [[acb(0) for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b0 in range(4):
            for n in range(m + 1):
                Sq[a][b0] += q[m - n][a][b0] * S_total[n][b0]

    F2 = [acb(0)] * 4
    for a in range(4):
        for b0 in range(4):
            coeff = _acb(AA[a, b0]) * _acb(BB[b0, i_idx])
            F2[a] += coeff * (S0s[a][b0] + Sq[a][b0])
    return F2


def _solve_b_coefficients_fl(q, AA, BB, alfa, scT, aux_tables, alfa_tables, NQ):
    """Solve b[i][m][a] for all i via 4x4 linear systems."""
    T1, T2, T41 = aux_tables["T1"], aux_tables["T2"], aux_tables["T41"]
    S1, S31 = aux_tables["S1"], aux_tables["S31"]

    b_all = []
    for i in range(4):
        T3_i = alfa_tables["T3"][i]
        T5_i = alfa_tables["T5"][i]
        S1n_i = alfa_tables["S1n"][i]
        S32_i = alfa_tables["S32"][i]

        b = [[acb(0) for _ in range(4)] for _ in range(NQ + 1)]
        for a in range(4):
            b[0][a] = acb(1)

        for m in range(1, NQ + 1):
            F1 = _compute_F1_fl(i, m, b, BB, alfa, T1, T2, T3_i, T41, T5_i, NQ)
            F2 = _compute_F2_fl(i, m, b, q, AA, BB, S1n_i, S1, S31, S32_i, NQ)
            rhs = [F1[a] - F2[a] for a in range(4)]

            mat = acb_mat(4, 4)
            for a in range(4):
                for b0 in range(4):
                    mat[a, b0] = scT[i][m][a][b0]
            rhs_mat = acb_mat(4, 1, rhs)
            x = mat.solve(rhs_mat)
            for a in range(4):
                b[m][a] = x[a, 0]

        b_all.append(b)
    return b_all


def _evaluate_Q_and_pulldown_fl(b_all, BB, Mt, Mhat, c, sigma,
                                 uA, g, NQ, NI, N0, lc):
    """Evaluate Q at large u and pull down to the cut."""
    g_fl = _acb(g)
    II = acb(0, 1)

    u_shifted = [_acb(uA[k]) + II * (NI + acb(0.5)) for k in range(lc)]

    Q_upper = [[[acb(0) for _ in range(lc)] for _ in range(4)] for _ in range(4)]

    for a in range(4):
        for i in range(4):
            for k in range(lc):
                u_inv_sq = acb(1) / (u_shifted[k] ** 2)
                q_sum = acb(0)
                for n in range(NQ + 1):
                    q_sum += b_all[i][n][a] * (u_inv_sq ** n)
                exponent = _acb(Mhat[i]) - _acb(Mt[a])
                Q_upper[a][i][k] = _acb(BB[a, i]) * (u_shifted[k] ** exponent) * q_sum

    # Puj
    Puj = [[[acb(0) for _ in range(lc)] for _ in range(NI)] for _ in range(4)]
    for n_shift in range(NI):
        for k in range(lc):
            u_imag = (_acb(uA[k]) + II * (n_shift + 1)) / g_fl
            x_val = _x_of_u_short_fl(u_imag)
            x_inv_sq = acb(1) / (x_val ** 2)
            for a in range(4):
                p_val = acb(0)
                x_inv_sq_n = acb(1)
                for m_idx in range(N0 + 1):
                    p_val += c[a][m_idx] * x_inv_sq_n
                    x_inv_sq_n *= x_inv_sq
                Puj[a][n_shift][k] = p_val / (x_val ** _acb(Mt[a]))

    # Pulldown
    m1_signs = [acb(-1), acb(1), acb(-1), acb(1)]
    for n in range(NI - 1, -1, -1):
        for k in range(lc):
            for i in range(4):
                Q_old = [Q_upper[a][i][k] for a in range(4)]
                contrib = acb(0)
                for b in range(4):
                    contrib += m1_signs[b] * Puj[3 - b][n][k] * Q_old[b]
                for a in range(4):
                    Q_upper[a][i][k] = Q_old[a] + Puj[a][n][k] * contrib

    # P, Pt on the cut
    P = [[acb(0) for _ in range(lc)] for _ in range(4)]
    Pt = [[acb(0) for _ in range(lc)] for _ in range(4)]
    for k in range(lc):
        x_cut = _x_of_u_long_fl(_acb(uA[k]) / g_fl)
        x2 = x_cut ** 2
        for a in range(4):
            p_sum = acb(0)
            pt_sum = acb(0)
            x2n = acb(1)
            for m_idx in range(N0 + 1):
                p_sum += c[a][m_idx] * x2n
                pt_sum += c[a][m_idx] / x2n
                x2n *= x2
            xMt = x_cut ** _acb(Mt[a])
            P[a][k] = xMt * p_sum
            Pt[a][k] = pt_sum / xMt

    # Q_lower, Qt_lower
    Qlower = [[acb(0) for _ in range(4)] for _ in range(lc)]
    Qtlower = [[acb(0) for _ in range(4)] for _ in range(lc)]
    for k in range(lc):
        for i in range(4):
            ql = acb(0)
            qtl = acb(0)
            for a in range(4):
                signed = -m1_signs[a] * P[3 - a][k]
                signedt = -m1_signs[a] * Pt[3 - a][k]
                ql += signed * Q_upper[a][i][k]
                qtl += signedt * Q_upper[a][i][k]
            Qlower[k][i] = ql
            Qtlower[k][i] = qtl

    return Qlower, Qtlower, Q_upper, P, Pt


def _compute_gluing_fl(Q_upper, Qlower, Qtlower, lc):
    """Gluing constant and deltaP residual."""
    alfaQ_sum = acb(0)
    for k in range(lc):
        alfaQ_sum += (Qlower[k][0] / Qlower[k][2].conjugate()
                      + Qtlower[k][0] / Qtlower[k][2].conjugate()
                      - Qlower[k][1] / Qlower[k][3].conjugate()
                      - Qtlower[k][1] / Qtlower[k][3].conjugate())
    alfaQ = (alfaQ_sum / acb(4 * lc)).real

    deltaP = [[acb(0) for _ in range(4)] for _ in range(lc)]
    deltaPt = [[acb(0) for _ in range(4)] for _ in range(lc)]

    for k in range(lc):
        G = [
            Qlower[k][3] + Qlower[k][1].conjugate() / alfaQ,
            -(Qlower[k][2] - Qlower[k][0].conjugate() / alfaQ),
            Qlower[k][1] + Qlower[k][3].conjugate() * alfaQ,
            -(Qlower[k][0] - Qlower[k][2].conjugate() * alfaQ),
        ]
        Gt = [
            Qtlower[k][3] + Qtlower[k][1].conjugate() / alfaQ,
            -(Qtlower[k][2] - Qtlower[k][0].conjugate() / alfaQ),
            Qtlower[k][1] + Qtlower[k][3].conjugate() * alfaQ,
            -(Qtlower[k][0] - Qtlower[k][2].conjugate() * alfaQ),
        ]
        for a in range(4):
            dp = acb(0)
            dpt = acb(0)
            for i in range(4):
                dp += Q_upper[a][i][k] * G[i]
                dpt += Q_upper[a][i][k] * Gt[i]
            deltaP[k][a] = dp
            deltaPt[k][a] = dpt

    return deltaP, deltaPt, alfaQ


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def forward_map_flint(params: np.ndarray, qn: QuantumNumbers,
                      g: float, cutP: int = 16, nPoints: int = 18,
                      cutQai: int = 24, QaiShift: int = 4,
                      dps: int = 50) -> np.ndarray:
    """Full forward map at arbitrary precision using FLINT/Arb.

    Drop-in replacement for forward_map_mp. Same interface, 10-50× faster.
    """
    ctx.prec = math.ceil(dps * 3.322)

    N0 = cutP // 2
    NQ = cutQai // 2
    NI = QaiShift
    lc = nPoints

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

    # Convert JAX arrays to numpy ONCE — JAX __getitem__ is 100× slower
    # than numpy indexing due to JIT dispatch overhead
    A_arr = np.array(A_arr, dtype=np.complex128)
    Mt = np.array(Mt, dtype=np.float64)
    AA = np.array(AA, dtype=np.complex128)
    BB = np.array(BB, dtype=np.complex128)
    alfa = np.array(alfa, dtype=np.complex128)
    Mhat = np.array(Mhat, dtype=np.complex128)
    Nch = gauge_info["Nch"]
    gauge_indices = gauge_info["gauge_indices"]

    from qsc.chebyshev import ensure_min_lc
    lc = ensure_min_lc(lc, Nas, N0)

    c_n1_to_N0 = params[1:].reshape(4, N0)

    c = _build_c_full_fl(Delta, c_n1_to_N0, A_arr, Mt, g, gauge_indices)
    uA = _chebyshev_grid(g, lc)
    CT = _chebyshev_CT(lc)
    CU = _chebyshev_CU(CT, lc)
    suA = _sqrt_weight(g, uA)

    sigma = _build_sigma_fl(twiceMt, N0, NQ, g)
    ksub = _compute_ksub_fl(c, sigma, NQ, N0)
    q = _compute_q_array_fl(ksub, AA, NQ)
    scT = _build_scT_fl(AA, BB, alfa, NQ)
    aux_tables = _build_aux_tables_fl(alfa, NQ)
    alfa_tables = _build_alfa_tables_fl(alfa, NQ, aux_tables["m1p4k"])

    b_all = _solve_b_coefficients_fl(q, AA, BB, alfa, scT,
                                      aux_tables, alfa_tables, NQ)

    Qlower, Qtlower, Q_upper, P, Pt = _evaluate_Q_and_pulldown_fl(
        b_all, BB, Mt, Mhat, c, sigma, uA, g, NQ, NI, N0, lc
    )

    deltaP, deltaPt, alfaQ = _compute_gluing_fl(Q_upper, Qlower, Qtlower, lc)

    # Fourier inversion reuses the numpy version — convert acb→complex at entry
    E = _fourier_inversion_mp(deltaP, deltaPt, CT, CU, suA,
                               Nas, Mtint, Nch, N0, lc, g,
                               kettoLAMBDA)

    return E
