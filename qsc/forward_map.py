"""Complete forward map for QSC TypeI: (params, g) -> residual.

This is the core computation that must be JAX-traceable for automatic
differentiation. Given expansion coefficients c_{a,n} and the anomalous
dimension Delta, it computes the self-consistency residual F.

The algorithm follows TypeI_core.cpp:
1. Construct P_a(u) on the Chebyshev grid
2. Build 1/u expansion coefficients (ksub)
3. Compute b_{a|i,n} via sequential 4x4 linear solves
4. Evaluate Q_{a|i}(u) at large u, pull down to the cut
5. Compute Q_lower, Q_tilde_lower via P-function contraction
6. Compute gluing constant alpha_Q
7. Compute deltaP residual from gluing equations
8. Fourier-invert deltaP to get equation residual E
"""

import jax
import jax.numpy as jnp

from qsc.chebyshev import (
    chebyshev_CT,
    chebyshev_CU,
    chebyshev_grid,
    ensure_min_lc,
    sqrt_weight,
)
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
from qsc.zhukovsky import (
    build_sigma_table,
    cbinomial,
    x_of_u_long,
    x_of_u_short,
)


from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfig:
    """Hyperparameters for the QSC solver."""

    cutP: int = 16        # Fourier mode cutoff (N_trunc = cutP/2 for TypeI)
    nPoints: int = 18     # Chebyshev grid size (typically cutP + 2)
    cutQai: int = 24      # Large-u expansion order for Q
    QaiShift: int = 60    # Imaginary pull-down cutoff

    @property
    def N0(self) -> int:
        """Truncation order (number of c_{a,n} coefficients per a)."""
        return self.cutP // 2

    @property
    def NQ(self) -> int:
        """Q expansion cutoff."""
        return self.cutQai // 2

    @property
    def NI(self) -> int:
        """Imaginary pull-down steps."""
        return self.QaiShift


def _unpack_params_typeI(params: jnp.ndarray, N0: int) -> tuple:
    """Unpack flat parameter vector to (Delta, c[4][N0+1]).

    TypeI format: [Delta, c[0,1], c[0,2], ..., c[0,N0],
                          c[1,1], ..., c[1,Nch1], ...]

    But note: the params from the Mathematica pipeline use a different format.
    Here we use the internal C++ convention where c[a] has N0+1 entries (n=0..N0).
    c[a][0] is computed from Delta, not stored in params.

    For the initial implementation, params = [Delta, c[0,1],...,c[0,N0],
    c[1,1],...,c[1,N0], c[2,1],...,c[2,N0], c[3,1],...,c[3,N0]].
    Total length = 1 + 4*N0.
    """
    Delta = params[0]
    c_flat = params[1:]
    c = c_flat.reshape(4, N0)  # c[a][n] for n=1..N0
    return Delta, c


def _build_c_full(Delta: complex, c_n1_to_N0: jnp.ndarray,
                  A: jnp.ndarray, Mt: jnp.ndarray, g: complex,
                  Mtint: jnp.ndarray, gauge_indices: list) -> list:
    """Build full c[a][0..N0] arrays including c[a][0] from A and gauge fixes.

    c[a][0] = A[a] / g^{Mt[a]}

    For TypeI, c[0] and c[2] are imaginary, c[1] and c[3] are real.
    The C++ code stores: c[0][n] = i * c0_input[n], c[2][n] = i * c2_input[n].
    """
    N0 = c_n1_to_N0.shape[1]
    c = []
    for a in range(4):
        c_a = jnp.zeros(N0 + 1, dtype=complex)
        # c[a][0] = A[a] / g^Mt[a]
        c_a = c_a.at[0].set(A[a] / g**Mt[a])
        # c[a][1..N0] from params
        c_a = c_a.at[1:].set(c_n1_to_N0[a])
        c.append(c_a)

    # Apply gauge fixing: set specified coefficients to zero
    for a_gauge, n_gauge in gauge_indices:
        c[a_gauge] = c[a_gauge].at[n_gauge].set(0.0 + 0j)

    return c


def _compute_ksub(c: list, sigma: list, NQ: int, N0: int) -> list:
    """Compute 1/u expansion coefficients of P_a(u).

    ksub[a][n] = sum_{r=0}^{min(n,N0)} c[a][r] * sigma[a][r, n]

    sigma[a] has shape (N0+1, NQ+1).
    """
    ksub = []
    for a in range(4):
        k_a = jnp.zeros(NQ + 1, dtype=complex)
        for n in range(NQ + 1):
            rmax = min(n, N0)
            k_a = k_a.at[n].set(jnp.sum(c[a][:rmax + 1] * sigma[a][:rmax + 1, n]))
        ksub.append(k_a)
    return ksub


def _compute_q_array(ksub: list, AA: jnp.ndarray, NQ: int) -> jnp.ndarray:
    """Compute q-array: q[n, a, b] = sum_m ksub[a][m]*(-1)^{b+1}*ksub[3-b][n-m] / AA[a,b].

    Returns array of shape (NQ+1, 4, 4).
    """
    m1_signs = jnp.array([(-1.0) ** (b + 1) for b in range(4)])

    q = jnp.zeros((NQ + 1, 4, 4), dtype=complex)
    for n in range(NQ + 1):
        for a in range(4):
            for b in range(4):
                val = 0.0 + 0j
                for m in range(n + 1):
                    val += ksub[a][m] * m1_signs[b] * ksub[3 - b][n - m]
                q = q.at[n, a, b].set(val / AA[a, b])
    return q


def _build_scT_matrices(AA: jnp.ndarray, BB: jnp.ndarray, alfa: jnp.ndarray,
                        NQ: int) -> list:
    """Build scT[i][m] matrices (4x4) for each i=0..3, m=0..NQ.

    From totalscTmaker2LRi in C++ (note: the C++ 'B' parameter is actually BB):
    scT[m][a][b0] = AA[a][b0]*BB[b0][i]  for a != b0
    scT[m][a][a]  = AA[a][a]*BB[a][i] - i*BB[a][i]*(2m - alfa[a][i])
    scT[0] = 0.
    """
    II = 1j
    scT_all = []
    for i in range(4):
        scT_i = jnp.zeros((NQ + 1, 4, 4), dtype=complex)
        for m in range(1, NQ + 1):
            for a in range(4):
                for b0 in range(4):
                    if a == b0:
                        val = AA[a, b0] * BB[b0, i] - II * BB[a, i] * (2 * m - alfa[a, i])
                    else:
                        val = AA[a, b0] * BB[b0, i]
                    scT_i = scT_i.at[m, a, b0].set(val)
        scT_all.append(scT_i)
    return scT_all


def _build_auxiliary_tables(alfa: jnp.ndarray, NQ: int) -> dict:
    """Build T1, T2, T41, T3, T5, S1, S1n, S31, S32 tables.

    These are precomputed arrays of binomial coefficients used in the
    F1 and F2 source functions for the b-coefficient recursion.
    """
    lmax = NQ
    m1p4k = jnp.array([(-0.25) ** j for j in range(lmax + 2)])
    m4k = jnp.array([(-4.0) ** j for j in range(lmax + 2)])

    # T1[l, k] for l=0..lmax, k=0..lmax-2
    T1 = jnp.zeros((lmax + 1, max(lmax - 1, 1)), dtype=complex)
    for l in range(2, lmax + 1):
        for k in range(min(l - 1, lmax - 1)):
            T1 = T1.at[l, k].set(
                cbinomial(-2 * (k + 1), 2 * (l - k) - 1) * m1p4k[l - k - 1]
            )

    # T2[l, k]
    T2 = jnp.zeros((lmax + 1, max(lmax - 1, 1)), dtype=complex)
    for l in range(2, lmax + 1):
        for k in range(min(l - 1, lmax - 1)):
            T2 = T2.at[l, k].set(
                cbinomial(-2 * (k + 1), 2 * (l - k - 1)) * m1p4k[l - k - 1]
            )

    # T41[m, k] for m=0..2*lmax-3, k=0..lmax-1
    T41 = jnp.zeros((max(2 * lmax - 2, 1), lmax), dtype=complex)
    for m in range(2 * lmax - 2):
        ref = m // 2
        for k in range(min(ref + 1, lmax)):
            T41 = T41.at[m, k].set(
                cbinomial(-2 * (k + 1), m - 2 * k) * m4k[k + 1]
            )

    # S1[n, j] for n=0..NQ, j=0..NQ-2
    S1 = jnp.zeros((NQ + 1, max(NQ - 1, 1)), dtype=complex)
    for n in range(NQ + 1):
        for j in range(min(n - 1, NQ - 1)):
            if j >= 0:
                S1 = S1.at[n, j].set(
                    cbinomial(-2 * (j + 1), 2 * (n - j - 1)) * m1p4k[n - j - 1]
                )

    # S31[k, j] for k=0..2*NQ-2, j=0..NQ-2
    S31 = jnp.zeros((max(2 * NQ - 1, 1), max(NQ - 1, 1)), dtype=complex)
    for k in range(2 * NQ - 1):
        ref = (k + 1) // 2 - 1
        for j in range(min(ref + 1, NQ - 1)):
            if j >= 0:
                S31 = S31.at[k, j].set(
                    cbinomial(-2 * (j + 1), k - 2 * j - 1) * m4k[j + 1]
                )

    return {"T1": T1, "T2": T2, "T41": T41, "S1": S1, "S31": S31,
            "m1p4k": m1p4k, "m4k": m4k}


def _build_alfa_tables(alfa: jnp.ndarray, NQ: int,
                       m1p4k: jnp.ndarray) -> dict:
    """Build alfa-dependent auxiliary tables for each i.

    alfaais[i][m, a] = C(alfa[a][i], m)   (binomial)
    T3[i][l, a] = alfaais[2l+1, a] * m1p4k[l]
    T5[i][m, l, a] = alfaais[2l-m-1, a] * m1p4k[l]   for m <= 2l-3
    S1n[i][n, a] = alfaais[2n, a] * m1p4k[n]
    S32[i][n, k, a] = alfaais[2n-k-1, a] * m1p4k[n]   for k <= 2n-2
    """
    result = {"alfaais": [], "T3": [], "T5": [], "S1n": [], "S32": []}

    for i in range(4):
        # alfaais[m, a] for m=0..2*NQ+1
        alfaais = jnp.zeros((2 * NQ + 2, 4), dtype=complex)
        for a in range(4):
            for m in range(2 * NQ + 2):
                alfaais = alfaais.at[m, a].set(cbinomial(alfa[a, i], m))

        # T3[l, a]
        T3 = jnp.zeros((NQ + 1, 4), dtype=complex)
        for l in range(NQ + 1):
            for a in range(4):
                T3 = T3.at[l, a].set(alfaais[2 * l + 1, a] * m1p4k[l])

        # T5[m, l, a] — m up to 2*NQ-3, l up to NQ
        T5 = jnp.zeros((max(2 * NQ - 2, 1), NQ + 1, 4), dtype=complex)
        for l in range(2, NQ + 1):
            ref = 2 * l - 3
            for m in range(min(ref + 1, max(2 * NQ - 2, 1))):
                for a in range(4):
                    T5 = T5.at[m, l, a].set(alfaais[2 * l - m - 1, a] * m1p4k[l])

        # S1n[n, a]
        S1n = jnp.zeros((NQ + 1, 4), dtype=complex)
        for n in range(NQ + 1):
            for a in range(4):
                S1n = S1n.at[n, a].set(alfaais[2 * n, a] * m1p4k[n])

        # S32[n, k, a]
        kNQm1 = max(2 * NQ - 1, 1)
        S32 = jnp.zeros((NQ + 1, kNQm1, 4), dtype=complex)
        for n in range(NQ + 1):
            for k in range(min(2 * n - 1, kNQm1)):
                if k >= 0:
                    for a in range(4):
                        S32 = S32.at[n, k, a].set(
                            alfaais[2 * n - k - 1, a] * m1p4k[n]
                        )

        result["alfaais"].append(alfaais)
        result["T3"].append(T3)
        result["T5"].append(T5)
        result["S1n"].append(S1n)
        result["S32"].append(S32)

    return result


def _compute_F1(i: int, m: int, b: jnp.ndarray, BB: jnp.ndarray,
                alfa: jnp.ndarray, T1: jnp.ndarray, T2: jnp.ndarray,
                T3_i: jnp.ndarray, T41: jnp.ndarray,
                T5_i: jnp.ndarray) -> jnp.ndarray:
    """Compute F1 source for b-coefficient recursion at step (i, m).

    F1[a] = -i * BB[a,i] * (T1s + T2s + T3s + T4s)
    """
    mI = -1j
    F1 = jnp.zeros(4, dtype=complex)

    for a in range(4):
        # T1s
        T1s = 0.0 + 0j
        for k in range(m - 1):
            T1s += b[k + 1, a] * T1[m, k]

        # T2s
        T2s = 0.0 + 0j
        for k in range(m - 1):
            T2s += b[k + 1, a] * T2[m, k]
        T2s *= alfa[a, i]

        # T3s
        T3s = b[0, a] * T3_i[m, a]

        # T4s
        T4s = 0.0 + 0j
        for mm in range(max(2 * m - 2, 0)):
            ref = mm // 2
            T4spart = 0.0 + 0j
            for k in range(ref + 1):
                T4spart += b[k + 1, a] * T41[mm, k]
            T4s += T5_i[mm, m, a] * T4spart

        F1 = F1.at[a].set(mI * BB[a, i] * (T1s + T2s + T3s + T4s))
    return F1


def _compute_F2(i: int, m: int, b: jnp.ndarray, q: jnp.ndarray,
                AA: jnp.ndarray, BB: jnp.ndarray,
                S1n_i: jnp.ndarray, S1: jnp.ndarray,
                S31: jnp.ndarray, S32_i: jnp.ndarray,
                NQ: int) -> jnp.ndarray:
    """Compute F2 source for b-coefficient recursion at step (i, m).

    F2[a] = sum_b AA[a,b]*BB[b,i] * (S0s + Sq)
    """
    F2 = jnp.zeros(4, dtype=complex)

    for a in range(4):
        total = 0.0 + 0j
        for b0 in range(4):
            # S0s: sum over previous b's convolved with q
            S0s = 0.0 + 0j
            if m >= 2:
                for n in range(1, m):
                    S0s += b[n, b0] * q[m - n, a, b0]

            # Sq
            Sq = 0.0 + 0j
            for n in range(m + 1):
                S1s = S1n_i[n, b0]

                S2s = 0.0 + 0j
                for j in range(max(n - 1, 0)):
                    S2s += b[j + 1, b0] * S1[n, j]

                S3s = 0.0 + 0j
                for k in range(max(2 * n - 1, 0)):
                    S3spart = 0.0 + 0j
                    for j in range((k + 1) // 2):
                        S3spart += b[j + 1, b0] * S31[k, j]
                    S3s += S32_i[n, k, b0] * S3spart

                Sq += q[m - n, a, b0] * (S1s + S2s + S3s)

            total += AA[a, b0] * BB[b0, i] * (S0s + Sq)
        F2 = F2.at[a].set(total)
    return F2


def _solve_b_coefficients(q: jnp.ndarray, AA: jnp.ndarray, BB: jnp.ndarray,
                          B: jnp.ndarray, alfa: jnp.ndarray,
                          scT: list, aux_tables: dict,
                          alfa_tables: dict, NQ: int) -> list:
    """Solve for b_{a|i,n} coefficients via sequential 4x4 linear systems.

    Returns list of 4 arrays b[i] of shape (NQ+1, 4).
    """
    T1 = aux_tables["T1"]
    T2 = aux_tables["T2"]
    T41 = aux_tables["T41"]
    S1 = aux_tables["S1"]
    S31 = aux_tables["S31"]

    b_all = []
    for i in range(4):
        # Initialize: b[0, a] = 1 for all a
        b = jnp.zeros((NQ + 1, 4), dtype=complex)
        b = b.at[0, :].set(1.0)

        T3_i = alfa_tables["T3"][i]
        T5_i = alfa_tables["T5"][i]
        S1n_i = alfa_tables["S1n"][i]
        S32_i = alfa_tables["S32"][i]

        for m in range(1, NQ + 1):
            F1 = _compute_F1(i, m, b, BB, alfa, T1, T2, T3_i, T41, T5_i)
            F2 = _compute_F2(i, m, b, q, AA, BB, S1n_i, S1, S31, S32_i, NQ)

            # Solve scT[i][m] @ x = F1 - F2
            rhs = F1 - F2
            mat = scT[i][m]
            x = jnp.linalg.solve(mat, rhs)
            b = b.at[m, :].set(x)

        b_all.append(b)
    return b_all


def _evaluate_Q_and_pulldown(b_all: list, BB: jnp.ndarray, Mt: jnp.ndarray,
                             Mhat: jnp.ndarray, c: list, sigma: list,
                             uA: jnp.ndarray, g: complex,
                             NQ: int, NI: int, N0: int,
                             lc: int) -> tuple:
    """Evaluate Q_{a|i}(u_k) at large u, then pull down to the cut.

    Returns (Qlower, Qtlower) each of shape (lc, 4).
    """
    II = 1j

    # Precompute TuANIn[i][n, k] = 1 / (uA[k] + i*(NI + 0.5))^{2n}
    # and TuMiMaNI[k, a, i] = (uA[k] + i*(NI + 0.5))^{Mhat[i] - Mt[a]}
    u_shifted = uA + II * (NI + 0.5)
    TuANIn = []
    for i in range(4):
        table = jnp.zeros((NQ + 1, lc), dtype=complex)
        for n in range(NQ + 1):
            table = table.at[n, :].set(1.0 / u_shifted ** (2 * n))
        TuANIn.append(table)

    TuMiMaNI = jnp.zeros((lc, 4, 4), dtype=complex)
    for i in range(4):
        for a in range(4):
            TuMiMaNI = TuMiMaNI.at[:, a, i].set(u_shifted ** (Mhat[i] - Mt[a]))

    # Q[a][i][k] = BB[a,i] * TuMiMaNI[k,a,i] * sum_n b[i][n,a] * TuANIn[i][n,k]
    Q = jnp.zeros((4, 4, lc), dtype=complex)
    for i in range(4):
        for a in range(4):
            q_sum = jnp.zeros(lc, dtype=complex)
            for n in range(NQ + 1):
                q_sum += b_all[i][n, a] * TuANIn[i][n, :]
            Q = Q.at[a, i, :].set(BB[a, i] * TuMiMaNI[:, a, i] * q_sum)

    # Precompute P_a(u_k + i*n) for n=1,...,NI (pull-down)
    # Using short-cut Zhukovsky for |u + in| > 2g
    # PujT[a, n, k] where n=0..NI-1 means shift by (n+1)
    Puj = jnp.zeros((4, NI, lc), dtype=complex)
    for n_shift in range(NI):
        u_imag = (uA + II * (n_shift + 1)) / g
        x_vals = x_of_u_short(u_imag)
        for a in range(4):
            p_val = jnp.zeros(lc, dtype=complex)
            for m in range(N0 + 1):
                p_val += c[a][m] / x_vals ** (2 * m)
            Puj = Puj.at[a, n_shift, :].set(p_val / x_vals ** Mt[a])

    # Pull-down: iterate from n=NI-1 down to 0
    m1_signs = jnp.array([(-1.0) ** (b + 1) for b in range(4)])

    for k in range(lc):
        for i in range(4):
            for n in range(NI - 1, -1, -1):
                Q_old = Q[:, i, k].copy()
                for a in range(4):
                    new_val = Q_old[a]
                    contrib = 0.0 + 0j
                    for b0 in range(4):
                        contrib += m1_signs[b0] * Puj[3 - b0, n, k] * Q_old[b0]
                    new_val = Puj[a, n, k] * contrib + Q_old[a]
                    Q = Q.at[a, i, k].set(new_val)

    # P_a(u_k) = sum_n c[a][n] * x^{Mt[a]+2n}   (C++ convention: PaT = x^Mt * x^{2n})
    # Ptilde:    sum_n c[a][n] * x^{-Mt[a]-2n}  (C++ convention: PtaT = x^{-Mt} * x^{-2n})
    x_cut_rescaled = x_of_u_long(uA / g, 1.0)  # X(u/g) on the cut
    P = jnp.zeros((4, lc), dtype=complex)
    Pt = jnp.zeros((4, lc), dtype=complex)
    for a in range(4):
        xMt = x_cut_rescaled ** Mt[a]
        p_sum = jnp.zeros(lc, dtype=complex)
        pt_sum = jnp.zeros(lc, dtype=complex)
        for n in range(N0 + 1):
            x2n = x_cut_rescaled ** (2 * n)
            p_sum += c[a][n] * x2n       # x^{+2n}
            pt_sum += c[a][n] / x2n      # x^{-2n}
        P = P.at[a, :].set(xMt * p_sum)         # x^{Mt+2n}
        Pt = Pt.at[a, :].set((1.0 / xMt) * pt_sum)  # x^{-Mt-2n}

    # Compute Qlower[k, i] = -sum_a (-1)^{a+1} * P[3-a, k] * Q[a, i, k]
    Qlower = jnp.zeros((lc, 4), dtype=complex)
    Qtlower = jnp.zeros((lc, 4), dtype=complex)
    for k in range(lc):
        for i in range(4):
            val = 0.0 + 0j
            valt = 0.0 + 0j
            for a in range(4):
                val -= m1_signs[a] * P[3 - a, k] * Q[a, i, k]
                valt -= m1_signs[a] * Pt[3 - a, k] * Q[a, i, k]
            Qlower = Qlower.at[k, i].set(val)
            Qtlower = Qtlower.at[k, i].set(valt)

    return Qlower, Qtlower, Q, P, Pt


def _compute_gluing_residual(Q_upper: jnp.ndarray, Qlower: jnp.ndarray,
                             Qtlower: jnp.ndarray,
                             lc: int) -> tuple:
    """Compute gluing constant alpha_Q and deltaP residual.

    Returns (deltaP, deltaPt) each of shape (lc, 4).
    """
    # alpha_Q = Re(mean(Q0/conj(Q2) + Qt0/conj(Qt2) - Q1/conj(Q3) - Qt1/conj(Qt3))) / 4
    alfaQ = 0.0 + 0j
    for k in range(lc):
        alfaQ += (Qlower[k, 0] / jnp.conj(Qlower[k, 2])
                  + Qtlower[k, 0] / jnp.conj(Qtlower[k, 2])
                  - Qlower[k, 1] / jnp.conj(Qlower[k, 3])
                  - Qtlower[k, 1] / jnp.conj(Qtlower[k, 3]))
    alfaQ = alfaQ / (4 * lc)
    alfaQ = jnp.real(alfaQ) + 0j  # take real part

    # deltaP[k, a] from gluing equations
    deltaP = jnp.zeros((lc, 4), dtype=complex)
    deltaPt = jnp.zeros((lc, 4), dtype=complex)

    for a in range(4):
        for k in range(lc):
            dp = (Q_upper[a, 0, k] * (Qlower[k, 3] + jnp.conj(Qlower[k, 1]) / alfaQ)
                  - Q_upper[a, 1, k] * (Qlower[k, 2] - jnp.conj(Qlower[k, 0]) / alfaQ)
                  + Q_upper[a, 2, k] * (Qlower[k, 1] + jnp.conj(Qlower[k, 3]) * alfaQ)
                  - Q_upper[a, 3, k] * (Qlower[k, 0] - jnp.conj(Qlower[k, 2]) * alfaQ))
            deltaP = deltaP.at[k, a].set(dp)

            dpt = (Q_upper[a, 0, k] * (Qtlower[k, 3] + jnp.conj(Qtlower[k, 1]) / alfaQ)
                   - Q_upper[a, 1, k] * (Qtlower[k, 2] - jnp.conj(Qtlower[k, 0]) / alfaQ)
                   + Q_upper[a, 2, k] * (Qtlower[k, 1] + jnp.conj(Qtlower[k, 3]) * alfaQ)
                   - Q_upper[a, 3, k] * (Qtlower[k, 0] - jnp.conj(Qtlower[k, 2]) * alfaQ))
            deltaPt = deltaPt.at[k, a].set(dpt)

    return deltaP, deltaPt, alfaQ


def _fourier_inversion(deltaP: jnp.ndarray, deltaPt: jnp.ndarray,
                       CT: jnp.ndarray, CU: jnp.ndarray,
                       suA: jnp.ndarray, Nas: list,
                       Mtint: jnp.ndarray, Nch: list,
                       N0: int, lc: int, g: complex,
                       kettoLAMBDA: int,
                       x_cut_half: jnp.ndarray) -> jnp.ndarray:
    """Fourier-invert deltaP to get equation residual E.

    Matches QtoEtypeIInewton from C++.
    """
    # Half-integer correction (when kettoLAMBDA is odd)
    if kettoLAMBDA % 2 != 0:
        for a in range(4):
            deltaP = deltaP.at[:, a].mul(1.0 / x_cut_half)
            deltaPt = deltaPt.at[:, a].mul(x_cut_half)

    # Symmetric and antisymmetric combinations
    fS = (deltaP + deltaPt) / 2  # (lc, 4)
    fA = jnp.zeros((lc, 4), dtype=complex)
    for a in range(4):
        fA = fA.at[:, a].set((deltaP[:, a] - deltaPt[:, a]) / (2 * suA))

    # Chebyshev transform: cS[n, a] = (1/lc) * sum_k fS[k, a] * CT[lc-1-k, n]
    # Note the reversed index on CT
    CT_rev = CT[::-1, :]  # reverse rows
    cS = jnp.zeros((lc, 4), dtype=complex)
    for a in range(4):
        cS = cS.at[:, a].set(CT_rev.T @ fS[:, a] / lc)

    # cA[n, a] = (2ig/lc) * sum_k fA[k, a] * CU[lc-1-k, n-1]  for n >= 1
    CU_rev = CU[::-1, :]
    cA = jnp.zeros((lc, 4), dtype=complex)
    for a in range(4):
        # n=0: cA[0, a] = 0
        # n=1..lc-1: CU index is n-1
        cA_inner = CU_rev.T @ fA[:, a] / lc
        cA = cA.at[1:, a].set(cA_inner[:lc - 1] * 2j * g)

    # Assemble E vector
    Mtint_np = [int(Mtint[i]) for i in range(4)]
    dimV = 1 + N0 + Nch[1] + Nch[2] + Nch[3]
    E = jnp.zeros(dimV, dtype=complex)

    # a=0: n=0..N0
    a = 0
    for n in range(N0 + 1):
        idx = abs(-Nas[a][0] + 2 * n)
        if 2 * n >= Nas[a][0]:
            E = E.at[n].set(cS[idx, a] + cA[idx, a])
        else:
            E = E.at[n].set(cS[idx, a] - cA[idx, a])

    # a=1: n=1..N0, skip gauge-fixed
    offset = N0 + 1
    a = 1
    k_skip = 0
    for n in range(1, N0 + 1):
        if 2 * n == Mtint_np[0] - Mtint_np[1]:
            k_skip += 1
            continue
        idx = abs(-Nas[a][0] + 2 * n)
        if 2 * n >= Nas[a][0]:
            E = E.at[offset + n - 1 - k_skip].set(cS[idx, a] + cA[idx, a])
        else:
            E = E.at[offset + n - 1 - k_skip].set(cS[idx, a] - cA[idx, a])

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
        if 2 * n >= Nas[a][0]:
            E = E.at[offset + n - 1 - k_skip].set(cS[idx, a] + cA[idx, a])
        else:
            E = E.at[offset + n - 1 - k_skip].set(cS[idx, a] - cA[idx, a])

    # a=3
    offset = N0 + 1 + Nch[1] + Nch[2]
    a = 3
    k_skip = 0
    for n in range(1, N0 + 1):
        if 2 * n == Mtint_np[0] - Mtint_np[3]:
            k_skip += 1
            continue
        idx = abs(-Nas[a][0] + 2 * n)
        if 2 * n >= Nas[a][0]:
            E = E.at[offset + n - 1 - k_skip].set(cS[idx, a] + cA[idx, a])
        else:
            E = E.at[offset + n - 1 - k_skip].set(cS[idx, a] - cA[idx, a])

    return E


def params_to_V(params: jnp.ndarray, gauge_indices: list, N0: int) -> jnp.ndarray:
    """Convert raw params (1+4*N0) to free variable vector V (dimV).

    Removes gauge-fixed entries from the parameter vector.
    """
    # params = [Delta, c[0][1],...,c[0][N0], c[1][1],..., c[3][N0]]
    # gauge_indices = [(a, n)] pairs where c[a][n] = 0
    # The corresponding param index is 1 + a*N0 + (n-1) = a*N0 + n
    skip_indices = set()
    for a, n in gauge_indices:
        skip_indices.add(a * N0 + n)  # index within the c-block, offset by 1 for Delta

    V = []
    V.append(params[0])  # Delta is always included
    for idx in range(1, len(params)):
        if (idx - 1) not in skip_indices:
            V.append(params[idx])
    return jnp.array(V)


def V_to_params(V: jnp.ndarray, gauge_indices: list, N0: int) -> jnp.ndarray:
    """Convert free variable vector V (dimV) back to raw params (1+4*N0).

    Inserts zeros at gauge-fixed positions.
    """
    skip_indices = set()
    for a, n in gauge_indices:
        skip_indices.add(a * N0 + n)

    params = [V[0]]  # Delta
    v_idx = 1
    for idx in range(4 * N0):
        if idx in skip_indices:
            params.append(0.0 + 0j)
        else:
            params.append(V[v_idx])
            v_idx += 1
    return jnp.array(params)


def forward_map_typeI(params: jnp.ndarray, qn: QuantumNumbers,
                      g: float, config: SolverConfig) -> jnp.ndarray:
    """Complete forward map for TypeI states.

    params: flat array [Delta, c[0,1],...,c[0,N0], c[1,1],...,c[3,N0]]
            where c[a][n] are in the INTERNAL (denormalized) convention.
            Length = 1 + 4*N0 (includes gauge-fixed entries as zeros).
    qn: quantum numbers
    g: coupling constant
    config: solver hyperparameters

    Returns: residual vector E of dimension dimV.
    """
    N0 = config.N0
    NQ = config.NQ
    NI = config.NI
    lc = config.nPoints

    # Quantum number derived quantities
    Mtint = compute_Mtint(qn)
    kettoLAMBDA = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kettoLAMBDA)
    twiceMt = 2 * Mtint + kettoLAMBDA
    Mhat0 = compute_Mhat0(qn, kettoLAMBDA)
    Nas = compute_Nas(Mtint, kettoLAMBDA)
    gauge_info = compute_gauge_info(Mtint, N0)
    Nch = gauge_info["Nch"]

    # Ensure lc is large enough
    lc = ensure_min_lc(lc, Nas, N0)

    # Unpack params
    Delta = params[0]
    c_n1_to_N0 = params[1:].reshape(4, N0)

    # Delta-dependent quantities
    Mhat = compute_Mhat(Mhat0, Delta)
    A, Af, AA = compute_A(Mt, Mhat)
    B = compute_B(Mt, Mhat)
    BB = compute_BB(A, B, Mt, Mhat)
    alfa = compute_alfa(Mt, Mhat)

    # Build full c arrays including c[a][0]
    c = _build_c_full(Delta, c_n1_to_N0, A, Mt, g, Mtint,
                      gauge_info["gauge_indices"])

    # Chebyshev grid
    uA = chebyshev_grid(g, lc)
    CT = chebyshev_CT(lc)
    CU = chebyshev_CU(CT, lc)
    suA = sqrt_weight(g, uA)

    # Sigma tables
    sigma = build_sigma_table(twiceMt, N0, NQ, g)

    # 1/u expansion coefficients
    ksub = _compute_ksub(c, sigma, NQ, N0)

    # q-array
    q = _compute_q_array(ksub, AA, NQ)

    # scT matrices
    scT = _build_scT_matrices(AA, BB, alfa, NQ)

    # Auxiliary tables
    aux_tables = _build_auxiliary_tables(alfa, NQ)
    alfa_tables = _build_alfa_tables(alfa, NQ, aux_tables["m1p4k"])

    # Solve for b-coefficients
    b_all = _solve_b_coefficients(q, AA, BB, B, alfa, scT,
                                  aux_tables, alfa_tables, NQ)

    # Evaluate Q and pull down
    Qlower, Qtlower, Q_upper, P, Pt = _evaluate_Q_and_pulldown(
        b_all, BB, Mt, Mhat, c, sigma, uA, g, NQ, NI, N0, lc
    )

    # Gluing residual
    deltaP, deltaPt, alfaQ = _compute_gluing_residual(
        Q_upper, Qlower, Qtlower, lc
    )

    # Half-integer correction factor
    x_cut_rescaled = x_of_u_long(uA / g, 1.0)
    x_cut_half = jnp.sqrt(x_cut_rescaled)

    # Fourier inversion to get equation residual
    E = _fourier_inversion(deltaP, deltaPt, CT, CU, suA, Nas,
                           Mtint, Nch, N0, lc, g, kettoLAMBDA,
                           x_cut_half)

    return E
