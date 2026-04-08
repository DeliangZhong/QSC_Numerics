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
                  Mtint: jnp.ndarray, gauge_indices: list) -> jnp.ndarray:
    """Build full c[a][0..N0] arrays including c[a][0] from A and gauge fixes.

    c[a][0] = A[a] / g^{Mt[a]}

    For TypeI, c[0] and c[2] are imaginary, c[1] and c[3] are real.
    The C++ code stores: c[0][n] = i * c0_input[n], c[2][n] = i * c2_input[n].

    Returns array of shape (4, N0+1).
    """
    N0 = c_n1_to_N0.shape[1]
    # c[a][0] = A[a] / g^Mt[a]
    c0 = A / g**Mt  # shape (4,)
    # Concatenate c0 column with c_n1_to_N0
    c = jnp.concatenate([c0[:, None], c_n1_to_N0], axis=1)  # (4, N0+1)

    # Apply gauge fixing: set specified coefficients to zero
    for a_gauge, n_gauge in gauge_indices:
        c = c.at[a_gauge, n_gauge].set(0.0 + 0j)

    return c


def _compute_ksub(c: jnp.ndarray, sigma: jnp.ndarray,
                  NQ: int, N0: int) -> jnp.ndarray:
    """Compute 1/u expansion coefficients of P_a(u).

    ksub[a, n] = sum_{r=0}^{min(n,N0)} c[a, r] * sigma[a, r, n]

    c has shape (4, N0+1), sigma has shape (4, N0+1, NQ+1).
    Returns array of shape (4, NQ+1).

    Vectorized: ksub = einsum over the r dimension, with sigma already
    containing zeros for r > n (built that way by fsigma returning 0 when
    the sum range is empty).
    """
    # sigma[a, r, n] is zero when r > n, so a simple contraction works.
    # ksub[a, n] = sum_r c[a, r] * sigma[a, r, n]
    return jnp.einsum('ar,arn->an', c, sigma)


def _compute_q_array(ksub: jnp.ndarray, AA: jnp.ndarray,
                     NQ: int) -> jnp.ndarray:
    """Compute q-array via convolution.

    q[n, a, b] = sum_m ksub[a,m]*(-1)^{b+1}*ksub[3-b,n-m] / AA[a,b]

    ksub has shape (4, NQ+1). Returns array of shape (NQ+1, 4, 4).

    Vectorized using the fact that sum_m f[m]*g[n-m] is a convolution.
    We compute the outer convolution for all (a,b) pairs.
    """
    m1_signs = jnp.array([(-1.0) ** (b + 1) for b in range(4)])  # (4,)

    # For each (a, b) pair, we need:
    #   conv[n] = sum_{m=0}^{n} ksub[a, m] * ksub[3-b, n-m]
    # This is a polynomial multiplication (truncated to NQ+1 terms).

    # ksub_rev[b] = ksub[3-b]
    ksub_rev = ksub[jnp.array([3, 2, 1, 0]), :]  # (4, NQ+1)

    # Compute convolution via outer product and cumulative sums.
    # conv[a, b, n] = sum_{m=0}^{n} ksub[a, m] * ksub_rev[b, n-m]
    # Build the Toeplitz-like product: prod[a, b, m, n-m] = ksub[a,m]*ksub_rev[b,n-m]
    # Then sum over m where m + (n-m) = n, i.e., second index = n - m.

    # Use jnp.convolve logic: for each (a,b), the convolution of two
    # length-(NQ+1) sequences gives length 2*NQ+1, and we take first NQ+1.

    # Vectorized approach: build a (4, 4, NQ+1) result using vmap or explicit.
    # ksub: (4, NQ+1), ksub_rev: (4, NQ+1)
    # We want conv[a, b, n] for n=0..NQ

    # Method: Toeplitz matrix multiply.
    # Build lower-triangular Toeplitz matrix T[n, m] = ksub_rev[b, n-m] for n>=m
    # Then conv[a, b] = T[b] @ ksub[a]
    # T[b][n, m] = ksub_rev[b, n-m] if n >= m, else 0

    nq1 = NQ + 1
    # Build index array: idx[n, m] = n - m
    n_idx = jnp.arange(nq1)[:, None]  # (NQ+1, 1)
    m_idx = jnp.arange(nq1)[None, :]  # (1, NQ+1)
    diff = n_idx - m_idx  # (NQ+1, NQ+1)
    mask = diff >= 0  # lower triangular

    # For each b, build Toeplitz matrix and multiply
    # T_b[n, m] = ksub_rev[b, diff[n,m]] * mask[n,m]
    # Use advanced indexing: ksub_rev[b, diff_clipped] * mask
    diff_clipped = jnp.where(mask, diff, 0)  # safe indexing

    # T_all[b, n, m] = ksub_rev[b, diff_clipped[n,m]] * mask[n,m]
    T_all = ksub_rev[:, diff_clipped] * mask[None, :, :]  # (4, NQ+1, NQ+1)

    # conv[a, b, n] = sum_m T_all[b, n, m] * ksub[a, m]
    # = einsum('bnm,am->abn')
    conv = jnp.einsum('bnm,am->abn', T_all, ksub)  # (4, 4, NQ+1)

    # q[n, a, b] = conv[a, b, n] * m1_signs[b] / AA[a, b]
    # conv is (4, 4, NQ+1), transpose to (NQ+1, 4, 4)
    q = conv.transpose(2, 0, 1) * m1_signs[None, None, :] / AA[None, :, :]

    return q


def _build_scT_matrices(AA: jnp.ndarray, BB: jnp.ndarray, alfa: jnp.ndarray,
                        NQ: int) -> jnp.ndarray:
    """Build scT[i, m] matrices (4x4) for each i=0..3, m=0..NQ.

    From totalscTmaker2LRi in C++ (note: the C++ 'B' parameter is actually BB):
    scT[m][a][b0] = AA[a][b0]*BB[b0][i]  for a != b0
    scT[m][a][a]  = AA[a][a]*BB[a][i] - i*BB[a][i]*(2m - alfa[a][i])
    scT[0] = 0.

    Returns array of shape (4, NQ+1, 4, 4).
    """
    II = 1j

    # Off-diagonal part (independent of m): AA[a, b0] * BB[b0, i]
    # For each i: off_diag[a, b0] = AA[a, b0] * BB[b0, i]
    # BB has shape (4, 4), AA has shape (4, 4)

    # off_diag[i, a, b0] = AA[a, b0] * BB[b0, i]
    off_diag = AA[:, :, None] * BB[None, :, :]  # (4, 4, 4) -> AA[a,b0]*BB[b0,i]
    # Rearrange: we want [i, a, b0]
    # off_diag_full[a, b0, i] = AA[a, b0] * BB[b0, i]
    off_diag_full = AA[:, :, None] * BB[None, :, :]  # (4_a, 4_b0, 4_i)

    # Diagonal correction: for a == b0:
    # scT[m, a, a] = AA[a,a]*BB[a,i] - i*BB[a,i]*(2m - alfa[a,i])
    #              = off_diag[a, a, i] - i*BB[a,i]*(2m - alfa[a,i])
    # The extra term: -i * BB[a, i] * (2m - alfa[a, i])

    m_vals = jnp.arange(NQ + 1)  # (NQ+1,)

    # Build full array: scT[i, m, a, b0]
    # Start with off_diag broadcast over m:
    # scT_base[i, m, a, b0] = off_diag_full[a, b0, i] for all m
    scT = jnp.broadcast_to(
        off_diag_full.transpose(2, 0, 1)[None, :, :, :],  # (1, 4_i, 4_a, 4_b0) -- wrong
        (NQ + 1, 4, 4, 4)
    ).copy()
    # Let me redo this more carefully.
    # We want scT[i, m, a, b0].
    # Base value = AA[a, b0] * BB[b0, i], same for all m.
    # off_diag_full[a, b0, i] = AA[a, b0] * BB[b0, i]
    # So scT_base[i, m, a, b0] = off_diag_full[a, b0, i]

    # For diagonal (a == b0), add: -i * BB[a, i] * (2*m - alfa[a, i])
    # diag_corr[i, m, a] = -i * BB[a, i] * (2*m - alfa[a, i])

    # BB[a, i]: shape (4, 4)
    # alfa[a, i]: shape (4, 4)
    # 2*m: shape (NQ+1,)

    # diag_corr[a, i, m] = -i * BB[a, i] * (2*m - alfa[a, i])
    diag_corr = -II * BB[:, :, None] * (2 * m_vals[None, None, :] - alfa[:, :, None])
    # shape: (4_a, 4_i, NQ+1_m)

    # Build scT[i, m, a, b0]:
    # Start from base
    scT_result = jnp.zeros((4, NQ + 1, 4, 4), dtype=complex)

    # Fill base: for all (i, m, a, b0) = off_diag_full[a, b0, i]
    scT_result = jnp.broadcast_to(
        off_diag_full[:, :, :, None].transpose(2, 3, 0, 1),  # (4_i, 1, 4_a, 4_b0)
        (4, NQ + 1, 4, 4)
    ) + 0  # force copy

    # Add diagonal correction: only when a == b0
    # scT_result[i, m, a, a] += diag_corr[a, i, m]
    a_idx = jnp.arange(4)
    # diag_corr[a, i, m] -> we need to add to scT_result[i, m, a, a]
    # Transpose diag_corr to (i, m, a): diag_corr.transpose(1, 2, 0)
    diag_corr_reordered = diag_corr.transpose(1, 2, 0)  # (4_i, NQ+1_m, 4_a)

    # Add to diagonal: scT_result[i, m, a, a] += diag_corr_reordered[i, m, a]
    scT_result = scT_result.at[:, :, a_idx, a_idx].add(diag_corr_reordered)

    # Zero out m=0
    scT_result = scT_result.at[:, 0, :, :].set(0.0)

    return scT_result


def _build_auxiliary_tables(alfa: jnp.ndarray, NQ: int) -> dict:
    """Build T1, T2, T41, S1, S31 tables.

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


def _cbinomial_table(z: jnp.ndarray, max_n: int) -> jnp.ndarray:
    """Compute C(z, n) for n=0..max_n-1 for each element in z.

    z: array of shape (...), complex
    Returns: array of shape (max_n, ...) where result[n, ...] = C(z[...], n).

    Uses the recurrence C(z, 0) = 1, C(z, n) = C(z, n-1) * (z - n + 1) / n.
    Fully JAX-traceable.
    """
    # result[0] = 1
    result = [jnp.ones_like(z, dtype=complex)]
    prev = jnp.ones_like(z, dtype=complex)
    for k in range(1, max_n):
        prev = prev * (z - k + 1) / k
        result.append(prev)
    return jnp.stack(result, axis=0)  # (max_n, ...)


def _build_alfa_tables(alfa: jnp.ndarray, NQ: int,
                       m1p4k: jnp.ndarray) -> dict:
    """Build alfa-dependent auxiliary tables for each i.

    alfaais[i][m, a] = C(alfa[a][i], m)   (binomial)
    T3[i][l, a] = alfaais[2l+1, a] * m1p4k[l]
    T5[i][m, l, a] = alfaais[2l-m-1, a] * m1p4k[l]   for m <= 2l-3
    S1n[i][n, a] = alfaais[2n, a] * m1p4k[n]
    S32[i][n, k, a] = alfaais[2n-k-1, a] * m1p4k[n]   for k <= 2n-2

    Vectorized: uses _cbinomial_table to compute all binomial coefficients
    at once for each i.
    """
    max_m = 2 * NQ + 2
    kNQm1 = max(2 * NQ - 1, 1)
    mm_dim = max(2 * NQ - 2, 1)

    result = {"alfaais": [], "T3": [], "T5": [], "S1n": [], "S32": []}

    for i in range(4):
        # alfaais[m, a] for m=0..2*NQ+1
        # alfa[:, i] has shape (4,) -- one value per a
        # _cbinomial_table returns (max_m, 4)
        alfaais = _cbinomial_table(alfa[:, i], max_m)  # (max_m, 4)

        # T3[l, a] = alfaais[2*l+1, a] * m1p4k[l]
        l_vals = jnp.arange(NQ + 1)
        T3_idx = 2 * l_vals + 1  # (NQ+1,)
        T3 = alfaais[T3_idx, :] * m1p4k[l_vals, None]  # (NQ+1, 4)

        # T5[m, l, a] = alfaais[2*l - m - 1, a] * m1p4k[l]  for m <= 2*l - 3
        # Build full array using index arrays
        m_grid = jnp.arange(mm_dim)[:, None]  # (mm_dim, 1)
        l_grid = jnp.arange(NQ + 1)[None, :]  # (1, NQ+1)
        T5_idx_raw = 2 * l_grid - m_grid - 1  # (mm_dim, NQ+1)
        T5_valid = (m_grid <= 2 * l_grid - 3) & (T5_idx_raw >= 0) & (T5_idx_raw < max_m)
        T5_idx = jnp.clip(T5_idx_raw, 0, max_m - 1)
        # alfaais[T5_idx, :]: shape (mm_dim, NQ+1, 4)
        T5 = alfaais[T5_idx, :] * m1p4k[l_grid, None]  # broadcast
        T5 = jnp.where(T5_valid[:, :, None], T5, 0.0)  # (mm_dim, NQ+1, 4)

        # S1n[n, a] = alfaais[2*n, a] * m1p4k[n]
        n_vals = jnp.arange(NQ + 1)
        S1n_idx = 2 * n_vals  # (NQ+1,)
        S1n = alfaais[S1n_idx, :] * m1p4k[n_vals, None]  # (NQ+1, 4)

        # S32[n, k, a] = alfaais[2*n - k - 1, a] * m1p4k[n]  for k <= 2*n - 2
        n_grid = jnp.arange(NQ + 1)[:, None]  # (NQ+1, 1)
        k_grid = jnp.arange(kNQm1)[None, :]   # (1, kNQm1)
        S32_idx_raw = 2 * n_grid - k_grid - 1  # (NQ+1, kNQm1)
        S32_valid = (k_grid <= 2 * n_grid - 2) & (S32_idx_raw >= 0) & (S32_idx_raw < max_m)
        S32_idx = jnp.clip(S32_idx_raw, 0, max_m - 1)
        S32 = alfaais[S32_idx, :] * m1p4k[n_grid, None]  # (NQ+1, kNQm1, 4)
        S32 = jnp.where(S32_valid[:, :, None], S32, 0.0)

        result["alfaais"].append(alfaais)
        result["T3"].append(T3)
        result["T5"].append(T5)
        result["S1n"].append(S1n)
        result["S32"].append(S32)

    return result


def _compute_F1_traceable(i_idx: int, m: jnp.ndarray, b: jnp.ndarray,
                          BB: jnp.ndarray, alfa: jnp.ndarray,
                          T1: jnp.ndarray, T2: jnp.ndarray,
                          T3_i: jnp.ndarray, T41: jnp.ndarray,
                          T5_i: jnp.ndarray, NQ: int) -> jnp.ndarray:
    """Compute F1 source for b-coefficient recursion at step (i, m).

    F1[a] = -i * BB[a,i] * (T1s + T2s + T3s + T4s)

    Fully JAX-traceable (no Python conditionals on m).
    Uses masks to handle variable-length summations.
    b has shape (NQ+1, 4). m is a traced scalar.
    """
    mI = -1j
    NQ1 = NQ + 1

    # Mask for k indices: k_mask[k] = 1 if k < m-1, else 0
    # b[k+1, a] * T1[m, k] for k=0..m-2
    k_idx = jnp.arange(T1.shape[1])  # (K,)
    k_mask = (k_idx < (m - 1)).astype(complex)  # (K,)

    # T1s[a] = sum_k b[k+1, a] * T1[m, k] * k_mask[k]
    T1_m = T1[m, :]  # (K,), T1 is indexed by m (traced)
    T1s = (b[1:T1.shape[1] + 1, :].T) @ (T1_m * k_mask)  # (4,)

    # T2s[a] = alfa[a, i] * sum_k b[k+1, a] * T2[m, k] * k_mask[k]
    T2_m = T2[m, :]  # (K,)
    T2s = (b[1:T2.shape[1] + 1, :].T) @ (T2_m * k_mask) * alfa[:, i_idx]  # (4,)

    # T3s[a] = b[0, a] * T3_i[m, a]
    T3s = b[0, :] * T3_i[m, :]  # (4,)

    # T4s: use full arrays (T5_i and T41 already have zeros outside valid ranges)
    # T4spart[mm, a] = sum_k T41[mm, k] * b[k+1, a]
    n_b = min(NQ1 - 1, T41.shape[1])
    T4spart = T41[:, :n_b] @ b[1:n_b + 1, :]  # (mm_dim, 4)

    # mm_mask[mm] = 1 if mm < 2*m - 2, else 0
    mm_idx = jnp.arange(T41.shape[0])
    mm_mask = (mm_idx < (2 * m - 2)).astype(complex)  # (mm_dim,)

    # T4s[a] = sum_mm T5_i[mm, m, a] * T4spart[mm, a] * mm_mask[mm]
    T5_m = T5_i[:T41.shape[0], m, :]  # (mm_dim, 4)
    T4s = jnp.sum(T5_m * T4spart * mm_mask[:, None], axis=0)  # (4,)

    F1 = mI * BB[:, i_idx] * (T1s + T2s + T3s + T4s)
    return F1


def _compute_F2_traceable(i_idx: int, m: jnp.ndarray, b: jnp.ndarray,
                          q: jnp.ndarray, AA: jnp.ndarray, BB: jnp.ndarray,
                          S1n_i: jnp.ndarray, S1: jnp.ndarray,
                          S31: jnp.ndarray, S32_i: jnp.ndarray,
                          NQ: int) -> jnp.ndarray:
    """Compute F2 source for b-coefficient recursion at step (i, m).

    F2[a] = sum_b AA[a,b]*BB[b,i] * (S0s + Sq)

    Fully JAX-traceable (no Python conditionals on m).
    b has shape (NQ+1, 4), q has shape (NQ+1, 4, 4). m is a traced scalar.
    """
    NQ1 = NQ + 1

    # coeff[a, b0] = AA[a, b0] * BB[b0, i]
    coeff = AA * BB[:, i_idx][None, :]  # (4, 4)

    # --- S0s[a, b0] = sum_{n=1}^{m-1} b[n, b0] * q[m-n, a, b0] ---
    # Use full range n=1..NQ with mask n < m
    n_idx_s0 = jnp.arange(1, NQ1)  # (NQ,) values 1..NQ
    n_mask_s0 = (n_idx_s0 < m).astype(complex)  # (NQ,)

    # q_at_mn[n, a, b0] = q[m - n, a, b0] for n=1..NQ
    # m - n_idx_s0: traced indexing. Use gather:
    mn_idx = m - n_idx_s0  # (NQ,) traced
    # Clip to valid range for safe indexing (mask handles correctness)
    mn_idx_safe = jnp.clip(mn_idx, 0, NQ)
    q_at_mn = q[mn_idx_safe, :, :]  # (NQ, 4, 4)

    # b[n, b0] for n=1..NQ
    b_s0 = b[1:NQ1, :]  # (NQ, 4)

    # S0s[a, b0] = sum_n b_s0[n, b0] * q_at_mn[n, a, b0] * n_mask_s0[n]
    S0s = jnp.einsum('nb,nab,n->ab', b_s0, q_at_mn, n_mask_s0)  # (4, 4)

    # --- Sq[a, b0] = sum_{n=0}^{m} q[m-n, a, b0] * S_total[n, b0] ---
    # S_total[n, b0] = S1s + S2s + S3s  (precomputed for all n, masked)

    # S1s[n, b0] = S1n_i[n, b0]  -- no masking needed, used with n_mask below
    S1s_all = S1n_i  # (NQ+1, 4)

    # S2s[n, b0] = sum_j b[j+1, b0] * S1[n, j]
    # S1 already has zeros for j >= n-1, so full matmul is fine
    n_b_s1 = min(NQ1 - 1, S1.shape[1])
    S2s_all = S1[:, :n_b_s1] @ b[1:n_b_s1 + 1, :]  # (NQ+1, 4)

    # S3s[n, b0] = sum_k S32_i[n, k, b0] * T4spart_s3[k, b0]
    # T4spart_s3[k, b0] = sum_j S31[k, j] * b[j+1, b0]
    n_b_s31 = min(NQ1 - 1, S31.shape[1])
    T4spart_s3 = S31[:, :n_b_s31] @ b[1:n_b_s31 + 1, :]  # (kmax, 4)

    k_max = min(S32_i.shape[1], T4spart_s3.shape[0])
    S3s_all = jnp.einsum('nkb,kb->nb', S32_i[:, :k_max, :],
                         T4spart_s3[:k_max, :])  # (NQ+1, 4)

    S_total = S1s_all + S2s_all + S3s_all  # (NQ+1, 4)

    # q[m-n, a, b0] for n=0..m, masked by n <= m
    n_idx_sq = jnp.arange(NQ1)  # (NQ+1,)
    n_mask_sq = (n_idx_sq <= m).astype(complex)  # (NQ+1,)

    mn_idx_sq = m - n_idx_sq  # (NQ+1,) traced
    mn_idx_sq_safe = jnp.clip(mn_idx_sq, 0, NQ)
    q_at_mn_sq = q[mn_idx_sq_safe, :, :]  # (NQ+1, 4, 4)

    # Sq[a, b0] = sum_n q_at_mn_sq[n, a, b0] * S_total[n, b0] * n_mask_sq[n]
    Sq = jnp.einsum('nab,nb,n->ab', q_at_mn_sq, S_total, n_mask_sq)  # (4, 4)

    # F2[a] = sum_b coeff[a, b0] * (S0s[a, b0] + Sq[a, b0])
    F2 = jnp.sum(coeff * (S0s + Sq), axis=1)  # (4,)

    return F2


def _solve_b_coefficients_one_i(i_idx: int, q: jnp.ndarray, AA: jnp.ndarray,
                                BB: jnp.ndarray, alfa: jnp.ndarray,
                                scT_i: jnp.ndarray, T1: jnp.ndarray,
                                T2: jnp.ndarray, T41: jnp.ndarray,
                                S1: jnp.ndarray, S31: jnp.ndarray,
                                T3_i: jnp.ndarray, T5_i: jnp.ndarray,
                                S1n_i: jnp.ndarray, S32_i: jnp.ndarray,
                                NQ: int) -> jnp.ndarray:
    """Solve for b_{a|i,n} for a single i, using jax.lax.scan.

    Returns b of shape (NQ+1, 4).
    """

    def scan_body(b_prev: jnp.ndarray, m: jnp.ndarray) -> tuple:
        """One step of the b-coefficient recursion."""
        F1 = _compute_F1_traceable(i_idx, m, b_prev, BB, alfa, T1, T2,
                                   T3_i, T41, T5_i, NQ)
        F2 = _compute_F2_traceable(i_idx, m, b_prev, q, AA, BB,
                                   S1n_i, S1, S31, S32_i, NQ)
        rhs = F1 - F2
        mat = scT_i[m]
        x = jnp.linalg.solve(mat, rhs)
        b_new = b_prev.at[m, :].set(x)
        return b_new, x

    b_init = jnp.zeros((NQ + 1, 4), dtype=complex)
    b_init = b_init.at[0, :].set(1.0)

    m_range = jnp.arange(1, NQ + 1)
    b_final, _ = jax.lax.scan(scan_body, b_init, m_range)
    return b_final


def _solve_b_coefficients(q: jnp.ndarray, AA: jnp.ndarray, BB: jnp.ndarray,
                          B: jnp.ndarray, alfa: jnp.ndarray,
                          scT: jnp.ndarray, aux_tables: dict,
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
        T3_i = alfa_tables["T3"][i]
        T5_i = alfa_tables["T5"][i]
        S1n_i = alfa_tables["S1n"][i]
        S32_i = alfa_tables["S32"][i]

        b = _solve_b_coefficients_one_i(
            i, q, AA, BB, alfa, scT[i], T1, T2, T41, S1, S31,
            T3_i, T5_i, S1n_i, S32_i, NQ
        )
        b_all.append(b)
    return b_all


def _evaluate_Q_and_pulldown(b_all: list, BB: jnp.ndarray, Mt: jnp.ndarray,
                             Mhat: jnp.ndarray, c: jnp.ndarray,
                             sigma: jnp.ndarray,
                             uA: jnp.ndarray, g: complex,
                             NQ: int, NI: int, N0: int,
                             lc: int) -> tuple:
    """Evaluate Q_{a|i}(u_k) at large u, then pull down to the cut.

    Returns (Qlower, Qtlower) each of shape (lc, 4).
    """
    II = 1j

    # Precompute u_shifted = uA + i*(NI + 0.5)
    u_shifted = uA + II * (NI + 0.5)  # (lc,)

    # TuANIn[n, k] = 1 / u_shifted[k]^{2n}
    # All 4 i-values share the same u_shifted, so TuANIn is the same for all i.
    n_vals = jnp.arange(NQ + 1)  # (NQ+1,)
    # u_shifted^{2n}: (NQ+1, lc)
    u_inv = 1.0 / u_shifted  # (lc,)
    u_inv_sq = u_inv ** 2  # (lc,)
    # TuANIn[n, k] = u_inv_sq[k]^n
    # Build via: u_inv_sq[None, :] ** n_vals[:, None]
    TuANIn = u_inv_sq[None, :] ** n_vals[:, None]  # (NQ+1, lc)

    # TuMiMaNI[a, i] = u_shifted^{Mhat[i] - Mt[a]}
    # exponents[a, i] = Mhat[i] - Mt[a]
    exponents = Mhat[None, :] - Mt[:, None]  # (4_a, 4_i)
    # TuMiMaNI[k, a, i] = u_shifted[k]^exponents[a, i]
    TuMiMaNI = u_shifted[:, None, None] ** exponents[None, :, :]  # (lc, 4, 4)

    # Q_upper[a, i, k] = BB[a,i] * TuMiMaNI[k,a,i] * sum_n b[i][n,a] * TuANIn[n,k]
    # Stack b_all into (4_i, NQ+1, 4_a)
    b_stack = jnp.stack(b_all, axis=0)  # (4, NQ+1, 4)

    # sum_n b[i][n,a] * TuANIn[n,k] = einsum('ina,nk->iak', ... but careful)
    # b_stack[i, n, a], TuANIn[n, k] -> q_sum[i, a, k] = sum_n b_stack[i,n,a] * TuANIn[n,k]
    q_sum = jnp.einsum('ina,nk->iak', b_stack, TuANIn)  # (4_i, 4_a, lc)

    # Q_upper[a, i, k] = BB[a,i] * TuMiMaNI[k,a,i] * q_sum[i,a,k]
    # Rearrange: Q_upper[a, i, k]
    # TuMiMaNI[k, a, i] -> transpose to (a, i, k)
    TuMiMaNI_t = TuMiMaNI.transpose(1, 2, 0)  # (4_a, 4_i, lc)
    # q_sum[i, a, k] -> transpose to (a, i, k)
    q_sum_t = q_sum.transpose(1, 0, 2)  # (4_a, 4_i, lc)

    Q_upper = BB[:, :, None] * TuMiMaNI_t * q_sum_t  # (4_a, 4_i, lc)

    # Precompute P_a(u_k + i*n) for n=1,...,NI (pull-down)
    # Using short-cut Zhukovsky for |u + in| > 2g
    # Puj[a, n_shift, k] where n_shift=0..NI-1 means shift by (n_shift+1)

    # c has shape (4, N0+1)
    # For each n_shift: u_imag = (uA + i*(n_shift+1)) / g
    # x_vals = x_of_u_short(u_imag)  -- shape (lc,)
    # p_val[a, k] = sum_m c[a, m] / x_vals[k]^{2m} = c[a] @ (1/x_vals^{2m}) for m=0..N0
    # Then Puj[a, n_shift, k] = p_val / x_vals^{Mt[a]}

    # Precompute powers of 1/x for all shifts
    n_shifts = jnp.arange(NI)  # (NI,)
    u_imag_all = (uA[None, :] + II * (n_shifts[:, None] + 1)) / g  # (NI, lc)
    x_vals_all = x_of_u_short(u_imag_all)  # (NI, lc)

    # 1/x^{2m} for m=0..N0: shape (NI, lc, N0+1)
    x_inv = 1.0 / x_vals_all  # (NI, lc)
    x_inv_sq = x_inv ** 2  # (NI, lc)
    m_powers = jnp.arange(N0 + 1)  # (N0+1,)
    # x_inv_sq_pow[n_shift, k, m] = x_inv_sq[n_shift, k]^m
    x_inv_sq_pow = x_inv_sq[:, :, None] ** m_powers[None, None, :]  # (NI, lc, N0+1)

    # p_val[n_shift, a, k] = sum_m c[a, m] * x_inv_sq_pow[n_shift, k, m]
    # = einsum('am, nkm -> nak')
    p_val = jnp.einsum('am,nkm->nak', c, x_inv_sq_pow)  # (NI, 4_a, lc)

    # x_vals^{Mt[a]}: shape (NI, lc, 4_a) -- but Mt can be half-integer
    x_Mt = x_vals_all[:, :, None] ** Mt[None, None, :]  # (NI, lc, 4_a)

    # Puj[a, n_shift, k] = p_val[n_shift, a, k] / x_Mt[n_shift, k, a]
    Puj = p_val / x_Mt.transpose(0, 2, 1)  # (NI, 4_a, lc)
    # Reorder to (4_a, NI, lc)
    Puj = Puj.transpose(1, 0, 2)  # (4_a, NI, lc)

    # Pull-down: iterate from n=NI-1 down to 0
    # For each (k, i), and for each n:
    #   Q_new[a] = P[a,n,k] * sum_b (-1)^{b+1} * P[3-b,n,k] * Q_old[b] + Q_old[a]
    # This is: Q_new = outer(P, sum_b sign_b * P_rev_b * Q_old_b) + Q_old
    # = P * (sign . P_rev) . Q_old + Q_old
    # where the dot is over b.

    m1_signs = jnp.array([(-1.0) ** (b + 1) for b in range(4)])  # (4,)

    # Vectorize over k (grid points): Q_upper has shape (4_a, 4_i, lc)
    # For each pulldown step n (sequential):
    #   contrib[i, k] = sum_b m1_signs[b] * Puj[3-b, n, k] * Q_upper[b, i, k]
    #   Q_upper[a, i, k] += Puj[a, n, k] * contrib[i, k]

    # Puj_rev[b, n, k] = Puj[3-b, n, k]
    Puj_rev = Puj[jnp.array([3, 2, 1, 0]), :, :]  # (4_b, NI, lc)

    def pulldown_step(Q: jnp.ndarray, n: jnp.ndarray) -> tuple:
        """One pulldown step: n indexes into Puj."""
        # Puj[:, n, :]: shape (4, lc) -- P values at this shift
        P_n = Puj[:, n, :]  # (4_a, lc)
        P_rev_n = Puj_rev[:, n, :]  # (4_b, lc)

        # contrib[i, k] = sum_b m1_signs[b] * P_rev_n[b, k] * Q[b, i, k]
        # = sum_b (m1_signs[b] * P_rev_n[b, k]) * Q[b, i, k]
        signed_P_rev = m1_signs[:, None] * P_rev_n  # (4_b, lc)
        # Q[b, i, k] * signed_P_rev[b, k] -> sum over b
        contrib = jnp.einsum('bk,bik->ik', signed_P_rev, Q)  # (4_i, lc)

        # Q_new[a, i, k] = Q[a, i, k] + P_n[a, k] * contrib[i, k]
        Q_new = Q + P_n[:, None, :] * contrib[None, :, :]
        return Q_new, None

    # Iterate from n=NI-1 down to 0
    n_range_pulldown = jnp.arange(NI - 1, -1, -1)
    Q_upper, _ = jax.lax.scan(pulldown_step, Q_upper, n_range_pulldown)

    # P_a(u_k) on the cut
    x_cut_rescaled = x_of_u_long(uA / g, 1.0)  # (lc,)

    # P[a, k] = x^{Mt[a]} * sum_n c[a,n] * x^{2n}
    # Pt[a, k] = x^{-Mt[a]} * sum_n c[a,n] * x^{-2n}
    x2 = x_cut_rescaled ** 2  # (lc,)
    # Powers: x^{2n} for n=0..N0
    x2_pow = x2[None, :] ** m_powers[:, None]  # (N0+1, lc) -- but need (lc, N0+1)
    # Actually let's do x2_pow[k, n] = x2[k]^n
    x2_pow = x2[:, None] ** m_powers[None, :]  # (lc, N0+1)

    # p_sum[a, k] = sum_n c[a, n] * x2_pow[k, n]
    p_sum = jnp.einsum('an,kn->ak', c, x2_pow)  # (4, lc)
    # pt_sum[a, k] = sum_n c[a, n] / x2_pow[k, n]
    pt_sum = jnp.einsum('an,kn->ak', c, 1.0 / x2_pow)  # (4, lc)

    # x^{Mt[a]}: shape (4, lc)
    xMt = x_cut_rescaled[None, :] ** Mt[:, None]  # (4, lc)

    P = xMt * p_sum  # (4, lc)
    Pt = (1.0 / xMt) * pt_sum  # (4, lc)

    # Qlower[k, i] = -sum_a (-1)^{a+1} * P[3-a, k] * Q_upper[a, i, k]
    # Qtlower[k, i] = -sum_a (-1)^{a+1} * Pt[3-a, k] * Q_upper[a, i, k]

    # P_rev[a, k] = P[3-a, k]
    P_rev = P[jnp.array([3, 2, 1, 0]), :]  # (4_a, lc)
    Pt_rev = Pt[jnp.array([3, 2, 1, 0]), :]  # (4_a, lc)

    # -sum_a m1_signs[a] * P_rev[a, k] * Q_upper[a, i, k]
    signed_P_rev_cut = -m1_signs[:, None] * P_rev  # (4_a, lc)
    signed_Pt_rev_cut = -m1_signs[:, None] * Pt_rev  # (4_a, lc)

    # Qlower[k, i] = sum_a signed_P_rev_cut[a, k] * Q_upper[a, i, k]
    Qlower = jnp.einsum('ak,aik->ki', signed_P_rev_cut, Q_upper)  # (lc, 4_i)
    Qtlower = jnp.einsum('ak,aik->ki', signed_Pt_rev_cut, Q_upper)  # (lc, 4_i)

    return Qlower, Qtlower, Q_upper, P, Pt


def _compute_gluing_residual(Q_upper: jnp.ndarray, Qlower: jnp.ndarray,
                             Qtlower: jnp.ndarray,
                             lc: int) -> tuple:
    """Compute gluing constant alpha_Q and deltaP residual.

    Returns (deltaP, deltaPt) each of shape (lc, 4).
    """
    # alpha_Q = Re(mean(Q0/conj(Q2) + Qt0/conj(Qt2)
    #                   - Q1/conj(Q3) - Qt1/conj(Qt3))) / 4
    alfaQ = jnp.mean(
        Qlower[:, 0] / jnp.conj(Qlower[:, 2])
        + Qtlower[:, 0] / jnp.conj(Qtlower[:, 2])
        - Qlower[:, 1] / jnp.conj(Qlower[:, 3])
        - Qtlower[:, 1] / jnp.conj(Qtlower[:, 3])
    ) / 4
    alfaQ = jnp.real(alfaQ) + 0j  # take real part

    # deltaP[k, a] from gluing equations
    # dp[k, a] = Q_upper[a, 0, k] * (Qlower[k, 3] + conj(Qlower[k, 1]) / alfaQ)
    #          - Q_upper[a, 1, k] * (Qlower[k, 2] - conj(Qlower[k, 0]) / alfaQ)
    #          + Q_upper[a, 2, k] * (Qlower[k, 1] + conj(Qlower[k, 3]) * alfaQ)
    #          - Q_upper[a, 3, k] * (Qlower[k, 0] - conj(Qlower[k, 2]) * alfaQ)

    # Build the "gluing vector" G[k, i] for lower Q:
    G = jnp.stack([
        Qlower[:, 3] + jnp.conj(Qlower[:, 1]) / alfaQ,      # i=0
        -(Qlower[:, 2] - jnp.conj(Qlower[:, 0]) / alfaQ),    # i=1 (with minus)
        Qlower[:, 1] + jnp.conj(Qlower[:, 3]) * alfaQ,       # i=2
        -(Qlower[:, 0] - jnp.conj(Qlower[:, 2]) * alfaQ),    # i=3 (with minus)
    ], axis=1)  # (lc, 4_i)

    Gt = jnp.stack([
        Qtlower[:, 3] + jnp.conj(Qtlower[:, 1]) / alfaQ,
        -(Qtlower[:, 2] - jnp.conj(Qtlower[:, 0]) / alfaQ),
        Qtlower[:, 1] + jnp.conj(Qtlower[:, 3]) * alfaQ,
        -(Qtlower[:, 0] - jnp.conj(Qtlower[:, 2]) * alfaQ),
    ], axis=1)  # (lc, 4_i)

    # deltaP[k, a] = sum_i Q_upper[a, i, k] * G[k, i]
    deltaP = jnp.einsum('aik,ki->ka', Q_upper, G)  # (lc, 4_a)
    deltaPt = jnp.einsum('aik,ki->ka', Q_upper, Gt)  # (lc, 4_a)

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
        x_cut_half_inv = 1.0 / x_cut_half  # (lc,)
        deltaP = deltaP * x_cut_half_inv[:, None]
        deltaPt = deltaPt * x_cut_half[:, None]

    # Symmetric and antisymmetric combinations
    fS = (deltaP + deltaPt) / 2  # (lc, 4)
    fA = (deltaP - deltaPt) / (2 * suA[:, None])  # (lc, 4)

    # Chebyshev transform: cS[n, a] = (1/lc) * sum_k fS[k, a] * CT[lc-1-k, n]
    CT_rev = CT[::-1, :]  # reverse rows
    cS = CT_rev.T @ fS / lc  # (lc, 4)

    # cA[n, a] = (2ig/lc) * sum_k fA[k, a] * CU[lc-1-k, n-1]  for n >= 1
    CU_rev = CU[::-1, :]
    cA_inner = CU_rev.T @ fA / lc  # (lc, 4)
    cA = jnp.zeros((lc, 4), dtype=complex)
    cA = cA.at[1:, :].set(cA_inner[:lc - 1, :] * 2j * g)

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
    # gauge_indices = [(a, n)] pairs where c[a][n] = 0 (C++ convention, n=0..N0)
    # c[a][n] maps to params index 1 + a*N0 + (n-1), so c-block index = a*N0 + (n-1)
    skip_indices = set()
    for a, n in gauge_indices:
        skip_indices.add(a * N0 + (n - 1))  # n >= 1 always for gauge-fixed coefficients

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
        skip_indices.add(a * N0 + (n - 1))

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
