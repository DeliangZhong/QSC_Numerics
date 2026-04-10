"""Spectral Q-solver: solve for Q_{a|i} via collocation in x-basis.

The 1/x (Zhukovsky) expansion converges on and near the cut because
the nearest Q-singularity maps to |x| = r < 1. On the cut, x = e^{iθ},
so the expansion becomes a Fourier series — perfectly conditioned.

Q_{a|i}(u) = (g·x(u))^{α_{a|i}} Σ_n q_{a|i,n} · x(u)^{-n}

The QQ-relation at collocation points gives a linear system for q_n.
"""

import jax.numpy as jnp

from qsc.zhukovsky import x_of_u_long, x_of_u_short


def spectral_Q_solve(c, BB, Mt, Mhat, g, NQ, N0, lc, uA):
    """Solve for Q_{a|i} at probe points via spectral collocation in x-basis.

    Uses the Zhukovsky variable x: Q_{a|i}(u) = (gx)^α Σ_n q_n x^{-n}
    where α = Mhat[i] - Mt[a]. The expansion converges for |x| > r < 1.

    Returns: Q_upper (4,4,lc), P (4,lc), Pt (4,lc) — same as _evaluate_Q_and_pulldown.
    """
    II = 1j
    alpha_ai = Mhat[None, :] - Mt[:, None]  # (4_a, 4_i)
    m1_signs = jnp.array([-1.0, 1.0, -1.0, 1.0])

    # --- Collocation points in the UHP ---
    # Height H above the cut. x(u + iH) has |x| > 1 (in UHP).
    H = 1.5
    M = NQ + 4  # overdetermined
    theta_colloc = jnp.pi * (2 * jnp.arange(1, M + 1) - 1) / (2 * M)
    u_colloc = 2 * g * jnp.cos(theta_colloc) + II * H  # (M,)

    # x-values at collocation points and their ±i/2 shifts
    u_plus = u_colloc + 0.5 * II
    u_minus = u_colloc - 0.5 * II

    x_plus = x_of_u_short(u_plus / g)    # |x| > 1 (height H+0.5)
    x_minus = x_of_u_short(u_minus / g)  # |x| > 1 (height H-0.5 > 0.5)

    # --- Evaluate P at collocation points ---
    def eval_P_at_x(x_vals):
        """P_a(x) = x^{-Mt[a]} Σ_m c[a,m] x^{-2m}. Returns (4, len(x_vals))."""
        x_inv_sq = 1.0 / (x_vals ** 2)
        m_pows = jnp.arange(N0 + 1)
        x_inv_sq_pow = x_inv_sq[:, None] ** m_pows[None, :]  # (M, N0+1)
        p_sum = jnp.einsum('am,km->ak', c, x_inv_sq_pow)     # (4, M)
        x_Mt = x_vals[None, :] ** Mt[:, None]                 # (4, M)
        return p_sum / x_Mt

    # Evaluate P at collocation points (not shifted — the QQ-relation uses P(u))
    x_colloc = x_of_u_short(u_colloc / g)
    P_colloc = eval_P_at_x(x_colloc)  # P_a(u_m), (4, M)
    Pt_colloc = m1_signs[:, None] * P_colloc[jnp.array([3, 2, 1, 0]), :]  # P̃^b, (4, M)

    # --- Build and solve the spectral system for each i ---
    Q_upper_all = jnp.zeros((4, 4, lc), dtype=complex)

    for i_idx in range(4):
        n_unknowns = 4 * NQ
        n_equations = 4 * M

        A_mat = jnp.zeros((n_equations, n_unknowns), dtype=complex)
        rhs = jnp.zeros(n_equations, dtype=complex)

        for m_idx in range(M):
            P_m = P_colloc[:, m_idx]    # (4,)
            Pt_m = Pt_colloc[:, m_idx]  # (4,)
            xp = x_plus[m_idx]          # x(u_m + i/2)
            xm = x_minus[m_idx]         # x(u_m - i/2)

            for a in range(4):
                row = m_idx * 4 + a
                alpha_a = alpha_ai[a, i_idx]

                # coeff[b] = δ_{ab} + P_a P̃^b
                coeff_ab = jnp.where(
                    jnp.arange(4) == a, 1.0 + P_m[a] * Pt_m, P_m[a] * Pt_m
                )

                # n=0 (known: q_{b,0} = BB[b,i]) → RHS
                for b in range(4):
                    alpha_b = alpha_ai[b, i_idx]
                    # [δ+PP̃] Q+(u_m) for n=0: coeff[b] * BB[b,i] * (g·x+)^α_b
                    rhs = rhs.at[row].add(
                        -coeff_ab[b] * BB[b, i_idx] * (g * xp) ** alpha_b
                    )
                # -Q-(u_m) for n=0: BB[a,i] * (g·x-)^α_a
                rhs = rhs.at[row].add(BB[a, i_idx] * (g * xm) ** alpha_a)

                # n=1..NQ → A matrix
                for n in range(1, NQ + 1):
                    for b in range(4):
                        col = b * NQ + (n - 1)
                        alpha_b = alpha_ai[b, i_idx]
                        # [δ+PP̃] * (g·x+)^α_b * x+^{-n}
                        A_mat = A_mat.at[row, col].add(
                            coeff_ab[b] * (g * xp) ** alpha_b * xp ** (-n)
                        )
                    # -Q-(u_m) for n: -(g·x-)^α_a * x-^{-n}
                    col_a = a * NQ + (n - 1)
                    A_mat = A_mat.at[row, col_a].add(
                        -(g * xm) ** alpha_a * xm ** (-n)
                    )

        # Solve
        q_sol, _, _, _ = jnp.linalg.lstsq(A_mat, rhs, rcond=1e-12)

        # Unpack q coefficients
        q_coeffs = jnp.zeros((4, NQ + 1), dtype=complex)
        for a in range(4):
            q_coeffs = q_coeffs.at[a, 0].set(BB[a, i_idx])
            for n in range(1, NQ + 1):
                q_coeffs = q_coeffs.at[a, n].set(q_sol[a * NQ + (n - 1)])

        # Evaluate Q at probe points: u_k + i/2 on the cut
        u_probe = uA + 0.5 * II  # (lc,)
        x_probe = x_of_u_short(u_probe / g)  # Zhukovsky at probe points

        for a in range(4):
            alpha_a = alpha_ai[a, i_idx]
            n_range = jnp.arange(NQ + 1)
            # Q = (g·x)^α Σ_n q_n x^{-n}
            eval_mat = (g * x_probe[:, None]) ** alpha_a * \
                       x_probe[:, None] ** (-n_range[None, :])  # (lc, NQ+1)
            Q_vals = eval_mat @ q_coeffs[a, :]  # (lc,)
            Q_upper_all = Q_upper_all.at[a, i_idx, :].set(Q_vals)

    # --- P, Pt on the cut ---
    x_cut = x_of_u_long(uA / g, 1.0)
    x2 = x_cut ** 2
    m_powers = jnp.arange(N0 + 1)
    x2_pow = x2[:, None] ** m_powers[None, :]
    p_sum = jnp.einsum('an,kn->ak', c, x2_pow)
    pt_sum = jnp.einsum('an,kn->ak', c, 1.0 / x2_pow)
    xMt = x_cut[None, :] ** Mt[:, None]
    P_cut = xMt * p_sum
    Pt_cut = (1.0 / xMt) * pt_sum

    return Q_upper_all, P_cut, Pt_cut
