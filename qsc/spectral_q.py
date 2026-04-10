"""Direct Q evaluation at probe points via basis conversion.

Instead of pulldown (which loses ~QaiShift digits), convert the
b-coefficients from the 1/u^{2n} basis (where they're computed
accurately) to the 1/x^{2n} basis (which converges at the probe
points on the cut). Then evaluate Q directly at the probe points.

The conversion uses: u = g(x + 1/x), so
  u^{s} = g^s (x+1/x)^s = g^s x^s Σ_k C(s,k) x^{-2k}

where C(s,k) is the complex binomial coefficient.
"""

import jax.numpy as jnp

from qsc.zhukovsky import x_of_u_long, x_of_u_short, cbinomial


def evaluate_Q_direct(b_all, BB, Mt, Mhat, c, g, N0, NQ, lc, uA):
    """Evaluate Q_{a|i}(u_k + i/2) directly from b-coefficients.

    Converts the 1/u expansion to 1/x expansion (convergent at probe
    points) and evaluates without pulldown.

    Returns: Q_upper (4,4,lc), P (4,lc), Pt (4,lc)
    """
    alpha_ai = Mhat[None, :] - Mt[:, None]  # (4_a, 4_i)

    # Probe points: u_k + i/2, with x_+ = x((u_k+i/2)/g)
    u_probe_shifted = uA + 0.5j  # (lc,)
    x_probe = x_of_u_short(u_probe_shifted / g)  # (lc,)

    Q_upper = jnp.zeros((4, 4, lc), dtype=complex)

    for i_idx in range(4):
        b = b_all[i_idx]  # (NQ+1, 4)

        for a in range(4):
            alpha = alpha_ai[a, i_idx]
            # q_n = BB[a,i] * b[i,n,a] — the 1/u expansion coefficient
            # of Q_{a|i}(u) = Σ_n q_n * u^{α - 2n}

            # Convert to x-basis: Q'_m = Σ_{n=0}^{m} q_n * g^{α-2n} * C(α-2n, m-n)
            # Then Q(u) = Σ_m Q'_m * x^{α - 2m}
            NQ_use = min(NQ, b.shape[0] - 1)
            Q_x = jnp.zeros(NQ_use + 1, dtype=complex)

            for m in range(NQ_use + 1):
                val = 0.0 + 0j
                for n in range(m + 1):
                    s = alpha - 2 * n  # exponent
                    q_n = BB[a, i_idx] * b[n, a]
                    # C(s, m-n) * g^s
                    binom = cbinomial(complex(s), m - n)
                    val += q_n * (g ** complex(s)) * binom
                Q_x = Q_x.at[m].set(val)

            # Evaluate at probe points: Q = Σ_m Q'_m * x^{α - 2m}
            n_range = jnp.arange(NQ_use + 1)
            # eval_mat[k, m] = x_probe[k]^{α - 2m}
            eval_mat = x_probe[:, None] ** (alpha - 2 * n_range[None, :])
            Q_vals = eval_mat @ Q_x  # (lc,)
            Q_upper = Q_upper.at[a, i_idx, :].set(Q_vals)

    # P, Pt on the cut (same as standard)
    x_cut = x_of_u_long(uA / g, 1.0)
    x2 = x_cut ** 2
    m_powers = jnp.arange(N0 + 1)
    x2_pow = x2[:, None] ** m_powers[None, :]
    p_sum = jnp.einsum('an,kn->ak', c, x2_pow)
    pt_sum = jnp.einsum('an,kn->ak', c, 1.0 / x2_pow)
    xMt = x_cut[None, :] ** Mt[:, None]
    P_cut = xMt * p_sum
    Pt_cut = (1.0 / xMt) * pt_sum

    return Q_upper, P_cut, Pt_cut
