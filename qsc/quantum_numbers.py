"""Quantum number specification and derived quantities for QSC states."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class QuantumNumbers:
    """Oscillator quantum numbers defining an N=4 SYM state.

    nb: bosonic oscillator numbers [nb1, nb2]
    nf: fermionic oscillator numbers [nf1, nf2, nf3, nf4]
    na: auxiliary oscillator numbers [na1, na2]
    sol: solution multiplicity label
    """

    nb: tuple[int, int]
    nf: tuple[int, int, int, int]
    na: tuple[int, int]
    sol: int = 1

    @property
    def L(self) -> int:
        """Lorentz spin length."""
        return (sum(self.nf) + sum(self.na) - sum(self.nb)) // 2

    @property
    def Delta0(self) -> int:
        """Bare dimension."""
        return sum(self.nf) // 2 + sum(self.na)


# Konishi operator
KONISHI = QuantumNumbers(nb=(0, 0), nf=(1, 1, 1, 1), na=(0, 0), sol=1)


def compute_Mtint(qn: QuantumNumbers) -> jnp.ndarray:
    """Integer parts of Mt[a] = Lambda_0[a] + {2,1,0,-1}[a].

    Returns array of 4 integers.
    """
    shifts = jnp.array([2, 1, 0, -1])
    nf = jnp.array(qn.nf)
    return nf + shifts


def compute_kettoLAMBDA(Mtint: jnp.ndarray) -> int:
    """2*Lambda = 1 - Mtint[0] - Mtint[3]."""
    return int(1 - Mtint[0] - Mtint[3])


def compute_Mt(Mtint: jnp.ndarray, kettoLAMBDA: int) -> jnp.ndarray:
    """Mt[a] = Mtint[a] + kettoLAMBDA/2. Returns complex array."""
    return Mtint.astype(float) + kettoLAMBDA / 2.0


def compute_Mhat0(qn: QuantumNumbers, kettoLAMBDA: int) -> jnp.ndarray:
    """Mhat at Delta=0 (bare)."""
    L = qn.L
    nb1, nb2 = qn.nb
    na1, na2 = qn.na
    kL2 = kettoLAMBDA / 2.0
    return jnp.array([
        L + nb1 + 1 + kL2,
        L + nb2 + 2 + kL2,
        -1.0 - na1 + kL2,
        -1.0 * na2 + kL2,
    ])


def compute_Mhat(Mhat0: jnp.ndarray, Delta: complex) -> jnp.ndarray:
    """Mhat[i] = Mhat0[i] +/- Delta/2."""
    signs = jnp.array([1.0, 1.0, -1.0, -1.0])
    return Mhat0 + signs * Delta / 2.0


def compute_A(Mt: jnp.ndarray, Mhat: jnp.ndarray) -> jnp.ndarray:
    """Compute A_a vector (Volin convention) and derived quantities.

    Returns (A, Af, AA) where:
    - A[a]: the A_a coefficients
    - Af[a]: the A^a coefficients (upper index)
    - AA[a][b]: A_a * A^b matrix
    """
    II = 1j

    # AAproduct[a] = i * prod_j(Mt[a]-Mhat[j]) / prod_{j!=a}(Mt[a]-Mt[j])
    AAproduct = jnp.zeros(4, dtype=complex)
    for a in range(4):
        r = II
        for j in range(4):
            r = r * (Mt[a] - Mhat[j])
        for j in range(4):
            if j != a:
                r = r / (Mt[a] - Mt[j])
        AAproduct = AAproduct.at[a].set(r)

    # VolinAfunc: A[a] = (Mhat[0]-Mt[a])*(Mt[a]-Mhat[1]) / prod_{j>a}(i*(Mt[a]-Mt[j]))
    def volin_A(a):
        r = (Mhat[0] - Mt[a]) * (Mt[a] - Mhat[1])
        for j in range(a + 1, 4):
            r = r / (II * (Mt[a] - Mt[j]))
        return r

    A = jnp.zeros(4, dtype=complex)
    A = A.at[0].set(volin_A(0))
    A = A.at[1].set(volin_A(1))
    A = A.at[2].set(AAproduct[1] / A[1])
    A = A.at[3].set(-AAproduct[0] / A[0])

    Af = jnp.zeros(4, dtype=complex)
    Af = Af.at[0].set(AAproduct[0] / A[0])
    Af = Af.at[1].set(AAproduct[1] / A[1])
    Af = Af.at[2].set(AAproduct[2] / A[2])
    Af = Af.at[3].set(AAproduct[3] / A[3])

    # From Ca0funcLR (line 567): AA[a][i] = A[a] * (-1)^{3-i} * A[3-i]
    m1_signs = jnp.array([(-1.0) ** (3 - i) for i in range(4)])
    A_rev = jnp.array([A[3 - i] for i in range(4)])
    AA = A[:, None] * (m1_signs * A_rev)[None, :]
    return A, Af, AA


def compute_B(Mt: jnp.ndarray, Mhat: jnp.ndarray) -> jnp.ndarray:
    """Compute B_i vector.

    Returns B[4] array.
    """
    II = 1j

    # Bnikafunc: B[a] = 1 / prod_{j>a}(i*(Mhat[j]-Mhat[a]))
    def bnika(a):
        r = 1.0 + 0j
        for j in range(a + 1, 4):
            r = r / (II * (Mhat[j] - Mhat[a]))
        return r

    # BBfunc: i * prod_j(Mhat[a]-Mt[j]) / prod_{j!=a}(Mhat[a]-Mhat[j])
    def bb_func(a):
        r = II
        for j in range(4):
            r = r * (Mhat[a] - Mt[j])
        for j in range(4):
            if j != a:
                r = r / (Mhat[a] - Mhat[j])
        return r

    B = jnp.zeros(4, dtype=complex)
    B = B.at[0].set(bnika(0))
    B = B.at[1].set(bnika(1))
    B = B.at[2].set(bb_func(1) / B[1])
    B = B.at[3].set(bb_func(3) / B[0])
    return B


def compute_BB(A: jnp.ndarray, B: jnp.ndarray, Mt: jnp.ndarray,
               Mhat: jnp.ndarray) -> jnp.ndarray:
    """BB[a][i] = i * A[a] * B[i] / (Mt[a] - Mhat[i])."""
    II = 1j
    # Outer products divided element-wise
    Mt_grid = Mt[:, None]  # (4,1)
    Mhat_grid = Mhat[None, :]  # (1,4)
    return II * jnp.outer(A, B) / (Mt_grid - Mhat_grid)


def compute_alfa(Mt: jnp.ndarray, Mhat: jnp.ndarray) -> jnp.ndarray:
    """alfa[a][i] = Mhat[i] - Mt[a]."""
    return Mhat[None, :] - Mt[:, None]


def compute_gauge_info(Mtint: jnp.ndarray, N0: int) -> dict:
    """Compute gauge-fixing info for TypeI: which c[a][n] are set to zero.

    Returns dict with:
    - Nch: number of free coefficients per a (excluding gauge-fixed)
    - dimV: total dimension of unconstrained variable vector
    - gauge_indices: list of (a, n) pairs that are gauge-fixed to zero
    """
    Mtint_np = [int(Mtint[i]) for i in range(4)]

    # Nch[0] = N0 always (a=0 is reference, no gauge fixing on c[0])
    Nch = [N0, N0, N0, N0]
    gauge_indices = []

    # a=1: if (Mtint[0]-Mtint[1]) is even and <= 2*N0, one coefficient is fixed
    diff01 = Mtint_np[0] - Mtint_np[1]
    if diff01 % 2 == 0 and 0 <= diff01 <= 2 * N0:
        Nch[1] -= 1
        gauge_indices.append((1, diff01 // 2))

    # a=2: check both (0-2) and (1-2)
    diff02 = Mtint_np[0] - Mtint_np[2]
    if diff02 % 2 == 0 and 0 <= diff02 <= 2 * N0:
        Nch[2] -= 1
        gauge_indices.append((2, diff02 // 2))
    diff12 = Mtint_np[1] - Mtint_np[2]
    if diff12 % 2 == 0 and 0 <= diff12 <= 2 * N0:
        Nch[2] -= 1
        gauge_indices.append((2, diff12 // 2))

    # a=3: check (0-3)
    diff03 = Mtint_np[0] - Mtint_np[3]
    if diff03 % 2 == 0 and 0 <= diff03 <= 2 * N0:
        Nch[3] -= 1
        gauge_indices.append((3, diff03 // 2))

    dimV = 1 + N0 + Nch[1] + Nch[2] + Nch[3]
    return {"Nch": Nch, "dimV": dimV, "gauge_indices": gauge_indices}


def compute_Nas(Mtint: jnp.ndarray, kettoLAMBDA: int) -> list[list[int]]:
    """Compute Nas[a][0..1] array used in Fourier inversion.

    Nas[a][0] = -Mtint[a] - kettoLAMBDA/2  (when kettoLAMBDA is even)
    """
    Mtint_np = [int(Mtint[i]) for i in range(4)]
    Nas = [[0, 0] for _ in range(4)]
    if kettoLAMBDA % 2 == 0:
        for a in range(4):
            Nas[a][0] = -Mtint_np[a] - kettoLAMBDA // 2
            Nas[a][1] = Mtint_np[a] + kettoLAMBDA // 2 - 1
    else:
        half_sum = (Mtint_np[0] + Mtint_np[3]) // 2
        for a in range(4):
            Nas[a][0] = -Mtint_np[a] + half_sum
            Nas[a][1] = Mtint_np[a] - 1 - half_sum
    return Nas


def compute_PhiV(N0: int, Nch: list[int]) -> jnp.ndarray:
    """Compute the phase vector indicating real (1) or imaginary (i) variables.

    For TypeI (LR + parity):
    - V[0] = Delta: real
    - V[1..N0] = c[0][1..N0]: imaginary (a=0 → c has Im part)
    - V[N0+1..N0+Nch[1]] = c[1] entries: real
    - V[...] = c[2] entries: imaginary
    - V[...] = c[3] entries: real
    """
    parts = [
        jnp.array([1.0 + 0j]),
        jnp.full(N0, 1j),
        jnp.full(Nch[1], 1.0 + 0j),
        jnp.full(Nch[2], 1j),
        jnp.full(Nch[3], 1.0 + 0j),
    ]
    return jnp.concatenate(parts)
