import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize, broyden1
from scipy.interpolate import RegularGridInterpolator


# ================================================================
# Fixed parameters
# ================================================================

DT          = 0.01   # Time step
T_MAX       = 5.0    # Horizon (backward induction runs from T_MAX to 0)
N_STEPS     = int(T_MAX / DT)

MU          = 0.02   # Drift
SIGMA       = 0.2    # Volatility
KAPPA       = 3.0    # Ambiguity aversion
GAMMA       = 2.0    # Risk aversion
DELTA       = 0.05   # Discount rate
DECOH       = 0.5    # Decoherence rate

J           = 5      # Number of agents
AD_1_SUPPLY = 0.0    # Net supply of Arrow security 1
AD_2_SUPPLY = 0.0    # Net supply of Arrow security 2
BAR_ALPHA_1 = float(J)   # Total supply of asset 1
BAR_ALPHA_2 = float(J)   # Total supply of asset 2

# Grid resolution per dimension (coarse — increase for production runs)
N_OMEGA     = 8
N_Z         = 8
N_AD        = 4      # Points per Arrow-security holding dimension
N_ALPHA     = 4      # Points per risky-asset holding dimension
N_COH       = 3      # Points for coherence ∈ [0,1]

# Price approximation: 6 price functions × 3 log-linear coefficients = 18 params
# Order: P_1, P_2, Q_1H, Q_1L, Q_2H, Q_2L
N_PRICE_PARAMS = 18

# Equilibrium solver tolerances
BROYDEN_TOL = 1e-4
MAX_PRICE_ITER = 50


# ================================================================
# State and control layout
# ================================================================
#
# State vector (length 9):
#   [0]     omega       – aggregate endowment
#   [1]     z           – individual endowment
#   [2:4]   ad_1        – Arrow-security 1 holdings, shape (2,)  (H,L states)
#   [4:6]   ad_2        – Arrow-security 2 holdings, shape (2,)
#   [6]     alpha_1     – risky-asset 1 holding
#   [7]     alpha_2     – risky-asset 2 holding
#   [8]     coherence   – quantum coherence ∈ [0,1]
#
# Control vector (length 28):
#   [0:4]   C           – consumption in each of the 4 next-period nodes
#   [4:8]   AD_1_H      – Arrow-security 1 purchases, H branch (4 nodes)
#   [8:12]  AD_1_L      – Arrow-security 1 purchases, L branch (4 nodes)
#   [12:16] AD_2_H      – Arrow-security 2 purchases, H branch (4 nodes)
#   [16:20] AD_2_L      – Arrow-security 2 purchases, L branch (4 nodes)
#   [20:24] ALPHA_1     – risky-asset 1 chosen (same across nodes → broadcast)
#   [24:28] ALPHA_2     – risky-asset 2 chosen


# ================================================================
# Helper: unpack
# ================================================================

def unpack_controls(controls):
    """
    Returns
    -------
    C       : (4,1)  consumption in each next-period node
    AD_1    : (4,2)  Arrow-sec-1 holdings  [:, 0] = H branch, [:, 1] = L branch
    AD_2    : (4,2)  Arrow-sec-2 holdings
    ALPHA_1 : (4,1)  risky-asset 1
    ALPHA_2 : (4,1)  risky-asset 2
    """
    C      = controls[0:4].reshape(4, 1)
    # Each of H/L is a length-4 vector; stack as columns → (4,2)
    AD_1   = np.column_stack([controls[4:8], controls[8:12]])
    AD_2   = np.column_stack([controls[12:16], controls[16:20]])
    ALPHA_1 = controls[20:24].reshape(4, 1)
    ALPHA_2 = controls[24:28].reshape(4, 1)
    return C, AD_1, AD_2, ALPHA_1, ALPHA_2


def unpack_states(state):
    omega     = state[0]
    z         = state[1]
    ad_1      = state[2:4]   # shape (2,)
    ad_2      = state[4:6]   # shape (2,)
    alpha_1   = state[6]
    alpha_2   = state[7]
    coherence = state[8]
    return omega, z, ad_1, ad_2, alpha_1, alpha_2, coherence


# ================================================================
# Helper: transitions
# ================================================================

def evolve_omega(omega):
    """Returns shape (2,): [omega_up, omega_dn]"""
    base = (MU - 0.5 * SIGMA**2) * DT
    return np.array([
        omega * np.exp(base + SIGMA * np.sqrt(DT)),
        omega * np.exp(base - SIGMA * np.sqrt(DT)),
    ])


def evolve_z(z):
    """Returns shape (2,): [z_up, z_dn]"""
    base = (MU - 0.5 * SIGMA**2) * DT
    return np.array([
        z * np.exp(base + SIGMA * np.sqrt(DT)),
        z * np.exp(base - SIGMA * np.sqrt(DT)),
    ])


def next_period_nodes(omega, z):
    """
    4 next-period nodes ordered as:
        0: (omega_up, z_up)
        1: (omega_up, z_dn)
        2: (omega_dn, z_up)
        3: (omega_dn, z_dn)

    Returns
    -------
    omegas : (4,)
    zs     : (4,)
    """
    o = evolve_omega(omega)  # [up, dn]
    zv = evolve_z(z)         # [up, dn]
    omegas = np.array([o[0], o[0], o[1], o[1]])
    zs     = np.array([zv[0], zv[1], zv[0], zv[1]])
    return omegas, zs


# ================================================================
# Density operator
# ================================================================

def density_operator(coherence):
    """
    Constructs the 4×4 density matrix.

    The maximally mixed diagonal is modified by coherence:
    ρ = diag(ρ_mixed) + coherence * off_diag(ρ_pure).
    """
    v_up = np.array([1, 0, 1, 0], dtype=float) / np.sqrt(2)
    v_dn = np.array([0, 1, 0, 1], dtype=float) / np.sqrt(2)
    rho_pure  = 0.5 * np.outer(v_up, v_up) + 0.5 * np.outer(v_dn, v_dn)
    rho_mixed = np.diag(np.diag(rho_pure))
    return rho_mixed + coherence * (rho_pure - rho_mixed)


# ================================================================
# Price approximation  (log-linear in (1, omega, z))
# ================================================================

def price_approx(A, omegas, zs):
    """
    Parameters
    ----------
    A      : (3,)  log-linear coefficients [a0, a1, a2]
    omegas : (4,)  next-period aggregate endowments
    zs     : (4,)  next-period individual endowments

    Returns
    -------
    prices : (4,1)  approximated price in each node
    """
    log_states = np.column_stack([np.ones(4), np.log(omegas), np.log(zs)])  # (4,3)
    return np.exp(log_states @ A).reshape(4, 1)


def unpack_price_params(theta):
    """Split 18-vector into 6 coefficient triplets."""
    return [theta[3*k : 3*k+3] for k in range(6)]


# ================================================================
# Budget constraint  (inequality: residual ≥ 0 for SLSQP)
# ================================================================

def budget_constraint(controls, state, theta):
    C, AD_1, AD_2, ALPHA_1, ALPHA_2 = unpack_controls(controls)
    omega, z, ad_1, ad_2, alpha_1, alpha_2, _ = unpack_states(state)

    omegas, zs = next_period_nodes(omega, z)

    coeffs = unpack_price_params(theta)
    P_1  = price_approx(coeffs[0], omegas, zs)   # (4,1)
    P_2  = price_approx(coeffs[1], omegas, zs)
    Q_1H = price_approx(coeffs[2], omegas, zs)
    Q_1L = price_approx(coeffs[3], omegas, zs)
    Q_2H = price_approx(coeffs[4], omegas, zs)
    Q_2L = price_approx(coeffs[5], omegas, zs)

    # Current Arrow-security payoffs received this period
    # ad_1[0] pays in H nodes (nodes 0,1), ad_1[1] in L nodes (nodes 2,3)
    ad_1_payoff = np.array([ad_1[0], ad_1[0], ad_1[1], ad_1[1]]).reshape(4, 1)
    ad_2_payoff = np.array([ad_2[0], ad_2[1], ad_2[0], ad_2[1]]).reshape(4, 1)
    ad_payoff   = ad_1_payoff + ad_2_payoff

    # Budget: wealth in + asset income ≥ consumption + new portfolio cost
    wealth_in = (P_1 * alpha_1 + P_2 * alpha_2) * DT + ad_payoff
    new_cost  = (C * DT
                 + P_1 * ALPHA_1 + P_2 * ALPHA_2
                 + Q_1H * AD_1[:, 0:1] + Q_1L * AD_1[:, 1:2]
                 + Q_2H * AD_2[:, 0:1] + Q_2L * AD_2[:, 1:2])

    residual = wealth_in - new_cost   # (4,1), should be ≥ 0
    return residual.flatten()


# ================================================================
# Laws of motion  →  4 next-period state vectors, shape (4, 9)
# ================================================================

def laws_of_motion(controls, state):
    """
    Returns next_states : (4, 9) — one row per next-period node.
    Node ordering: (ω↑,z↑), (ω↑,z↓), (ω↓,z↑), (ω↓,z↓).
    """
    _, AD_1, AD_2, ALPHA_1, ALPHA_2 = unpack_controls(controls)
    omega, z, _, _, _, _, coherence = unpack_states(state)

    omegas, zs = next_period_nodes(omega, z)
    coh_next = coherence * np.exp(-DECOH * DT) * np.ones(4)

    next_states = np.column_stack([
        omegas,                     # (4,)
        zs,                         # (4,)
        AD_1,                       # (4,2)
        AD_2,                       # (4,2)
        ALPHA_1.flatten(),          # (4,)
        ALPHA_2.flatten(),          # (4,)
        coh_next,                   # (4,)
    ])
    return next_states              # (4, 9)


# ================================================================
# Utility
# ================================================================

def utility_matrix(C):
    """
    C : (4,1)
    Returns U : (4,4)  where U_ii = C_i^(1-γ)/(1-γ) and off-diagonal
    encodes ambiguity aversion via KAPPA * exp(-(C_i - C_j)^2).
    """
    C_flat = C.flatten()
    diag_vals = C_flat**(1.0 - GAMMA) / (1.0 - GAMMA)
    diff = C_flat[:, None] - C_flat[None, :]   # (4,4)
    U = KAPPA * np.exp(-(diff**2)) + np.diag(diag_vals)
    return U


def utility(controls, state):
    _, _, _, _, _, _, coherence = unpack_states(state)
    C, *_ = unpack_controls(controls)
    U   = utility_matrix(C)
    rho = density_operator(coherence)
    return np.trace(U @ rho)


# ================================================================
# Expected continuation value
# ================================================================

def expected_V(V_interp, next_states):
    """
    Equal-probability average over 4 nodes (each has prob 0.25).

    V_interp : callable  (RegularGridInterpolator)
    next_states : (4, 9)
    """
    vals = V_interp(next_states)   # (4,)
    return 0.25 * vals.sum()


# ================================================================
# Bellman objective  (negated for minimizer)
# ================================================================

def objective(controls, state, V_interp):
    next_states = laws_of_motion(controls, state)
    val = (utility(controls, state) * DT
           + np.exp(-DELTA * DT) * expected_V(V_interp, next_states))
    return -val


# ================================================================
# Agent optimization at a single state
# ================================================================

def agent_optimization(theta, V_interp, state):
    constraints = [{
        'type': 'ineq',
        'fun': budget_constraint,
        'args': (state, theta),
    }]
    x0     = np.ones(28) * 0.5
    bounds = [(1e-4, 5.0)] + [(-3.0, 3.0)] * 27

    result = minimize(
        objective,
        x0,
        args=(state, V_interp),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'maxiter': 500},
    )
    return result


# ================================================================
# Grid construction
# ================================================================

def build_grids():
    """
    Returns a tuple of 1-D arrays (one per state dimension) for use with
    RegularGridInterpolator, and a flat list of all grid points.
    """
    omega_grid = np.linspace(0.5, 2.0, N_OMEGA)
    z_grid     = np.linspace(0.5, 2.0, N_Z)
    ad1_grid   = np.linspace(-1.0, 1.0, N_AD)    # Arrow-sec-1 H holding
    ad1L_grid  = np.linspace(-1.0, 1.0, N_AD)    # Arrow-sec-1 L holding
    ad2_grid   = np.linspace(-1.0, 1.0, N_AD)
    ad2L_grid  = np.linspace(-1.0, 1.0, N_AD)
    a1_grid    = np.linspace(0.0, 2.0, N_ALPHA)
    a2_grid    = np.linspace(0.0, 2.0, N_ALPHA)
    coh_grid   = np.linspace(0.0, 1.0, N_COH)

    grids = (omega_grid, z_grid,
             ad1_grid, ad1L_grid,
             ad2_grid, ad2L_grid,
             a1_grid, a2_grid,
             coh_grid)

    # Build flat list of all grid points
    mesh   = np.meshgrid(*grids, indexing='ij')
    points = np.column_stack([m.ravel() for m in mesh])  # (N_total, 9)
    shape  = tuple(g.size for g in grids)

    return grids, points, shape


# ================================================================
# Compute excess demand given solutions for J agents
# ================================================================

def compute_excess_demand(solutions_per_agent):
    """
    solutions_per_agent : list of J arrays, each shape (N_grid, 28)

    Markets cleared:
      Arrow-sec 1 (H & L): net demand = 0  (zero net supply)
      Arrow-sec 2 (H & L): net demand = 0
      Asset 1: sum of ALPHA_1 = BAR_ALPHA_1
      Asset 2: sum of ALPHA_2 = BAR_ALPHA_2

    Returns excess : (6,) vector  [ED_AD1H, ED_AD1L, ED_AD2H, ED_AD2L,
                                   ED_A1,   ED_A2]
    """
    agg = np.zeros((solutions_per_agent[0].shape[0], 28))
    for sol in solutions_per_agent:
        agg += sol

    # AD_1 H: indices 4:8 (but per-agent they're length-4 vectors for nodes)
    # For equilibrium we need the *chosen* portfolio (scalar per agent per grid
    # point).  We take the mean across the 4 contingent nodes as a representative
    # demand measure.
    ED = np.zeros(6)
    ED[0] = agg[:, 4:8].mean(axis=1).sum()            # AD_1 H
    ED[1] = agg[:, 8:12].mean(axis=1).sum()           # AD_1 L
    ED[2] = agg[:, 12:16].mean(axis=1).sum()          # AD_2 H
    ED[3] = agg[:, 16:20].mean(axis=1).sum()          # AD_2 L
    ED[4] = agg[:, 20:24].mean(axis=1).sum() - BAR_ALPHA_1  # Asset 1
    ED[5] = agg[:, 24:28].mean(axis=1).sum() - BAR_ALPHA_2  # Asset 2

    return ED


# ================================================================
# One backward step: solve agents, update V, return V and policies
# ================================================================

def backward_step(theta, V_next_interp, grids, grid_points, grid_shape):
    """
    Given price parameters theta and V^{t+dt} (as interpolator), solve each
    agent's problem at every grid point and return:
        V_t      : ndarray of shape grid_shape  (value function at time t)
        policies : list of J arrays, each (N_grid, 28)
    """
    N_grid = grid_points.shape[0]

    # For a representative-agent version we solve once; with J heterogeneous
    # agents you would maintain J separate state grids.  Here we demonstrate
    # the single-agent (or symmetric) case; replace with a loop over j for
    # full heterogeneity.
    V_flat    = np.zeros(N_grid)
    policy    = np.zeros((N_grid, 28))

    for idx in range(N_grid):
        state  = grid_points[idx]
        result = agent_optimization(theta, V_next_interp, state)

        # Bellman value at the optimum
        V_flat[idx]   = -result.fun   # we minimised the negative
        policy[idx]   = result.x

    V_t = V_flat.reshape(grid_shape)

    # Build interpolator for V_t (used in the next backward step)
    V_t_interp = RegularGridInterpolator(
        grids,
        V_t,
        method='linear',
        bounds_error=False,
        fill_value=None,
    )

    return V_t, V_t_interp, [policy] * J   # replicate for symmetric agents


# ================================================================
# Equilibrium price update via Broyden
# ================================================================

def equilibrium_prices_at_t(theta_init, V_next_interp, grids, grid_points, grid_shape):
    """
    Outer loop: iterate theta until excess demand ≈ 0 using Broyden.

    Returns theta_eq, V_t, policies.
    """
    def excess_demand_fn(theta):
        _, _, policies = backward_step(theta, V_next_interp,
                                       grids, grid_points, grid_shape)
        ED = compute_excess_demand(policies)
        print(f"    |ED| = {la.norm(ED):.6f}")
        return ED

    # Broyden finds theta such that excess_demand_fn(theta) = 0
    theta_eq = broyden1(
        excess_demand_fn,
        theta_init,
        f_tol=BROYDEN_TOL,
        maxiter=MAX_PRICE_ITER,
        verbose=False,
    )

    V_t, V_t_interp, policies = backward_step(theta_eq, V_next_interp,
                                               grids, grid_points, grid_shape)
    return theta_eq, V_t, V_t_interp, policies


# ================================================================
# Main backward induction loop
# ================================================================

def run_backward_induction():
    """
    Runs the full backward induction from t = T_MAX to t = 0.

    Returns
    -------
    theta_path : (N_STEPS, N_PRICE_PARAMS)  price coefficients at each step
    V_path     : list of N_STEPS value-function arrays (each shape grid_shape)
    """
    grids, grid_points, grid_shape = build_grids()

    # Terminal condition: V^T = 0
    V_terminal = np.zeros(grid_shape)
    V_interp   = RegularGridInterpolator(
        grids,
        V_terminal,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    # Initial price guess: all log-linear coefficients = 0 (prices = 1)
    theta = np.zeros(N_PRICE_PARAMS)

    theta_path = np.zeros((N_STEPS, N_PRICE_PARAMS))
    V_path     = []

    print(f"Starting backward induction: {N_STEPS} steps.")

    for step in range(N_STEPS):
        t = T_MAX - step * DT
        print(f"  Step {step+1}/{N_STEPS},  t = {t:.3f}")

        theta, V_t, V_interp, policies = equilibrium_prices_at_t(
            theta, V_interp, grids, grid_points, grid_shape
        )

        theta_path[step] = theta
        V_path.append(V_t)

    print("Backward induction complete.")
    return theta_path, V_path, grids


# ================================================================
# Diagnostic plot
# ================================================================

def plot_value_function_slice(V_path, grids, step=0):
    """Plot a 2-D slice of V (omega, z) at fixed midpoint of other dims."""
    V = V_path[step]
    mid = tuple(s // 2 for s in V.shape[2:])   # midpoint indices for dims 2–8
    idx = (slice(None), slice(None)) + mid

    plt.figure(figsize=(7, 5))
    plt.contourf(grids[0], grids[1], V[idx], levels=20, cmap='viridis')
    plt.colorbar(label='V')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$z$')
    plt.title(f'Value function slice at step {step}')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/V_slice.png', dpi=150)
    plt.close()
    print("Saved V_slice.png")


# ================================================================
# Entry point
# ================================================================

if __name__ == '__main__':
    theta_path, V_path, grids = run_backward_induction()
    plot_value_function_slice(V_path, grids, step=0)
