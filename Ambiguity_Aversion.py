import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, lognorm
from scipy.interpolate import RegularGridInterpolator


# ================================================================
# Fixed paramters:
# ================================================================


DT = 0.01   # Time step
MU = 0.02   # Drift
SIGMA = 0.2 # Volatility
KAPPA = 3.0 # Ambiguity aversion
GAMMA = 2.0 # Risk aversion
PI = np.pi  # Pi constant
DELTA = 0.05  # Discount rate
UPDATE_SPEED = 0.1  # Speed of updating the value function
J = 5 # Number of agents
AD_1_SUPPLY = 0.0 # Supply of Arrow security 1
AD_2_SUPPLY = 0.0 # Supply of Arrow security 2
BAR_ALPHA_1 = J # Supply of asset 1
BAR_ALPHA_2 = J # Supply of asset 2
DECOH = 0.5 # Decoherence rate


# ================================================================
# Helper functions:
# ================================================================


def unpack_controls(controls):
    C = controls[0:4].reshape(-1, 1)           # Shape: (4,1)
    AD_1 = np.array([controls[4:8], controls[8:12]]).reshape(-1, 1)   # Shape: (4,2)
    AD_2 = np.array([controls[12:16], controls[16:20]]).reshape(-1, 1) # Shape: (4,2)
    ALPHA_1 = controls[20:24].reshape(-1, 1)  # Shape: (4,1)
    ALPHA_2 = controls[24:28].reshape(-1, 1)  # Shape: (4,1)
    
    return C, AD_1, AD_2, ALPHA_1, ALPHA_2


def unpack_states(states):
    omega = states[0]
    z = states[1]
    ad_1 = states[2:4]
    ad_2 = states[4:6]
    alpha_1 = states[6]
    alpha_2 = states[7]
    coherence = states[8]

    return omega, z, ad_1, ad_2, alpha_1, alpha_2, coherence


def evolve_omega(omega):
    omega_up = omega * np.exp((MU - SIGMA**2 / 2.0) * DT + SIGMA * np.sqrt(DT))
    omega_dn = omega * np.exp((MU - SIGMA**2 / 2.0) * DT - SIGMA * np.sqrt(DT))

    return np.array([[omega_up, omega_dn]])


def evolve_z(z):
    z_up = z * np.exp((MU - SIGMA**2 / 2.0) * DT + SIGMA * np.sqrt(DT))
    z_dn = z * np.exp((MU - SIGMA**2 / 2.0) * DT - SIGMA * np.sqrt(DT))

    return np.array([[z_up, z_dn]])


def density_operator(coherence):
    vector_up = np.array([[1/np.sqrt(2)], [0], [1/np.sqrt(2)], [0]]).reshape(-1, 1) # Shape: (4,1)
    vector_dn = np.array([[0], [1/np.sqrt(2)], [0], [1/np.sqrt(2)]]).reshape(-1, 1) # Shape: (4,1)
    matrix_up = vector_up @ vector_up.T
    matrix_dn = vector_dn @ vector_dn.T
    matrix_tot = 0.5 * matrix_up + 0.5 * matrix_dn
    density_op = (matrix_tot 
                  - ( 1 - coherence) * (matrix_tot - np.diag(matrix_tot))) # Shape: (4,4)
    
    return density_op


def price_approximation_clean(A, omega, z):
    omega = omega.flatten()
    z = z.flatten()

    omega_1 = np.tile(omega, 2) # up, down, up, down
    z_1 = np.repeat(z, 2) # up, up, down, down
    one = np.ones_like(omega_1) * np.exp(1)
    states = np.vstack([one, omega_1, z_1])
    log_states = np.log(states) 
    log_prices = A[0:3].reshape(1, 3) @ log_states  # (1, n)

    return np.exp(log_prices)


# ================================================================
# Restrictions:
# ================================================================


def budget_constraint(controls, states, Approx):

    C, AD_1, AD_2, ALPHA_1, ALPHA_2 = unpack_controls(controls)
    omega, z, ad_1, ad_2, alpha_1, alpha_2 = unpack_states(states)

    omega_new = evolve_omega(omega)
    z_new = evolve_z(z)

    P_1 = price_approximation_clean(Approx[0:3], omega_new, z_new)
    P_2 = price_approximation_clean(Approx[3:6], omega_new, z_new)
    Q_1H = price_approximation_clean(Approx[6:9], omega_new, z_new)
    Q_1L = price_approximation_clean(Approx[9:12], omega_new, z_new)
    Q_2H = price_approximation_clean(Approx[12:15], omega_new, z_new)
    Q_2L = price_approximation_clean(Approx[15:18], omega_new, z_new)

    ad_2flat = np.tile(ad_2.flatten(), 2).reshape(-1,1) # up, down, up, down (omega)
    ad_1flat = np.repeat(ad_1.flatten(), 2).reshape(-1,1) # up, up, down, down (z)
    ad = ad_1flat + ad_2flat

    budget = (C * DT + P_1 @ ALPHA_1 + P_2 @ ALPHA_2 + Q_1H @ AD_1[:, 0] 
              + Q_1L @ AD_1[:, 1] - Q_2H @ AD_2[:, 0] - Q_2L @ AD_2[:, 1]
              - (P_1 @ alpha_1 + P_2 @ alpha_2) * DT - ad)

    return -budget


def laws_of_motion(controls, state):
    _, AD_1, AD_2, ALPHA_1, ALPHA_2 = unpack_controls(controls)
    omega, z, _, _, _, _, coherence = unpack_states(state)
    omega_new = evolve_omega(omega)
    z_new = evolve_z(z)
    coherence = coherence * np.exp(-DECOH * DT) * np.ones(4).reshape(-1,1)

    omega_1 = (np.tile(omega_new, 2)).reshape(-1,1) # up, down, up, down
    z_1 = (np.repeat(z_new, 2)).reshape(-1.1) # up, up, down, down
    exogenous = np.vstack([omega_1, z_1])
    alpha_1 = ALPHA_1 * np.ones(4).reshape(-1,1)
    alpha_2 = ALPHA_2 * np.ones(4).reshape(-1,1)

    states = np.hstack([exogenous, AD_1, AD_2, alpha_1, alpha_2, coherence])

    return states


# ================================================================
# Agent's problem:
# ================================================================


def utility_matrix(C):

    diagonal = np.diag(C**(1 - GAMMA) / (1 - GAMMA))
    off_diagonal = C - C.T
    matrix = KAPPA * np.exp(-off_diagonal**2) + diagonal

    return matrix


def utility(controls, state):
    _, _, _, _, _, _, coherence = unpack_states(state)

    C, _ = unpack_controls(controls)
    U = utility_matrix(C)
    rho_C = density_operator(coherence)

    return np.trace(U @ rho_C)


def expected_V(V, states_new):
    
    return (0.25 * V(states_new[1,:]) + 0.25 * V(states_new[2,:])
            + 0.25 * V(states_new[3,:]) + 0.25 * V(states_new[4,:]))


def objective(controls, state, V):

    states_new = laws_of_motion(controls, state)

    objective = utility(controls, state) * DT + expected_V(V, states_new)

    return -objective


def agent_optimization(Approx, V, states):

    constraints = [
        {'type': 'ineq',   'fun': budget_constraint,
            'args': (states, Approx)
            }
        ]
    
    initial_guess = np.ones(28)

    bounds = [(0.05, 4)] + [(-2, 2)] * 27

    result = minimize(
            objective,
            initial_guess,
            args=(states, V),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            )

    return result


# ================================================================
# Value function and price iteration:
# ================================================================


def value_function_iteration(Approx, V, grid):

    solution = np.zeros((*V.shape, 28))

    V_old = V + 1.0  # Initialize V_old to be different from V_0 to start the iteration
    state = np.zeros(9)

    V_old = V.copy()

    V_interpolator = RegularGridInterpolator(
        (grid),  # Tuple of all grid coordinates
        V,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Update V using the agent's optimization problem:
    for idx in np.ndindex(V.shape):
        state = grid[idx]
        result = agent_optimization(Approx, V_interpolator, state)
        V[idx] = ((1 - UPDATE_SPEED) * V_old[idx]      # was V_old (full array)
                    + UPDATE_SPEED * (-result.fun / DELTA)) 
        solution[idx, :] = result.x  # Store the optimal controls and parameters for this state

    return V, solution


def excess_demand(Approx, V, grid):
    V_new, solution = value_function_iteration(Approx, V, grid)
    