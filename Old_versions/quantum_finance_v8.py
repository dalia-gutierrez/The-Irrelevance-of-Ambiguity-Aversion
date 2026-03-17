from unittest import result
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
KAPPA = 0.5 # Ambiguity aversion
GAMMA = 2.0 # Risk aversion
PI = np.pi  # Pi constant
DELTA = 0.05  # Discount rate
UPDATE_SPEED = 0.1  # Speed of updating the value function
J = 5 # Number of agents
AD_1_SUPPLY = 0.0 # Supply of Arrow security 1
AD_2_SUPPLY = 0.0 # Supply of Arrow security 2
BAR_ALPHA_1 = J # Supply of asset 1
BAR_ALPHA_2 = J # Supply of asset 2


# ================================================================
# Fixed arrays:
# ================================================================


# Variable layout (all moments are observable E[X] and Cov(X_i,X_j)):
#   0=C, 1=AD_1, 2=AD_2, 3=A_1, 4=A_2  — controls
#   5=Q_1, 6=Q_2, 7=P_1, 8=P_2         — prices (exogenous)
#   9=Y                                — dividend income (exogenous)
#
# State variables (x): x[0]=AD_1, x[1]=AD_2, x[2]=A_1, x[3]=A_2
# Consumption is a control but NOT a state — it does not enter V derivatives.


# ================================================================
# Numerical approximations:
# ================================================================


def shift_index(idx, axis, step):

    # Shift the index along the specified axis by the given step
    idx = list(idx)
    idx[axis] += step

    return tuple(idx)


def first_derivatives_point(V, idx, omega_grid, x_grids):

    # Calculate the first derivatives of the value function V
    # at a given index idx using finite differences

    grids = [omega_grid] + x_grids
    dims = [len(g) for g in grids]

    d = len(x_grids)

    grad = np.zeros(d+1)

    for k in range(d+1):

        h = grids[k][1] - grids[k][0]
        pos = idx[k]

        idx_list = list(idx)

        if pos == 0:  # forward difference

            idx_p = list(idx); idx_p[k] += 1

            grad[k] = (V[tuple(idx_p)] - V[idx]) / h

        elif pos == dims[k]-1:  # backward difference

            idx_m = list(idx); idx_m[k] -= 1

            grad[k] = (V[idx] - V[tuple(idx_m)]) / h

        else:  # central difference

            idx_p = list(idx); idx_p[k] += 1
            idx_m = list(idx); idx_m[k] -= 1

            grad[k] = (V[tuple(idx_p)] - V[tuple(idx_m)]) / (2*h)

    return grad[0], grad[1:]


def second_derivative_point(V, idx, grids):

    # Calculate the second derivative of the value function V at a given index idx

    d = len(grids) - 1   # number of x variables

    # ---------- V_{ωω} ----------
    
    omega_grid = grids[0]
    h = omega_grid[1] - omega_grid[0]
    pos = idx[0]
    n = len(omega_grid)

    if pos == 0:

        i0 = list(idx)
        i1 = list(idx); i1[0]+=1
        i2 = list(idx); i2[0]+=2

        V_oo = (V[tuple(i2)] - 2*V[tuple(i1)] + V[tuple(i0)])/(h*h)

    elif pos == n-1:

        i0 = list(idx)
        i1 = list(idx); i1[0]-=1
        i2 = list(idx); i2[0]-=2

        V_oo = (V[tuple(i0)] - 2*V[tuple(i1)] + V[tuple(i2)])/(h*h)

    else:

        ip = list(idx); ip[0]+=1
        im = list(idx); im[0]-=1

        V_oo = (V[tuple(ip)] - 2*V[idx] + V[tuple(im)])/(h*h)


    # ---------- Hessian in x variables ----------

    H = np.zeros((d,d))

    for i in range(d):

        axis_i = i+1
        hi = grids[axis_i][1] - grids[axis_i][0]
        ni = len(grids[axis_i])
        pos_i = idx[axis_i]

        # diagonal terms V_{x_i x_i} with boundary guards
        if pos_i == 0:
            i0 = list(idx)
            i1 = list(idx); i1[axis_i] += 1
            i2 = list(idx); i2[axis_i] += 2
            H[i,i] = (V[tuple(i2)] - 2*V[tuple(i1)] + V[tuple(i0)]) / (hi*hi)
        elif pos_i == ni-1:
            i0 = list(idx)
            i1 = list(idx); i1[axis_i] -= 1
            i2 = list(idx); i2[axis_i] -= 2
            H[i,i] = (V[tuple(i0)] - 2*V[tuple(i1)] + V[tuple(i2)]) / (hi*hi)
        else:
            ip = list(idx); ip[axis_i] += 1
            im = list(idx); im[axis_i] -= 1
            H[i,i] = (V[tuple(ip)] - 2*V[idx] + V[tuple(im)]) / (hi*hi)

        # cross derivatives
        for j in range(i+1,d):

            axis_j = j+1
            hj = grids[axis_j][1] - grids[axis_j][0]

            ipp = list(idx); ipp[axis_i]+=1; ipp[axis_j]+=1
            ipm = list(idx); ipm[axis_i]+=1; ipm[axis_j]-=1
            imp = list(idx); imp[axis_i]-=1; imp[axis_j]+=1
            imm = list(idx); imm[axis_i]-=1; imm[axis_j]-=1

            val = (V[tuple(ipp)] - V[tuple(ipm)] - V[tuple(imp)] + V[tuple(imm)])/(4*hi*hj)

            H[i,j] = val
            H[j,i] = val

    return V_oo, H


# ================================================================
# Calculation of stochastic process moments:
# ================================================================


def calculate_drifts(means, x_state):
    return (means - x_state) / DT


def calculate_volatilities(means_obs, covs_obs):
    """
    Compute the instantaneous diffusion covariance Σ/dt for the HJB.

    Receives observable means E[X] and covariances Cov(X_i,X_j) directly.
    Returns covs_obs / DT — the instantaneous diffusion matrix for the HJB.
    No type-specific formulas needed: distribution type only matters when
    converting from a parameterisation to observable moments, which is done
    once in reconstruct_covs() before this function is ever called.
    """
    return covs_obs / DT


# ================================================================
# Helper functions:
# ================================================================


def build_covariance(params, n=5):
    """
    Cholesky decomposition to ensure positive semi-definiteness.
    params: lower-triangular entries of L, so Sigma = L @ L.T.
    Works identically whether Sigma is in log-space or observable space —
    the caller decides which space it operates in.
    """
    L = np.zeros((n, n))
    L[np.tril_indices(n)] = params
    return L @ L.T


def reconstruct_means(controls, means_prices, mean_Y):
    """
    Assemble the full 10-vector of observable means E[X].

    CHANGED (v9): controls[0:5] are now the observable means E[X] directly —
    E[C], E[AD_1], E[AD_2], E[A_1], E[A_2].  No log-space conversion needed.
    means_prices and mean_Y are also observable E[P] and E[Y].

    Variable layout:
      0=C, 1=AD_1, 2=AD_2, 3=A_1, 4=A_2,
      5=Q_1, 6=Q_2, 7=P_1, 8=P_2, 9=Y
    """
    means = np.zeros(10)
    means[0:5] = controls[0:5]   # observable E[X] for controls
    means[5:9] = means_prices    # observable E[P]
    means[9]   = mean_Y          # observable E[Y]
    return means


def reconstruct_covs(controls, covs_prices, var_Y):
    """
    Assemble the full 10×10 observable covariance matrix Cov(X_i, X_j).

    CHANGED (v9): all entries are now observable covariances directly.
      controls[5:20]  – 15 Cholesky entries for the 5×5 observable Σ_cc
                        (self-covariance of controls).
      controls[20:45] – 25 free entries for the 5×5 observable cross-covariance
                        Σ_cp = Cov(controls, prices/Y).
    covs_prices: 4×4 observable covariance matrix of prices Cov(P_i, P_j).
    var_Y:       scalar observable variance Var(Y).

    No log-space parameters or type-specific formulas appear here.
    """
    covs = np.zeros((10, 10))

    # exogenous blocks
    covs[5:9, 5:9] = covs_prices
    covs[9,   9  ] = var_Y

    # control self-covariance (Cholesky in observable space)
    covs[0:5, 0:5] = build_covariance(controls[5:20], 5)

    # cross-covariance controls × prices/Y (free, observable)
    Sigma_cp = controls[20:45].reshape(5, 5)
    covs[0:5, 5:10] = Sigma_cp
    covs[5:10, 0:5] = Sigma_cp.T

    return covs


def calculate_expectation_product(E_X, E_Y, E_Z, cov_XY, cov_XZ, cov_YZ):
     
    # Calculate E(XYZ) where X, Y are lognormal and Z normal

    E_XYZ = (cov_XY + E_XY) * (E_Z + cov_XZ / E_X + cov_YZ / E_Y)

    return E_XYZ


def calculate_variance_product(var_X, var_Z, cov_XZ, E_X, E_Z):

    # Calculate Var(XZ) where X is lognomal and Z normal

    var_XZ = (E_X**2 * var_Z + var_X * E_Z**2 + 2 * E_X * cov_XZ
              + var_X * var_Z + cov_XZ**2)

    return var_XZ


def calculate_variance_sum_and_product(var_X, var_Y, var_Z, cov_XY, cov_XZ, cov_YZ, E_X, E_Y, E_Z):

    # Calculate Var(X + YZ) where X, Y are lognormal and Z normal

    var_YZ = calculate_variance_product(var_Y, var_Z, cov_YZ, E_Y, E_Z)
    E_X_YZ = calculate_expectation_product(E_X, E_Y, E_Z, cov_XY, cov_XZ, cov_YZ)
    cov_X_YZ = E_X_YZ - E_X * E_Y * E_Z
    var_X_YZ = var_X + var_YZ + 2 * cov_X_YZ

    return var_X_YZ


# ================================================================
# Agent's restrictions:
# ================================================================


def calculate_Y_moments(omega, z, x_state):
    """
    Returns observable E[Y] and Var(Y).

    Y = A_1 * omega_{t+dt} + A_2 * z_{t+dt} is approximated by a 4-point
    equally-weighted discrete distribution over (omega_up/dn, z_up/dn).
    E[Y] and Var(Y) are computed directly from the 4 outcome values —
    no log-space intermediate is exposed outside this function.
    """
    alpha_1 = x_state[2]   # A_1
    alpha_2 = x_state[3]   # A_2

    omega_up = omega * np.exp((MU - SIGMA**2 / 2.0) * DT + SIGMA * np.sqrt(DT))
    omega_dn = omega * np.exp((MU - SIGMA**2 / 2.0) * DT - SIGMA * np.sqrt(DT))
    z_up     = z     * np.exp((MU - SIGMA**2 / 2.0) * DT + SIGMA * np.sqrt(DT))
    z_dn     = z     * np.exp((MU - SIGMA**2 / 2.0) * DT - SIGMA * np.sqrt(DT))

    outcomes = np.array([
        alpha_1 * omega_up + alpha_2 * z_up,
        alpha_1 * omega_dn + alpha_2 * z_up,
        alpha_1 * omega_up + alpha_2 * z_dn,
        alpha_1 * omega_dn + alpha_2 * z_dn,
    ])

    mean_Y = 0.25 * np.sum(outcomes)
    var_Y  = 0.25 * np.sum(outcomes**2) - mean_Y**2

    return mean_Y, var_Y


def budget_constraint(controls, x_state, means_prices, mean_Y, covs_prices, var_Y):
    """
    Budget constraint in observable means and covariances throughout.

    Tr[rho * (C*dt + A_1*P_1 + A_2*P_2 + AD_1*Q_1 + AD_2*Q_2
              - Y*dt - P_1*x_A1 - P_2*x_A2 - x_AD1 - x_AD2)] = 0

    Using E[X_i * X_j] = Cov(X_i, X_j) + E[X_i]*E[X_j] throughout.
    All arguments are observable quantities.

    Variable indices:
      0=C, 1=AD_1, 2=AD_2, 3=A_1, 4=A_2,
      5=Q_1, 6=Q_2, 7=P_1, 8=P_2, 9=Y
    """
    means = reconstruct_means(controls, means_prices, mean_Y)
    covs  = reconstruct_covs(controls, covs_prices, var_Y)

    budget = (means[0] * DT                              # E[C]*dt
              + covs[3, 7] + means[3] * means[7]         # E[A_1 * P_1]
              + covs[4, 8] + means[4] * means[8]         # E[A_2 * P_2]
              + covs[1, 5] + means[1] * means[5]         # E[AD_1 * Q_1]
              + covs[2, 6] + means[2] * means[6]         # E[AD_2 * Q_2]
              - means[9] * DT                            # -E[Y]*dt
              - means[7] * x_state[2]                   # -E[P_1]*x_A1
              - means[8] * x_state[3]                   # -E[P_2]*x_A2
              - x_state[0]                              # -E[x_AD1]
              - x_state[1])                             # -E[x_AD2]
    return -budget


def budget_constraint_zero_variance(controls, x_state, means_prices, mean_Y, covs_prices, var_Y):

    means = reconstruct_means(controls, means_prices, mean_Y)
    covs  = reconstruct_covs(controls, covs_prices, var_Y)

    budget = (covs[0,0] * DT**2                                 # (Var(C)*dt *
              + 



def interference_constraint(controls):
    """
    Feasibility constraint on the interference term for consumption C.

    CHANGED (v9): controls[0] is now observable E[C] directly, and
    covs[0,0] = Var(C) is obtained from the Cholesky block.  No conversion
    from log-space is needed.
    """
    E_C   = controls[0]                              # observable E[C]
    Var_C = build_covariance(controls[5:20], 5)[0, 0]  # observable Var(C)
    interference_C = controls[45]

    max_interference = (interference_C + 1
                        - 2 * np.sqrt(PI) * np.sqrt(Var_C)
                        * np.exp((E_C + Var_C / 2.0) / 2.0))

    return -max_interference


# ================================================================
# Agent's problem:
# ================================================================


def utility_function(E_C, Var_C, interference_C):
    """
    Expected CRRA utility of lognormal consumption C plus ambiguity term.

    Arguments are observable E[C] and Var(C).  The CRRA formula requires the
    log-space parameters (mu_C, sigma2_C) of the lognormal, so we recover them
    here from the observable moments — this is the only place log-space
    parameters appear, and they never leave this function.

        sigma2_C = log(1 + Var(C) / E[C]^2)
        mu_C     = log(E[C]) - 0.5 * sigma2_C
        E[C^(1-gamma)] / (1-gamma) = exp((1-gamma)*mu_C + 0.5*(1-gamma)^2*sigma2_C)
                                     / (1-gamma)
    """
    sigma2_C = np.log(1.0 + Var_C / (E_C**2 + 1e-12))
    mu_C     = np.log(E_C  + 1e-12) - 0.5 * sigma2_C

    return (np.exp((1 - GAMMA) * mu_C + 0.5 * (1 - GAMMA)**2 * sigma2_C) / (1 - GAMMA)
            + KAPPA * interference_C)


def calculate_terms_of_objective(V, idx, x_grids, omega_grid):

    omega   = omega_grid[idx[0]]
    x_state = [x_grids[i][idx[i + 2]] for i in range(len(x_grids))]

    V_omega, V_x         = first_derivatives_point(V, idx, omega_grid, x_grids)
    V_omegaomega, V_xx   = second_derivative_point(V, idx, [omega_grid] + x_grids)

    return omega, x_state, V_omega, V_x, V_omegaomega, V_xx


def objective_function(controls, omega, x_state, V_omega, V_x, V_omegaomega, V_xx,
                       means_prices, covs_prices, var_Y):
    """
    HJB objective. All quantities are observable throughout.

    controls[0:5]  = observable means  E[C], E[AD_1], E[AD_2], E[A_1], E[A_2]
    controls[5:20] = Cholesky entries for observable Sigma_cc (5x5)
    controls[20:45]= observable cross-covariances Sigma_cp (5x5 flat)
    controls[45]   = interference term

    Consumption (controls[0]) is a control but NOT a state variable — it does
    not enter drifts or volatilities multiplying V_x / V_xx.
    Only AD_1, AD_2, A_1, A_2 (controls[1:5]) are state variables.
    """
    covs = reconstruct_covs(controls, covs_prices, var_Y)
    interference_C = controls[45]

    E_C   = controls[0]
    Var_C = covs[0, 0]

    # state variables: observable means and covariances of AD_1, AD_2, A_1, A_2
    means_states = controls[1:5]                  # already observable E[X]
    covs_states  = covs[1:5, 1:5]                 # observable Cov block

    drifts       = calculate_drifts(means_states, x_state)
    volatilities = calculate_volatilities(means_states, covs_states)

    objective_func = (utility_function(E_C, Var_C, interference_C)
                      + MU * omega * V_omega
                      + 0.5 * SIGMA**2 * omega**2 * V_omegaomega
                      + np.dot(drifts, V_x)
                      + 0.5 * np.sum(volatilities * V_xx))

    return -objective_func


def agent_optimization(means_prices, covs_prices, idx, x_grids, omega_grid, z_grid, V):
    """
    Solve the agent's HJB problem at grid point idx.
    means_prices, covs_prices: observable E[P] and Cov(P_i,P_j).
    """
    omega, x_state, V_omega, V_x, V_omegaomega, V_xx = calculate_terms_of_objective(
        V, idx, x_grids, omega_grid)
    z = z_grid[idx[1]]

    mean_Y, var_Y = calculate_Y_moments(omega, z, x_state)

    constraints = [
        {'type': 'ineq', 'fun': interference_constraint},
        {'type': 'ineq', 'fun': budget_constraint,
         'args': (x_state, means_prices, mean_Y, covs_prices, var_Y)},
    ]

    initial_guess = np.concatenate((
        np.ones(5) * 0.5,    # 5 observable means E[X]
        np.ones(15) * 0.01,  # 15 Cholesky entries for observable Sigma_cc
        np.zeros(25),        # 25 observable cross-covariances Sigma_cp
        np.array([0.0]),     # 1 interference term
    ))

    # bounds: means > 0 for lognormal variables (C, A_2); relax to (-2,2) for all
    bounds = [(-2, 2)] * 5 + [(-3, 3)] * 40 + [(-1, None)]

    result = minimize(
        objective_function,
        initial_guess,
        args=(omega, x_state, V_omega, V_x, V_omegaomega, V_xx,
              means_prices, covs_prices, var_Y),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    return result


# ================================================================
# Value function iteration:
# ================================================================


def value_function_iteration(prices, V, x_grids, omega_grid, z_grid):

    means_prices = prices[0:4]
    covs_prices = build_covariance(prices[4:], 4)

    solution = np.zeros(V.size, 46)

    V_old = V + 1.0  # Initialize V_old to be different from V_0 to start the iteration
    for iteration in range(100):

        if np.max(np.abs(V - V_old)) < 1e-3:
            print(f"Convergence achieved after {iteration} iterations.")
            break

        print(f"Iteration {iteration}")
        print(f"Max change in V: {np.max(np.abs(V - V_old))}")

        V_old = V.copy()

        # Update V using the agent's optimization problem:
        for idx in np.ndindex(V.shape):
            result = agent_optimization(means_prices, covs_prices, idx, 
                                        x_grids, omega_grid, z_grid, V)
            V[idx] = ((1 - UPDATE_SPEED) * V_old[idx]      # was V_old (full array)
                      + UPDATE_SPEED * (-result.fun / DELTA)) 
            solution[idx, :] = result.x  # Store the optimal controls and parameters for this state

    return V, solution


# ================================================================
# Market clearing:
# ================================================================


class MarketSolver:


    def __init__(self, V):
        self.V = V
        self.prices = None
        self.sol_means = None
        self.sol_covs = None


    def market_residuals(self, prices, V, x_grids, omega_grid, z_grid, omega_t, z_t, x_states_t):
        self.prices = prices
        excess_demand, self.V, self.sol_means, self.sol_covs = self.market_solution(prices, V, x_grids, omega_grid, z_grid, omega_t, z_t, x_states_t)
        return excess_demand


    def market_solution(self, prices, V, x_grids, omega_grid, z_grid, omega_t, z_t, x_states_t):
        V_new, solution = value_function_iteration(prices, V, x_grids, omega_grid, z_grid)

        # Calculate market residuals based on the updated value function and agent's decisions

        means_prices = prices[0:4]
        covs_prices = build_covariance(prices[4:], 4)

        grid = [omega_grid, z_grid] + x_grids  # was [omega_grid, z_grid, x_grids] — x_grids must be unpacked

        interpolator = RegularGridInterpolator(
            (omega_grid, z_grid, *x_grids),  # Tuple of all grid coordinates
            solution,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        sol = np.zeros((J, 46))  # Store the optimal controls and parameters for each agent
        sol_means = np.zeros((J, 10))
        sol_covs = np.zeros((J, 10, 10))
        Sigma_assets = np.zeros((J, 4, 4))
        covs_assets_prices = np.zeros((J, 4, 4))
        covs_assets_2_assets = np.zeros((J, 4, 4))

        for j in range(J):
            means_Y, covs_Y = calculate_Y_moments(omega_t, z_t, x_states_t[j])
            point = (omega_t, z_t, *x_states_t[j])
            sol[j, :] = interpolator(point)

            sol_means[j,:] = reconstruct_means(sol[j,:], means_prices, means_Y)  # Reconstruct full means vector
            sol_covs[j][:,:] = reconstruct_covs(sol[j,:], covs_prices, covs_Y)  # Reconstruct full covariance matrix
            Sigma_assets[j][:,:] = sol_covs[j][1:5, 1:5]
            covs_assets_prices[j][:,:] = sol_covs[j][1:5, 5:9]
            covs_assets_2_assets[j][:,:] = covs_assets_prices[j][:,:] @ np.linalg.inv(covs_prices) @ covs_assets_prices[j][:,:]

        total_cov_assets = np.sum(Sigma_assets, axis=0)
        total_cov_assets_prices = np.sum(covs_assets_prices, axis=0)
        total_covs_assets_2_assets = np.sum(covs_assets_2_assets, axis=0)
        total_means = np.sum(sol_means, axis=0)

        market_clearing_average = total_means - np.array([AD_1_SUPPLY, AD_2_SUPPLY, BAR_ALPHA_1, BAR_ALPHA_2])
        market_clearing_variance = (total_cov_assets - total_covs_assets_2_assets 
                                    + total_cov_assets_prices @ np.linalg.inv(covs_prices) @ total_cov_assets_prices)

        excess_demand = (np.linalg.norm(market_clearing_average)**2 
                         + np.linalg.nrom(np.diag(market_clearing_variance))**2)
    
        return excess_demand, V_new, sol_means, sol_covs


    def solve_market(self, initial_prices, V, x_grids, omega_grid, z_grid, omega_t, z_t, x_states_t):
        result = minimize(
            self.market_residuals,
            initial_prices,
            args=(V, x_grids, omega_grid, z_grid, omega_t, z_t, x_states_t),
            method="Nelder-Mead",
            options={'maxiter': 50}
        )
        return result


def equilibrium_solver():
    means_prices = np.array([1.0, 1.0, 1.0, 1.0])   # observable E[P]
    covs_prices  = np.eye(4) * 0.01                  # observable Cov(P_i, P_j)
 
    x_grids    = [np.linspace(0.5, 1.5, 5) for _ in range(4)]  # AD_1, AD_2, A_1, A_2
    z_grid     = np.linspace(0.5, 1.5, 5)
    omega_grid = np.linspace(0.5, 1.5, 5)
 
    V   = np.zeros((len(omega_grid), len(z_grid), *[len(g) for g in x_grids]))
    idx = (0, len(z_grid) // 2) + tuple(len(g) // 2 for g in x_grids)
 
    result = agent_optimization(means_prices, covs_prices, idx,
                                x_grids, omega_grid, z_grid, V)
    print("Result:", result.x)
 
 
if __name__ == "__main__":
    equilibrium_solver()
