from calendar import c
import re
import types
from xml.etree.ElementTree import PI
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, lognorm
from scipy.interpolate import RegularGridInterpolator


# ================================================================
# Fixed paramters:
# ================================================================


DT = 0.01   # Time step
MU = 0.00   # Drift
SIGMA = 0.2 # Volatility
KAPPA = 0.5 # Ambiguity aversion
GAMMA = 2.0 # Risk aversion
PI = np.pi  # Pi constant
DELTA = 0.05  # Discount rate
UPDATE_SPEED = 0.1  # Speed of updating the value function
J = 5.0 # Number of agents
AD_1_SUPPLY = 0.0 # Supply of Arrow security 1
AD_2_SUPPLY = 0.0 # Supply of Arrow security 2
BAR_ALPHA_1 = J # Supply of asset 1
BAR_ALPHA_2 = J # Supply of asset 2


# ================================================================
# Fixed arrays:
# ================================================================


# Set the types of the distributions for the 8 variables
TYPES = ["lognormal", "normal", "normal", "normal", "lognormal", "lognormal", "lognormal", "lognormal"]


# x[0] = consumption, x[1] = AD_1, x[2] = AD_2, x[3] = A_1, x[4] = A_2


# ================================================================
# Numerical approximations:
# ================================================================


def shift_index(idx, axis, step):

    # Shift the index along the specified axis by the given step
    idx = list(idx)
    idx[axis] += step

    return tuple(idx)


import numpy as np

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

        # diagonal terms V_{x_i x_i}
        ip = list(idx); ip[axis_i]+=1
        im = list(idx); im[axis_i]-=1

        H[i,i] = (V[tuple(ip)] - 2*V[idx] + V[tuple(im)])/(hi*hi)

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


def calculate_volatilities(means, covs):

    # Calculate the volatilities (covariance matrix) of the stochastic processes
    # using the mu and sigmas of x_{t+dt}

    n = len(means)
    E = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i, n):
            
            ti = TYPES[i]
            tj = TYPES[j]
            
            if ti == "normal" and tj == "normal":
                E[i,j] = means[i]*means[j] + covs[i,j]
            elif ti == "lognormal" and tj == "lognormal":
                E[i,j] = np.exp(
                    means[i] + means[j] 
                    + 0.5*(covs[i,i] + covs[j,j]) 
                    + covs[i,j]
                )
            elif ti == "normal" and tj == "lognormal":
                E[i,j] = np.exp(means[j] + 0.5*covs[j,j]) * (means[i] + covs[i,j])            
            elif ti == "lognormal" and tj == "normal":
                E[i,j] = np.exp(means[i] + 0.5*covs[i,i]) * (means[j] + covs[i,j])

            E[j,i] = E[i,j]
    
    Sigma = (E - np.outer(means, means)) / DT

    return Sigma


# ================================================================
# Helper functions:
# ================================================================


def build_covariance(params, n=5):

    # Cholesky decomposition to ensure positive semi-definiteness of the covariance matrix

    L = np.zeros((n,n))
    idx = np.tril_indices(n)

    L[idx] = params
    Sigma = L @ L.T

    return Sigma


def reconstruct_means(controls, means_prices, means_Y):

    means = np.zeros(10)

    means[0:5] = controls[0:5]       # free variables
    means[5:9] = means_prices        # fixed prices
    means[9]   = means_Y             # fixed income

    return means


def reconstruct_covs(controls, covs_prices, covs_Y):

    covs = np.zeros((10,10))
    size = len(controls) - 1

    chol_params = controls[5:size]
    covs[0:5,0:5] = build_covariance(chol_params)
    covs[5:9,5:9] = covs_prices
    covs[9,9] = covs_Y

    return covs


# ================================================================
# Agent's restrictions:
# ================================================================


def calculate_Y_moments(omega, z, x_state):

    # Calculate dividend income moments of future income distribution by approximating
    # GBM with a discrete distribution with 4 outcomes (up-up, up-down, down-up, down-down)

    alpha_1 = x_state[3]  # Investment in asset 1
    alpha_2 = x_state[4]  # Investment in asset 2

    omega_1 = omega * np.exp((MU - SIGMA**2 / 2.0) * DT + SIGMA * np.sqrt(DT))
    omega_2 = omega * np.exp((MU - SIGMA**2 / 2.0) * DT - SIGMA * np.sqrt(DT))
    z_1 = z * np.exp((MU - SIGMA**2 / 2.0) * DT + SIGMA * np.sqrt(DT))
    z_2 = z * np.exp((MU - SIGMA**2 / 2.0) * DT - SIGMA * np.sqrt(DT))

    means_Y = (0.25 * np.log(alpha_1 * omega_1 + alpha_2 * z_1)
               + 0.25 * np.log(alpha_1 * omega_2 + alpha_2 * z_1)
                + 0.25 * np.log(alpha_1 * omega_1 + alpha_2 * z_2)
                + 0.25 * np.log(alpha_1 * omega_2 + alpha_2 * z_2))
    covs_Y = (0.25 * np.log(alpha_1 * omega_1 + alpha_2 * z_1)**2
               + 0.25 * np.log(alpha_1 * omega_2 + alpha_2 * z_1)**2
                + 0.25 * np.log(alpha_1 * omega_1 + alpha_2 * z_2)**2
                + 0.25 * np.log(alpha_1 * omega_2 + alpha_2 * z_2)**2) - means_Y**2

    return means_Y, covs_Y


def budget_constraint(controls, x_state, means_prices, means_Y, covs_prices, covs_Y):

    # Calculate the budget constraint using the means and covariances of the distributions
    # Term by term we have:
    # Tr[rho_t (C * dt + P_1 alpha_1 + P_2 alpha_2 + Q_1 A_1 + Q_2 A_2 - Y * dt
    # - P_1 x_{alpha_1} - P_2 x_{alpha_2} - I otimes M_|x_{a}})]

    # 0 = consumption, 1 = AD_1, 2 = AD_2, 3 = A_1, 4 = A_2, 
    # 5 = Q_1, 6 = Q_2, 7 = P_1, 8 = P_2, 9 = Y

    means = reconstruct_means(controls, means_prices, means_Y)
    covs = reconstruct_covs(controls, covs_prices, covs_Y)

    budget = (np.exp(means[0] + covs[0, 0] / 2.0) * DT 
              + (means[3] + covs[3, 7])*np.exp(means[7] + covs[7, 7] / 2.0)
              + (means[4] + covs[4, 8])*np.exp(means[8] + covs[8, 8] / 2.0)
              + (means[1] + covs[1, 5])*np.exp(means[5] + covs[5, 5] / 2.0)
              + (means[2] + covs[2, 6])*np.exp(means[6] + covs[6, 6] / 2.0)
              - np.exp(means[9] + covs[9, 9] / 2.0) * DT 
              - np.exp(means[7] + covs[7, 7] / 2.0) * x_state[3]
              - np.exp(means[8] + covs[8, 8] / 2.0) * x_state[4]
              - x_state[1] - x_state[2])
    return -budget


def interference_constraint(controls):

    # Returns maximum interference feasible given the means and covariances 
    # of the consumption distribution

    means_C = controls[0]
    covs_C = controls[5]
    interference_C = controls[len(controls) - 1]

    max_interference = (interference_C + 1 - 
                        2 * np.sqrt(PI) * np.sqrt(covs_C) 
                        * np.exp((means_C + covs_C / 2.0) / 2.0))

    return -max_interference


# ================================================================
# Agent's problem:
# ================================================================


def utility_function(means_C, covs_C, interference_C):

    # Calculate expected utility using the means, covariances and interferences
    # of the consumption distribution

    expected_utility = (np.exp((1 - GAMMA) * means_C
                               + 0.5 * (1 - GAMMA)**2 + covs_C) / (1 - GAMMA)
                        + KAPPA * interference_C)

    return expected_utility


def calculate_terms_of_objective(V, idx, x_grids, omega_grid):
    
    omega = omega_grid[idx[0]]
    x_state = [x_grids[i][idx[i+2]] for i in range(len(x_grids))]

    V_omega, V_x = first_derivatives_point(V, idx, omega_grid, x_grids)
    V_omegaomega, V_xx = second_derivative_point(V, idx, [omega_grid] + x_grids)

    return omega, x_state, V_omega, V_x, V_omegaomega, V_xx


def objective_function(controls, omega, x_state, V_omega, V_x, V_omegaomega, V_xx,
                       covs_prices, covs_Y):

    covs = reconstruct_covs(controls, covs_prices, covs_Y)
    interference_C = controls[len(controls) - 1]

    drifts = calculate_drifts(controls[0:5], x_state)
    volatilities = calculate_volatilities(controls[0:5], covs[0:5,0:5])

    objective_func = (utility_function(controls[0], covs[0, 0], interference_C) 
                      + MU * omega * V_omega + 0.5 * SIGMA**2 * omega**2 * V_omegaomega
                      + np.dot(drifts, V_x) + 0.5 * np.sum(volatilities * V_xx))

    return -objective_func


def agent_optimization(means_prices, covs_prices, idx, x_grids, omega_grid, z_grid, V):

    # Agent optimizes their expected utility

    omega, x_state, V_omega, V_x, V_omegaomega, V_xx = calculate_terms_of_objective(V, idx, x_grids, omega_grid)
    z = z_grid[idx[1]]
    means_Y, covs_Y = calculate_Y_moments(z, omega, x_state)

    constraints = [
        {'type': 'ineq', 'fun': interference_constraint},
        {'type': 'ineq', 'fun': budget_constraint, 'args': (x_state,means_prices, means_Y, covs_prices, covs_Y)}
    ]

    initial_guess = np.concatenate((np.ones(5) * 0.5, np.ones(15) * 0.01, np.array([0.0])))

    bounds = bounds = [(-2, 2)]*5 + [(-3, 3)]*15 + [(-1, None)]

    result = minimize(
        objective_function,
        initial_guess,
        args = (omega, x_state, V_omega, V_x, V_omegaomega, V_xx, 
                covs_prices, covs_Y),
        bounds = bounds,
        constraints = constraints,
        method = "SLSQP"
    )

    return result


# ================================================================
# Value function iteration:
# ================================================================


def value_function_iteration(prices, V, x_grids, omega_grid, z_grid):

    means_prices = prices[0:4]
    covs_prices = build_covariance(prices[4:], 4)

    consumption_mu = np.zeros_like(V)
    AD_1_mean   = np.zeros_like(V)
    AD_2_mean  = np.zeros_like(V)
    alpha_1_mean  = np.zeros_like(V)
    alpha_2_mean  = np.zeros_like(V)

    consumption_sigma = np.zeros_like(V)
    AD_1_std   = np.zeros_like(V)
    AD_2_std  = np.zeros_like(V)
    alpha_1_std  = np.zeros_like(V)
    alpha_2_std  = np.zeros_like(V)

    Interference_mean = np.zeros_like(V)

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
            V[idx] = ((1 - UPDATE_SPEED) * V_old 
                      + UPDATE_SPEED * (-result.fun / DELTA)) 
            consumption_mu[idx] = result.x[0]
            AD_1_mean[idx]   = result.x[1]
            AD_2_mean[idx]  = result.x[2]
            alpha_1_mean[idx]  = result.x[3]
            alpha_2_mean[idx]  = result.x[4]

            covs = build_covariance(result.x[5:20], 5)

            consumption_sigma[idx] = covs[1]
            AD_1_std[idx]   = covs[2]
            AD_2_std[idx]  = covs[3]
            alpha_1_std[idx]  = covs[4]
            alpha_2_std[idx]  = covs[5]

            Interference_mean[idx] = result.x[20]

    return V, consumption_mu, AD_1_mean, AD_2_mean, alpha_1_mean, alpha_2_mean, consumption_sigma, AD_1_std, AD_2_std, alpha_1_std, alpha_2_std, Interference_mean


# ================================================================
# Market clearing:
# ================================================================


def market_residuals(prices, V, x_grids, omega_grid, z_grid, omega_t, z_t, x_states_t):
    V_new, consumption_mu, AD_1_mean, AD_2_mean, alpha_1_mean, alpha_2_mean, consumption_sigma, AD_1_std, AD_2_std, alpha_1_std, alpha_2_std, Interference_mean = value_function_iteration(prices, V, x_grids, omega_grid, z_grid)

    # Calculate market residuals based on the updated value function and agent's decisions

    AD_1_t = np.zeros(J)
    AD_2_t = np.zeros(J)
    alpha_1_t = np.zeros(J)
    alpha_2_t = np.zeros(J)
    std_AD_1_t = np.zeros(J)
    std_AD_2_t = np.zeros(J)
    std_alpha_1_t = np.zeros(J)
    std_alpha_2_t = np.zeros(J)

    grid = [omega_grid, z_grid, x_grids]

    AD_1_interp = RegularGridInterpolator(grid, AD_1_mean, bounds_error=False, fill_value=None)
    AD_2_interp = RegularGridInterpolator(grid, AD_2_mean, bounds_error=False, fill_value=None)
    alpha_1_interp = RegularGridInterpolator(grid, alpha_1_mean, bounds_error=False, fill_value=None)
    alpha_2_interp = RegularGridInterpolator(grid, alpha_2_mean, bounds_error=False, fill_value=None)
    std_AD_1_interp = RegularGridInterpolator(grid, AD_1_std, bounds_error=False, fill_value=None)
    std_AD_2_interp = RegularGridInterpolator(grid, AD_2_std, bounds_error=False, fill_value=None)
    std_alpha_1_interp = RegularGridInterpolator(grid, alpha_1_std, bounds_error=False, fill_value=None)
    std_alpha_2_interp = RegularGridInterpolator(grid, alpha_2_std, bounds_error=False, fill_value=None)

    for j in range(J):
        AD_1_t[j] = AD_1_interp((omega_t[j], z_t[j], *x_states_t[j]))
        AD_2_t[j] = AD_2_interp((omega_t[j], z_t[j], *x_states_t[j]))
        alpha_1_t[j] = alpha_1_interp((omega_t[j], z_t[j], *x_states_t[j]))
        alpha_2_t[j] = alpha_2_interp((omega_t[j], z_t[j], *x_states_t[j]))
        std_AD_1_t = std_AD_1_interp((omega_t[j], z_t[j], *x_states_t[j]))
        std_AD_2_t = std_AD_2_interp((omega_t[j], z_t[j], *x_states_t[j]))
        std_alpha_1_t = std_alpha_1_interp((omega_t[j], z_t[j], *x_states_t[j]))
        std_alpha_2_t = std_alpha_2_interp((omega_t[j], z_t[j], *x_states_t[j]))


    expected_excess_demand = np.array(np.sum(AD_1_t) - AD_1_SUPPLY, 
                                      np.sum(AD_2_t) - AD_2_SUPPLY, 
                                      np.sum(alpha_1_t) - BAR_ALPHA_1, 
                                      np.sum(alpha_2_t) - BAR_ALPHA_2)

    excess_std = std_AD_1_t + std_AD_2_t + std_alpha_1_t + std_alpha_2_t

    # WORK IN PROGRESS


def equilibrium_solver():
    means_prices = np.array([1.0, 1.0, 1.0, 1.0])  # Initial mean prices
    covs_prices = np.eye(4) * 0.01  # Initial covariance of prices

    x_grids = [np.linspace(0.5, 1.5, 5) for _ in range(5)]  # Grids for state variables
    z_grid = np.linspace(0.5, 1.5, 5)  # Grid for z
    omega_grid = np.linspace(0.5, 1.5, 5)  # Grid for omega

    V = np.zeros((len(omega_grid), len(z_grid), *[len(g) for g in x_grids]))  # Initial value function
    idx = (0, len(z_grid)//2) + tuple(len(g)//2 for g in x_grids)  # Initial index for optimization

    result = agent_optimization(means_prices, covs_prices, idx, x_grids, omega_grid, z_grid, V)

    print("Result:", result.x)


if __name__ == "__main__":
    equilibrium_solver()