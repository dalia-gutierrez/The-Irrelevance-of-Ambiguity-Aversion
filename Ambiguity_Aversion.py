import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from multiprocessing import Pool, cpu_count
import pickle
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.optimize._differentiable_functions")

# ================================================================
# Fixed parameters
# ================================================================

DT          = 0.1 # Time step
T_MAX       = 1.0 # Time Horizon
N_STEPS     = int(T_MAX / DT)

MU          = 0.02 # Average growth of dividends, expected consumption growth
SIGMA       = 0.04 # Dividend volatility, half consumption volatility
KAPPA       = 0.5 # Ambiguity aversion
GAMMA       = 1.5 # Risk aversion
DELTA       = 0.05 # Discount rate
DECOH       = 0.5 # Decoherence rate

J           = 3 # Agents
BAR_ALPHA_1 = float(J) # Shares of first firm
BAR_ALPHA_2 = float(J) # Shares of second firm

N_SAMPLES   = 200 # Monte Carlo samples
N_MC_STATES = 10 # Monte Calro states
N_AGENTS_PER_STATE = J

N_PRICE_PARAMS = 18 # Number of parameters for approximating prices
INITIAL_GUESS = np.zeros(N_PRICE_PARAMS) # Initial guess
INITIAL_GUESS[0] = -np.log(DELTA)
INITIAL_GUESS[1] = 1
INITIAL_GUESS[3] = -np.log(DELTA)
INITIAL_GUESS[5] = 1
INITIAL_GUESS[6] = np.log(0.5 - 0.5 * DELTA * DT)
INITIAL_GUESS[9] = np.log(0.5 - 0.5 * DELTA * DT)
INITIAL_GUESS[12] = np.log(0.5 - 0.5 * DELTA * DT)
INITIAL_GUESS[15] = np.log(0.5 - 0.5 * DELTA * DT)

INITIAL_V = np.ones(10) # Initial guess for linear Valur Function
INITIAL_V[3] = 1
INITIAL_V[4] = 1
INITIAL_V[5] = 1
INITIAL_V[6] = 1
INITIAL_V[7] = 1

# ================================================================
# Loading previous simulation
# ================================================================

def load_latest_checkpoint(save_dir):
    import glob
    files = glob.glob(os.path.join(save_dir, 'checkpoint_step_*.pkl'))
    if not files:
        print("No checkpoints found.")
        return None, None, -1, None
    
    latest_file = max(files, key=os.path.getmtime)
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Resuming from: {latest_file}")
    return data['theta_path'], data['V_interp'], data['step'], data['t']

# ================================================================
# Picklable value-function wrappers
# ================================================================

class ZeroValue:
    """Terminal value at t > T_MAX → 0 (picklable)."""
    def __call__(self, s):
        s = np.atleast_2d(s)
        return np.zeros(s.shape[0])

class ValueInterpolator:
    """Linear V(s) = beta0 + beta · state (including coherence) – fully picklable."""
    def __init__(self, beta):
        self.beta = beta
    def __call__(self, s):
        s = np.atleast_2d(s)
        X = np.column_stack([np.ones(s.shape[0]), s])
        return X @ self.beta

# ================================================================
# Unpackings
# ================================================================

def unpack_states(state):
    omega   = state[0]
    z       = state[1]
    ad_1    = state[2:4]
    ad_2    = state[4:6]
    alpha_1 = state[6]
    alpha_2 = state[7]
    return omega, z, ad_1, ad_2, alpha_1, alpha_2

def unpack_controls(controls):
    C      = controls[0:4].reshape(4, 1)
    AD_1   = np.column_stack([controls[4:8], controls[8:12]])
    AD_2   = np.column_stack([controls[12:16], controls[16:20]])
    ALPHA_1 = controls[20:24].reshape(4, 1)
    ALPHA_2 = controls[24:28].reshape(4, 1)
    return C, AD_1, AD_2, ALPHA_1, ALPHA_2

# ================================================================
# Laws of motion
# ================================================================

def evolve_omega(omega):
    base = (MU - 0.5 * SIGMA**2) * DT
    return np.array([omega * np.exp(base + SIGMA * np.sqrt(DT)),
                     omega * np.exp(base - SIGMA * np.sqrt(DT))])

def evolve_z(z):
    base = (MU - 0.5 * SIGMA**2) * DT
    return np.array([z * np.exp(base + SIGMA * np.sqrt(DT)),
                     z * np.exp(base - SIGMA * np.sqrt(DT))])

def next_period_nodes(omega, z):
    o = evolve_omega(omega)
    zv = evolve_z(z)
    return np.array([o[0], o[0], o[1], o[1]]), np.array([zv[0], zv[1], zv[0], zv[1]])

def coherence_at(t):               
    return np.exp(-DECOH * t)

# ================================================================
# Adent's Problem
# ================================================================

def density_operator(coherence):
    v_up = np.array([1, 0, 1, 0], dtype=float) / np.sqrt(2)
    v_dn = np.array([0, 1, 0, 1], dtype=float) / np.sqrt(2)
    rho_pure  = 0.5 * np.outer(v_up, v_up) + 0.5 * np.outer(v_dn, v_dn)
    rho_mixed = np.diag(np.diag(rho_pure))
    return rho_mixed + coherence * (rho_pure - rho_mixed)

def utility_matrix(C):
    C_flat = C.flatten()
    C_safe = np.maximum(C_flat, 1e-4)          # ← prevent log(0) or negative
    diag_vals = C_safe ** (1.0 - GAMMA) / (1.0 - GAMMA)
    diff = C_flat[:, None] - C_flat[None, :]
    U = KAPPA * np.exp(-(diff**2)) + np.diag(diag_vals)
    return U

def utility(controls, t):
    C, *_ = unpack_controls(controls)
    return np.trace(utility_matrix(C) @ density_operator(coherence_at(t)))

def price_approx(A, omegas, zs):
    omegas = np.atleast_1d(omegas)
    zs     = np.atleast_1d(zs)
    n = len(omegas)
    if len(zs) == 1 and n > 1: zs = np.full(n, zs[0])
    log_states = np.column_stack([np.ones(n), np.log(omegas), np.log(zs)])
    return np.exp(log_states @ A)

def unpack_price_params(theta):
    return [theta[3*k:3*k+3] for k in range(6)]

def budget_constraint(controls, state, theta):
    C, AD_1, AD_2, ALPHA_1, ALPHA_2 = unpack_controls(controls)
    omega, z, ad_1, ad_2, alpha_1, alpha_2 = unpack_states(state)
    omegas, zs = next_period_nodes(omega, z)
    coeffs = unpack_price_params(theta)
    P_1  = price_approx(coeffs[0], omegas, zs)[:, None]
    P_2  = price_approx(coeffs[1], omegas, zs)[:, None]
    Q_1H = price_approx(coeffs[2], omegas, zs)[:, None]
    Q_1L = price_approx(coeffs[3], omegas, zs)[:, None]
    Q_2H = price_approx(coeffs[4], omegas, zs)[:, None]
    Q_2L = price_approx(coeffs[5], omegas, zs)[:, None]
    ad_1_payoff = np.array([ad_1[0], ad_1[0], ad_1[1], ad_1[1]]).reshape(4, 1)
    ad_2_payoff = np.array([ad_2[0], ad_2[1], ad_2[0], ad_2[1]]).reshape(4, 1)
    wealth_in = ((P_1 + omega * DT) * alpha_1 
                 + (P_2 + z * DT) * alpha_2) + ad_1_payoff + ad_2_payoff
    new_cost  = (C * DT + P_1 * ALPHA_1 + P_2 * ALPHA_2 +
                 Q_1H * AD_1[:, 0:1] + Q_1L * AD_1[:, 1:2] +
                 Q_2H * AD_2[:, 0:1] + Q_2L * AD_2[:, 1:2])
    return (wealth_in - new_cost).flatten()

def laws_of_motion(controls, state, t):
    _, AD_1, AD_2, ALPHA_1, ALPHA_2 = unpack_controls(controls)
    omega, z, _, _, _, _ = unpack_states(state)
    omegas, zs = next_period_nodes(omega, z)
    next_coh = coherence_at(t + DT)
    return np.column_stack([omegas, zs, AD_1, AD_2,
                            ALPHA_1.flatten(), ALPHA_2.flatten(),
                            np.full(4, next_coh)])

def fit_value_function(states, V):
    X = np.column_stack([np.ones(len(states)), states])
    return np.linalg.lstsq(X, V, rcond=None)[0]

def expected_V(V_interp, next_states):
    return 0.25 * V_interp(next_states).sum()

def objective(controls, state, t, V_interp):
    next_states = laws_of_motion(controls, state, t)
    val = utility(controls, t) * DT + np.exp(-DELTA * DT) * expected_V(V_interp, next_states)
    return -val

def agent_optimization(theta, V_interp, state, t):
    cons = [{'type': 'ineq', 'fun': budget_constraint, 'args': (state, theta)}]
    x0 = np.ones(28) * 0.5
    bounds = [(1e-1, 4.0)]*4 + [(-1.0, 1.0)]*16 + [(0.0, 2.0)]*8
    res = minimize(objective, x0, args=(state, t, V_interp),
                   method='trust-constr', bounds=bounds, constraints=cons,
                   options={'maxiter': 150})
    return res

def _solve_one(args):
    idx, state, theta, V_next_interp, t = args
    res = agent_optimization(theta, V_next_interp, state, t)
    return idx, -res.fun, res.x

# ================================================================
# Equilibrium
# ================================================================

def sample_states(N, t):
    omega = np.random.uniform(0.5, 2.0, N)
    z     = np.random.uniform(0.5, 2.0, N)
    ad1   = np.random.uniform(-1.0, 1.0, (N, 2))
    ad2   = np.random.uniform(-1.0, 1.0, (N, 2))
    a1    = np.random.uniform(0.0, 2.0, N)
    a2    = np.random.uniform(0.0, 2.0, N)
    coh   = np.full(N, coherence_at(t))
    return np.column_stack([omega, z, ad1[:,0], ad1[:,1],
                            ad2[:,0], ad2[:,1], a1, a2, coh])

def compute_state_excess(policies):
    ed = np.zeros(6)
    ed[0] = policies[:, 4:8].mean(axis=1).mean() - 0.0
    ed[1] = policies[:, 8:12].mean(axis=1).mean() - 0.0
    ed[2] = policies[:,12:16].mean(axis=1).mean() - 0.0
    ed[3] = policies[:,16:20].mean(axis=1).mean() - 0.0
    ed[4] = policies[:,20:24].mean(axis=1).mean() - 1.0
    ed[5] = policies[:,24:28].mean(axis=1).mean() - 1.0
    return ed

def backward_step(theta, V_next_interp, N_samples, t):
    states = sample_states(N_samples, t)
    args_list = [(i, states[i], theta, V_next_interp, t) for i in range(N_samples)]
#    with Pool(processes=cpu_count()) as pool:
#        results = pool.map(_solve_one, args_list)
    results = []
    for arg in args_list:
        results.append(_solve_one(arg))
    V = np.array([r[1] for r in results])
    policy = np.array([r[2] for r in results])
    beta = fit_value_function(states, V)
    return V, ValueInterpolator(beta), policy   # ← now picklable

def equilibrium_prices_at_t(theta_init, V_next_interp, t):
    def excess_demand_fn(theta):
        total = N_MC_STATES * N_AGENTS_PER_STATE
        all_states = []
        state_ids = np.empty(total, dtype=int)
        idx = 0
        for k in range(N_MC_STATES):
            omega_k = np.random.uniform(0.5, 2.0)
            z_k     = np.random.uniform(0.5, 2.0)
            coh_k   = coherence_at(t)
            for _ in range(N_AGENTS_PER_STATE):
                ad1 = np.random.uniform(-1.0, 1.0, 2)
                ad2 = np.random.uniform(-1.0, 1.0, 2)
                a1  = np.random.uniform(0.0, 2.0)
                a2  = np.random.uniform(0.0, 2.0)
                s = np.array([omega_k, z_k, ad1[0], ad1[1], ad2[0], ad2[1], a1, a2, coh_k])
                all_states.append(s)
                state_ids[idx] = k
                idx += 1
        states_arr = np.array(all_states)

        args_list = [(i, states_arr[i], theta, V_next_interp, t) for i in range(total)]
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(_solve_one, args_list)
#        results = []
#        for arg in args_list:
#            results.append(_solve_one(arg))
        policies = np.array([r[2] for r in results])

        avg_alpha1 = policies[:,20:24].mean()
        avg_alpha2 = policies[:,24:28].mean()
        print(f"  Avg chosen α1 = {avg_alpha1:.4f}   (should be close to 1.0)")
        print(f"  Avg chosen α2 = {avg_alpha2:.4f}")

        big_ed = []
        for k in range(N_MC_STATES):
            mask = (state_ids == k)
            ed_k = compute_state_excess(policies[mask])
            big_ed.append(ed_k)
        return np.concatenate(big_ed)

    bounds = (-5 * np.ones(N_PRICE_PARAMS), 5 * np.ones(N_PRICE_PARAMS))
    result = least_squares(excess_demand_fn, theta_init,
                           bounds=bounds, method='trf',
                           ftol=1e-8, xtol=1e-8, gtol=1e-6,
                           max_nfev=200)
    theta_eq = result.x
    ed_final = excess_demand_fn(theta_eq)          # evaluate at solution
    ed_initial = excess_demand_fn(theta_init)      # at starting point

    norm_initial = np.linalg.norm(ed_initial)
    norm_final = np.linalg.norm(ed_final)
    improvement = np.linalg.norm(ed_initial) - np.linalg.norm(ed_final)

    print(f"                  ED initial norm = {norm_initial:.6f}")
    print(f"                  ED final norm   = {norm_final:.6f}")
    print(f"                  Improvement     = {improvement:.6f}")
    print(f"                  Success: {result.success} | message: {result.message}")
    print(f"                  nfev: {result.nfev}")

    _, V_interp, _ = backward_step(theta_eq, V_next_interp, N_SAMPLES, t)
    return theta_eq, None, V_interp, None

def run_backward_induction(resume_from=None):
    if resume_from:
        theta_path, V_next_interp, start_step, start_t = load_latest_checkpoint(resume_from)
        if start_step >= N_STEPS - 1:
            print("Already completed.")
            return theta_path

        if theta_path is not None:
            theta = theta_path[-1]  # last computed theta
        else:
            theta = INITIAL_GUESS
            # V_next_interp = ZeroValue()
            V_next_interp = ValueInterpolator(INITIAL_V)
            theta_path = np.zeros((N_STEPS, N_PRICE_PARAMS))

        step_start = start_step + 1
        
    else:
        theta = INITIAL_GUESS
        # V_next_interp = ZeroValue()
        V_next_interp = ValueInterpolator(INITIAL_V)
        theta_path = np.zeros((N_STEPS, N_PRICE_PARAMS))
        step_start = 0

    for step in range(step_start, N_STEPS):
        t = T_MAX - step * DT
        print(f"  Step {step+1}/{N_STEPS} | t={t:.3f}")
        theta, _, V_interp, _ = equilibrium_prices_at_t(theta, V_next_interp, t)
        theta_path[step] = theta
        V_next_interp = V_interp

        # save_dir = r'C:\Users\alegu\Python\Quantum_finance\simulation_checkpoints'
        save_dir = os.path.join(os.getcwd(), 'simulation_checkpoints')
        os.makedirs(save_dir, exist_ok=True)  # create folder if it doesn't exist

        checkpoint_data = {
            'theta_path': theta_path[:step+1],
            'V_interp': V_next_interp,   # the ValueInterpolator object
            'step': step,
            't': t
        }

        checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{step+1:03d}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"Full checkpoint saved: {checkpoint_path}")
    
    return theta_path

# ================================================================
# Simulation
# ================================================================

def forward_simulate(theta_path, n_paths=50):
    steps = theta_path.shape[0]
    omega = np.ones(n_paths)
    z     = np.ones(n_paths)
    rf = np.zeros((steps, n_paths))
    r1 = np.zeros((steps, n_paths))
    r2 = np.zeros((steps, n_paths))
    P1nn = np.zeros((steps, n_paths))
    P2nn = np.zeros((steps, n_paths))

    for step in range(steps):
        theta = theta_path[step]
        coeffs = unpack_price_params(theta)
        shocks = np.random.choice([1, -1], size=(n_paths, 2))
        omega_n = omega * np.exp((MU - 0.5*SIGMA**2)*DT + shocks[:,0]*SIGMA*np.sqrt(DT))
        z_n     = z     * np.exp((MU - 0.5*SIGMA**2)*DT + shocks[:,1]*SIGMA*np.sqrt(DT))

#        P1n  = price_approx(coeffs[0], omega, z)
#        P1nn[step] = price_approx(coeffs[0], omega_n, z_n)
#        P2n  = price_approx(coeffs[1], omega, z)
#        P2nn[step] = price_approx(coeffs[1], omega_n, z_n)
#        Q1H  = price_approx(coeffs[2], omega, z)
#        Q1L  = price_approx(coeffs[3], omega, z)

        P1n  = np.asarray(price_approx(coeffs[0], omega, z)).reshape(-1)
        P2n  = np.asarray(price_approx(coeffs[1], omega, z)).reshape(-1)
        P1nn_step = np.asarray(price_approx(coeffs[0], omega_n, z_n)).reshape(-1)
        P2nn_step = np.asarray(price_approx(coeffs[1], omega_n, z_n)).reshape(-1)
        P1nn[step] = P1nn_step
        P2nn[step] = P2nn_step
        Q1H = np.asarray(price_approx(coeffs[2], omega, z)).reshape(-1)
        Q1L = np.asarray(price_approx(coeffs[3], omega, z)).reshape(-1)

        rf[step] = (1 / (Q1H + Q1L) - 1 ) / DT
        r1[step] = (P1nn_step - P1n) / (P1n * DT) + omega / P1n
        r2[step] = (P2nn_step - P2n) / (P2n * DT) + z / P2n

        omega, z = omega_n, z_n
    return rf, r1, r2, P1nn, P2nn

#def plot_returns(rf, r1, r2):
#    t_grid = np.linspace(0, T_MAX, rf.shape[0])
#    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
#    for a, arr, lab in zip(ax, [rf, r1, r2],
#                           ['Risk-free $r^f$', 'Asset 1 $r^1$', 'Asset 2 $r^2$']):
#        a.plot(t_grid, arr.mean(1), lw=2)
#        a.fill_between(t_grid, np.percentile(arr, 10, 1), np.percentile(arr, 90, 1), alpha=0.3)
#        a.set_ylabel(lab)
#        a.legend(['mean', '10–90%'])
#    ax[2].set_xlabel('Time')
#    fig.suptitle('Equilibrium returns')
#    plt.tight_layout()
##    plt.savefig(r'C:\Users\alegu\Python\Quantum_finance\returns.png', dpi=150)
#    plt.savefig(os.path.join(os.getcwd(), 'returns.png'), dpi=150)
#    plt.close()
#    print("Saved returns.png")

def plot_returns(rf, r1, r2):
    t_grid = np.linspace(0, T_MAX, rf.shape[0])
    dt = T_MAX / rf.shape[0]
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for a, arr, lab in zip(
        ax,
        [rf, r1, r2],
        ['Risk-free $r^f$', 'Asset 1 $r^1$', 'Asset 2 $r^2$']
    ):
        mean_inst = arr.mean(axis=1)

        log_returns = arr * dt
        cum_log_returns = np.cumsum(log_returns, axis=0)
        time_grid = t_grid.copy()
        time_grid[0] = dt  # avoid division by zero
        annualized = cum_log_returns / time_grid[:, None]

        p10 = np.percentile(annualized, 10, axis=1)
        p90 = np.percentile(annualized, 90, axis=1)

        # --- Plot ---
        a.plot(t_grid, mean_inst, lw=2, label='mean (instantaneous)')
        a.fill_between(t_grid, p10, p90, alpha=0.3, label='10–90% annualized')
        a.set_ylabel(lab)
        a.legend()

    ax[2].set_xlabel('Time')
    fig.suptitle('Instantaneous mean with annualized return bands')

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'returns.png'), dpi=150)
    plt.close()

    print("Saved returns.png")

# ================================================================
# Run
# ================================================================

if __name__ == '__main__':
    checkpoint_dir = os.path.join(os.getcwd(), 'simulation_checkpoints')
    theta_path = run_backward_induction(checkpoint_dir)
    # theta_path = run_backward_induction(r'C:\Users\alegu\Python\Quantum_finance\simulation_checkpoints')
    rf, r1, r2, P1nn, P2nn = forward_simulate(theta_path)
    plot_returns(rf, r1, r2)
    # Create time grid
    t_grid = np.linspace(0, T_MAX, rf.shape[0])

    # Average and percentiles
    data = {
        'Time': t_grid,
        'Risk-free rate (mean)': rf.mean(axis=1),
        'Risk-free rate (10th pct)': np.percentile(rf, 10, axis=1),
        'Risk-free rate (90th pct)': np.percentile(rf, 90, axis=1),
        'Asset 1 return (mean)': r1.mean(axis=1),
        'Asset 1 return (10th pct)': np.percentile(r1, 10, axis=1),
        'Asset 1 return (90th pct)': np.percentile(r1, 90, axis=1),
        'Asset 2 return (mean)': r2.mean(axis=1),
        'Asset 2 return (10th pct)': np.percentile(r2, 10, axis=1),
        'Asset 2 return (90th pct)': np.percentile(r2, 90, axis=1),
        'Asset 1 price (mean)': P1nn.mean(axis=1),
        'Asset 1 price (10th pct)': np.percentile(P1nn, 10, axis=1),
        'Asset 1 price (90th pct)': np.percentile(P1nn, 90, axis=1),
        'Asset 2 price (mean)': P2nn.mean(axis=1),
        'Asset 2 price (10th pct)': np.percentile(P2nn, 10, axis=1),
        'Asset 2 price (90th pct)': np.percentile(P2nn, 90, axis=1),
    }

    df = pd.DataFrame(data)
    excel_path = os.path.join(os.getcwd(), 'equilibrium_returns_summary.xlsx')
#    excel_path = r'C:\Users\alegu\Python\Quantum_finance\equilibrium_returns_summary.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"Saved simulation summary to {excel_path}")
    print("✅ Done!")