"""
Quantum Finance Equilibrium Solver — v4 (corrected)
=====================================================
"The Irrelevance of Ambiguity Aversion" — Gutierrez Valencia

CORRECTIONS FROM v3:
  1. AD short positions: a_payoff and AD holdings grids are symmetric around
     zero. LP allows negative AD holdings (short positions). Zero net supply
     requires longs and shorts to cancel.
  2. Budget constraint: income term is E[Y] = Tr[rho^TOT * Y], an expectation
     over the CHOSEN distribution of next-period dividends, not the realized
     scalar y_t. Specifically E[Y] = sum_k w_k*(E[alpha1']*omega_k + E[alpha2']*z_k).
  3. Continuation value: V evaluated at next-period a_payoff = E[a'_k] for
     each shock k, not at zero.
  4. AD market clearing in VFI: path-by-path per (omega,z) node. Prices depend
     on (omega,z,mu1,mu2,...) so clearing is enforced at each (omega,z) cell
     separately, not averaged over the inner (alpha,a_pay) dimensions.
  5. rho^u decay: controlled by kappa in [0,1]. kappa=0 => no imposed decay
     (decoherence emerges endogenously from equilibrium). kappa=1 => full decay
     at rate sigma2^2/2 per period. Default: kappa=0.
  6. Decoherence diagnostic: track off-diagonal norm of rho_C (the consumption
     marginal of the agent's optimal density operator), NOT rho^u. This is the
     correct object per Proposition 10 and Proposition 12.
  7. J=3 agents to start (fast, < 1 hour). Easily extended by editing ENDOWMENTS.

STATE:   (omega_i, z_j, alpha1_k, alpha2_l, a_payoff_m)
          omega, z   : recombining binomial (independent GBMs)
          alpha1/2   : equity holdings (slow state, carried between periods)
          a_payoff   : scalar AD payoff received THIS period (from last period's
                       portfolio; can be negative if agent was short)

CONTROLS (encoded in rho^TOT marginals):
  p_c   : distribution over consumption levels
  p_a1  : distribution over new alpha1
  p_a2  : distribution over new alpha2
  p_ad  : distribution over new AD holdings per shock (can be negative)

ALGORITHM: Krusell-Smith
  Outer : update forecasting rule coefficients via ridge regression
  Middle: VFI to convergence (warm-start), aggregate moments fixed at ergodic
  Inner : joint LP per agent per node

PARALLELISM: joblib loky, chunked. Set N_JOBS = physical core count.
             Windows requires if __name__=='__main__' guard.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
import warnings, time
warnings.filterwarnings('ignore')

N_JOBS     = 10
CHUNK_SIZE = 100
V_MAX      = 500.0

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PARAMETERS  — edit here to change the model
# ═══════════════════════════════════════════════════════════════════════════════

class Params:
    # ── GBM processes ──
    mu1, sigma1 = 0.05, 0.20      # asset 1: risky, classical
    mu2, sigma2 = 0.04, 0.25      # asset 2: ambiguous, quantum
    dt          = 1.0
    delta       = 0.05            # discount rate

    # ── Utility (identical for all agents) ──
    gamma_crra  = 2.0             # CRRA curvature
    gamma_amb   = 0.10            # off-diagonal ambiguity premium in U

    # ── rho^u decay parameter ──
    # kappa=0: no imposed decay, decoherence emerges endogenously
    # kappa=1: full decay at rate sigma2^2/2 per period (Axiom 4 imposed)
    kappa       = 0.0

    # ── Aggregate equity supply (total economy, normalised) ──
    ALPHA1_SUPPLY = 1.0           # sum_j alpha1^j / J = ALPHA1_SUPPLY
    ALPHA2_SUPPLY = 1.0

    # ── J=3 agents: different wealth AND composition ──
    # Mean of each column must equal supply. Extend to J=5 by adding rows.
    # Column sums / J must equal supply.
    ENDOWMENTS = np.array([
        [1.6, 0.4],   # j=0: asset-1 heavy
        [1.0, 1.0],   # j=1: equal weight
        [0.4, 1.6],   # j=2: asset-2 heavy
    ])
    # Check: mean alpha1 = (1.6+1.0+0.4)/3 = 1.0 ✓
    #        mean alpha2 = (0.4+1.0+1.6)/3 = 1.0 ✓

    # ── Grid sizes ──
    N_omega = 3    # recombining binomial for omega (3 nodes = fast)
    N_z     = 3    # recombining binomial for z
    N_alpha = 4    # equity holdings grid
    N_a     = 4    # AD payoff grid (SYMMETRIC around 0)
    N_c     = 5    # consumption grid

    # ── VFI ──
    eta         = 0.05    # damping (must be small for CRRA stability)
    tol_vfi     = 1e-3
    max_vfi     = 400

    # ── KS simulation ──
    J           = 3
    T_sim       = 300
    tol_ks      = 5e-4
    max_ks      = 30
    ridge_lam   = 1e-4

    # ── Newton market clearing ──
    newton_tol  = 1e-5
    newton_max  = 30

P = Params()
J = P.J

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  GRIDS
# ═══════════════════════════════════════════════════════════════════════════════

def recombining_grid(mu, sigma, dt, N):
    ks = np.arange(N) - (N - 1) / 2.0
    return np.exp(mu * dt) * np.exp(ks * sigma * np.sqrt(dt))

omega_grid = recombining_grid(P.mu1, P.sigma1, P.dt, P.N_omega)
z_grid     = recombining_grid(P.mu2, P.sigma2, P.dt, P.N_z)

def trans_idx(N):
    return np.minimum(np.arange(N) + 1, N - 1), np.maximum(np.arange(N) - 1, 0)

omega_up, omega_dn = trans_idx(P.N_omega)
z_up,     z_dn     = trans_idx(P.N_z)

# Equity grid: non-negative (can't hold negative equity)
alpha_grid = np.linspace(0.0, 2.5, P.N_alpha)

# AD payoff grid: SYMMETRIC around zero (short positions give negative payoff)
a_max          = 3.0
a_payoff_grid  = np.linspace(-a_max, a_max, P.N_a)

# AD holdings grid for LP: same symmetric range
ad_grid        = np.linspace(-a_max, a_max, P.N_alpha)  # reuse N_alpha points

# Consumption grid: strictly positive
c_grid = np.linspace(0.05, 4.0, P.N_c)

# State grid shape
ind_shape = (P.N_omega, P.N_z, P.N_alpha, P.N_alpha, P.N_a)
N_ind     = int(np.prod(ind_shape))
_all_idx  = np.array(np.unravel_index(np.arange(N_ind), ind_shape)).T

N_SHOCKS   = 4    # (omega_dir, z_dir) in {up,dn}^2
SHOCK_PROB = 0.25

# Precompute next-period (omega, z) values for each shock at each grid node
# shock_om_next[i0, k] = omega value for shock k from omega node i0
# k=0:(up,up) k=1:(up,dn) k=2:(dn,up) k=3:(dn,dn)
_om_dirs = [omega_up, omega_up, omega_dn, omega_dn]  # omega direction per shock
_z_dirs  = [z_up,    z_dn,    z_up,    z_dn]        # z direction per shock

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  RHO^u  (eq. 58, with controllable decay)
# ═══════════════════════════════════════════════════════════════════════════════

def evolve_rho_u(rho, mu, sigma, dt, kappa):
    """
    One step of eq. (58).
    kappa=0: no off-diagonal decay (pure phase rotation)
    kappa=1: full decay at rate sigma^2/2 per step
    """
    rho_new      = rho.copy().astype(complex)
    decay        = np.exp(-kappa * (sigma**2 / 2.0) * dt)
    phase        = np.exp(1j * (mu - sigma**2 / 2.0) * dt)
    rho_new[0,1] = rho[0,1] * decay * phase
    rho_new[1,0] = np.conj(rho_new[0,1])
    rho_new     /= np.trace(rho_new).real
    return rho_new

# Initial rho^u: maximum coherence (equal superposition)
rho_u_init = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
rho_u_path = [rho_u_init]
for _ in range(50):
    rho_u_path.append(
        evolve_rho_u(rho_u_path[-1], P.mu2, P.sigma2, P.dt, P.kappa))
rho_u_ss = rho_u_path[-1]

# Classical z-probability weights per shock from rho^u diagonal
# shock k: z=up for k in {0,2}, z=dn for k in {1,3}
pz_up = float(rho_u_ss[0, 0].real)
pz_dn = float(rho_u_ss[1, 1].real)
Z_W   = np.array([pz_up, pz_dn, pz_up, pz_dn])  # weight per shock

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  UTILITY OPERATOR  (identical for all agents)
# ═══════════════════════════════════════════════════════════════════════════════

def crra(c):
    return c**(1.0 - P.gamma_crra) / (1.0 - P.gamma_crra)

u_diag = crra(c_grid)   # (N_c,) diagonal of U

# Off-diagonal bonus for LP (first-order approximation of ambiguity premium)
# Tr[rho_C U] = sum_i p_c[i]*u_ii + 2*gamma_amb*sum_i sqrt(p_c[i]*p_c[i+1])
# The bonus adds gamma_amb to adjacent consumption levels to incentivise spread
_bonus = np.zeros(P.N_c)
_bonus[:-1] += P.gamma_amb
_bonus[1:]  += P.gamma_amb

# Full U matrix for rho_C evaluation
_Ufull = np.diag(u_diag)
for i in range(P.N_c - 1):
    _Ufull[i, i+1] = _Ufull[i+1, i] = P.gamma_amb

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  KRUSELL-SMITH FORECASTING RULES
# ═══════════════════════════════════════════════════════════════════════════════

# Feature vector: [1, ln(om), ln(z), ln(mu1), ln(mu2), ln(s1), ln(s2), rho12]
N_KS = 8

def ks_feat(omega, z, mu1, mu2, sig1, sig2, sig12):
    eps   = 1e-8
    rho12 = np.clip(sig12 / (max(sig1, eps) * max(sig2, eps)), -1.0, 1.0)
    return np.array([1.0,
                     np.log(max(omega, eps)),
                     np.log(max(z,     eps)),
                     np.log(max(mu1,   eps)),
                     np.log(max(mu2,   eps)),
                     np.log(max(sig1,  eps)),
                     np.log(max(sig2,  eps)),
                     rho12])

def ks_price(coef, feat):
    return float(np.exp(feat @ coef))

def ridge(X, y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)

def init_coefs():
    c = {}
    # P1 ~ omega / (delta + 0.5*sigma1^2)
    b = np.zeros(N_KS); b[0] = -np.log(P.delta + 0.5*P.sigma1**2); b[1] = 1.0
    c['P1'] = b.copy()
    # P2 ~ z / (delta + 0.5*sigma2^2)
    b = np.zeros(N_KS); b[0] = -np.log(P.delta + 0.5*P.sigma2**2); b[2] = 1.0
    c['P2'] = b.copy()
    # Q_k: equal AD prices summing to exp(-delta*dt)
    q0 = np.log(np.exp(-P.delta * P.dt) / N_SHOCKS)
    for k in range(N_SHOCKS):
        b = np.zeros(N_KS); b[0] = q0
        c[f'Q{k}'] = b.copy()
    # Law of motion for mu1, mu2: near unit root
    b = np.zeros(N_KS); b[3] = 1.0; c['mu1'] = b.copy()
    b = np.zeros(N_KS); b[4] = 1.0; c['mu2'] = b.copy()
    return c

def update_coefs(records, lam=P.ridge_lam):
    X     = np.array([r['feat']    for r in records])
    lnP1  = np.log(np.maximum([r['P1']  for r in records], 1e-8))
    lnP2  = np.log(np.maximum([r['P2']  for r in records], 1e-8))
    lnmu1 = np.log(np.maximum([r['mu1n'] for r in records], 1e-8))
    lnmu2 = np.log(np.maximum([r['mu2n'] for r in records], 1e-8))

    c = {}
    c['P1']  = ridge(X, lnP1,  lam)
    c['P2']  = ridge(X, lnP2,  lam)
    c['mu1'] = ridge(X, lnmu1, lam)
    c['mu2'] = ridge(X, lnmu2, lam)
    for k in range(N_SHOCKS):
        lnQk = np.log(np.maximum([r['Q'][k] for r in records], 1e-8))
        c[f'Q{k}'] = ridge(X, lnQk, lam)

    r2 = {}
    for key, y in [('P1',lnP1),('P2',lnP2),('mu1',lnmu1),('mu2',lnmu2)]:
        yh = X @ c[key]
        ss = np.sum((y - y.mean())**2) + 1e-12
        r2[key] = float(1.0 - np.sum((y - yh)**2) / ss)
    return c, r2

def coef_dist(c_new, c_old):
    return max(np.max(np.abs(c_new[k] - c_old[k])) for k in c_new)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  INTERPOLATOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_interp(Vg):
    return RegularGridInterpolator(
        (omega_grid, z_grid, alpha_grid, alpha_grid, a_payoff_grid),
        Vg, method='linear', bounds_error=False, fill_value=None)

def eval_V(interp, om, z, a1, a2, ap):
    return float(interp((
        np.clip(om, omega_grid[0],    omega_grid[-1]),
        np.clip(z,  z_grid[0],        z_grid[-1]),
        np.clip(a1, alpha_grid[0],    alpha_grid[-1]),
        np.clip(a2, alpha_grid[0],    alpha_grid[-1]),
        np.clip(ap, a_payoff_grid[0], a_payoff_grid[-1])
    )))

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SINGLE-NODE SOLVER  — the core LP
# ═══════════════════════════════════════════════════════════════════════════════

def solve_node(i0, i1, i2, i3, i4,
               V_interp, P1_s, P2_s, Q_vec, coefs, agg_state):
    """
    Joint LP for one agent at individual state node (i0..i4).

    BUDGET CONSTRAINT (corrected):
      E[c]
      + E[alpha1'] * P1  - alpha1_cur * P1      (net equity purchase, asset 1)
      + E[alpha2'] * P2  - alpha2_cur * P2      (net equity purchase, asset 2)
      + sum_k E[a'_k] * Q_k                     (AD portfolio cost, can be neg)
      <= a_payoff_cur                            (AD payoff from last period)
       + E[Y]                                   (expected dividend income)

    E[Y] = sum_k SHOCK_PROB * Z_W[k] *
               (E[alpha1'] * omega_k' + E[alpha2'] * z_k')

    This is CORRECT because at time t we choose distributions over t+dt outcomes.
    E[Y] is linear in (p_a1, p_a2) so it enters the LP naturally.

    CONTINUATION VALUE (corrected):
      C_jk = exp(-delta*dt) * sum_k w_k * V(omega_k', z_k',
                                             E[alpha1'], E[alpha2'],
                                             E[a'_k])
    where E[alpha1'], E[alpha2'] and E[a'_k] come from the LP solution.
    Since LP hasn't been solved yet, we use a two-pass approach:
      Pass 1: solve LP with C_jk=0 to get E[alpha'], E[a']
      Pass 2: recompute C_jk with correct next-period states, re-solve LP

    Returns (V_new, E_a1, E_a2, E_ad[N_SHOCKS], rho_C, lp_success)
    """
    omega  = omega_grid[i0];   z_val  = z_grid[i1]
    alpha1 = alpha_grid[i2];   alpha2 = alpha_grid[i3]
    a_pay  = a_payoff_grid[i4]
    Nc = len(c_grid);  Na = len(alpha_grid);  Nad = len(ad_grid)

    # Next-period (omega, z) values per shock
    om_next = [omega_grid[_om_dirs[k][i0]] for k in range(N_SHOCKS)]
    z_next  = [z_grid[_z_dirs[k][i1]]     for k in range(N_SHOCKS)]

    # Forecast next-period aggregate moments (for continuation value)
    feat_cur = ks_feat(*agg_state)
    mu1_next = float(np.exp(feat_cur @ coefs['mu1']))
    mu2_next = float(np.exp(feat_cur @ coefs['mu2']))
    sig1_next = agg_state[4];  sig2_next = agg_state[5]
    sig12_next = agg_state[6]

    def build_lp(C_jk, E_a1_guess, E_a2_guess, E_ad_guess):
        """
        Build and solve the LP.

        Variables (all entries non-negative, representing prob masses):
          p_c    [Nc]        : prob over consumption c_grid
          p_a1   [Na]        : prob over new alpha1  alpha_grid
          p_a2   [Na]        : prob over new alpha2  alpha_grid
          p_ad   [Nad x Nsh] : prob over (AD holding ad_grid, shock) pairs
                               FLAT: [ad0_sh0, ad0_sh1, ..., ad_{Nad-1}_sh3]

        Note on AD short positions:
          ad_grid includes negative values. The LP variable p_ad[i,k] is a
          probability (>=0), but the AD holding value ad_grid[i] can be negative.
          So E[a'_k] = sum_i p_ad[i,k] * ad_grid[i] can be negative (short).
          This correctly implements zero-net-supply AD market.
        """
        n = Nc + Na + Na + Nad * N_SHOCKS
        obj = np.concatenate([
            -(u_diag + _bonus + C_jk),           # maximize utility + cont.
            (alpha_grid - alpha1)**2,             # minimize alpha1 deviation
            (alpha_grid - alpha2)**2,             # minimize alpha2 deviation
            np.zeros(Nad * N_SHOCKS)              # AD: no direct objective
        ])

        # Simplex constraints (each sub-vector sums to 1)
        Aeq = np.zeros((4, n))
        Aeq[0, :Nc]                     = 1.0
        Aeq[1, Nc:Nc+Na]                = 1.0
        Aeq[2, Nc+Na:Nc+2*Na]           = 1.0
        Aeq[3, Nc+2*Na:]                = 1.0
        beq = np.ones(4)

        # Budget constraint (single inequality):
        # E[c] + E[alpha1']*P1 + E[alpha2']*P2 + sum_k E[a'_k]*Q_k
        #   - E[Y]
        # <= a_pay + alpha1*P1 + alpha2*P2
        #
        # E[Y] = sum_k SHOCK_PROB * Z_W[k] * (E[alpha1']*om_k + E[alpha2']*z_k)
        #      = E[alpha1'] * sum_k SHOCK_PROB*Z_W[k]*om_k
        #        + E[alpha2'] * sum_k SHOCK_PROB*Z_W[k]*z_k
        #
        # Let om_mean_w = sum_k SHOCK_PROB*Z_W[k]*om_k  (weighted mean next-omega)
        #     z_mean_w  = sum_k SHOCK_PROB*Z_W[k]*z_k
        # Normalise Z_W sum
        zw_sum   = float(Z_W.sum()) * SHOCK_PROB
        om_mean_w = sum(SHOCK_PROB * Z_W[k] * om_next[k] for k in range(N_SHOCKS)) / max(zw_sum, 1e-12)
        z_mean_w  = sum(SHOCK_PROB * Z_W[k] * z_next[k]  for k in range(N_SHOCKS)) / max(zw_sum, 1e-12)

        Aub = np.zeros((1, n))
        Aub[0, :Nc]           = c_grid                              # E[c]
        Aub[0, Nc:Nc+Na]      = alpha_grid * (P1_s - om_mean_w)   # E[a1']*(P1-E[omega'])
        Aub[0, Nc+Na:Nc+2*Na] = alpha_grid * (P2_s - z_mean_w)    # E[a2']*(P2-E[z'])
        # AD cost: for each shock k and AD level i: p_ad[i,k]*ad_grid[i]*Q_k
        for k in range(N_SHOCKS):
            idx = Nc + 2*Na + np.arange(Nad) * N_SHOCKS + k
            Aub[0, idx] = ad_grid * Q_vec[k]

        bub = np.array([a_pay + alpha1*P1_s + alpha2*P2_s])

        # Bounds: probabilities in [0,1] (ad_grid values can be negative
        # but probability weights are non-negative — this is correct)
        bounds = [(0.0, 1.0)] * n

        lp = linprog(obj, A_ub=Aub, b_ub=bub,
                     A_eq=Aeq, b_eq=beq,
                     bounds=bounds, method='highs')
        return lp

    # ── Pass 1: solve with C_jk=0 to get initial E[alpha'], E[a'] ──
    lp1 = build_lp(0.0, alpha1, alpha2, np.zeros(N_SHOCKS))

    if lp1.success and np.all(lp1.x >= -1e-9):
        x1    = np.maximum(lp1.x, 0.0)
        p_a1  = x1[P.N_c:P.N_c+P.N_alpha];      p_a1 /= max(p_a1.sum(), 1e-12)
        p_a2  = x1[P.N_c+P.N_alpha:P.N_c+2*P.N_alpha]; p_a2 /= max(p_a2.sum(), 1e-12)
        p_ad  = x1[P.N_c+2*P.N_alpha:].reshape(len(ad_grid), N_SHOCKS)
        p_ad /= max(p_ad.sum(), 1e-12)
        E_a1_1 = float(p_a1 @ alpha_grid)
        E_a2_1 = float(p_a2 @ alpha_grid)
        E_ad_1 = np.array([float(p_ad[:, k] @ ad_grid) for k in range(N_SHOCKS)])
    else:
        E_a1_1 = alpha1;  E_a2_1 = alpha2
        E_ad_1 = np.zeros(N_SHOCKS)

    # ── Compute C_jk with correct next-period a_payoff ──
    C_jk  = 0.0
    w_sum = 0.0
    for k in range(N_SHOCKS):
        w = SHOCK_PROB * Z_W[k]
        # Next-period a_payoff = E[a'_k] (the AD holding for realised shock k)
        ap_next = float(np.clip(E_ad_1[k], a_payoff_grid[0], a_payoff_grid[-1]))
        feat_nk = ks_feat(om_next[k], z_next[k],
                          mu1_next, mu2_next, sig1_next, sig2_next, sig12_next)
        # Prices at next-period state (for V evaluation we just need V values)
        V_k = eval_V(V_interp, om_next[k], z_next[k],
                     E_a1_1, E_a2_1, ap_next)
        C_jk  += w * V_k
        w_sum += w
    C_jk = float(np.exp(-P.delta * P.dt) * C_jk / max(w_sum, 1e-12))

    # ── Pass 2: solve with correct C_jk ──
    lp2 = build_lp(C_jk, E_a1_1, E_a2_1, E_ad_1)

    if lp2.success and np.all(lp2.x >= -1e-9):
        x2   = np.maximum(lp2.x, 0.0)
        Nc_  = P.N_c;  Na_ = P.N_alpha;  Nad_ = len(ad_grid)
        pc   = x2[:Nc_];               pc   /= max(pc.sum(),   1e-12)
        p_a1 = x2[Nc_:Nc_+Na_];       p_a1 /= max(p_a1.sum(),1e-12)
        p_a2 = x2[Nc_+Na_:Nc_+2*Na_]; p_a2 /= max(p_a2.sum(),1e-12)
        p_ad = x2[Nc_+2*Na_:].reshape(Nad_, N_SHOCKS)
        p_ad /= max(p_ad.sum(), 1e-12)
        ok = True
    else:
        # Fallback: consume a_pay + dividend (stay put in equity, zero AD)
        pc   = np.zeros(P.N_c)
        best = np.argmin(np.abs(c_grid - max(a_pay + alpha1*0.05, c_grid[0])))
        pc[best] = 1.0
        p_a1 = np.zeros(P.N_alpha); p_a1[np.argmin(np.abs(alpha_grid-alpha1))] = 1.0
        p_a2 = np.zeros(P.N_alpha); p_a2[np.argmin(np.abs(alpha_grid-alpha2))] = 1.0
        p_ad = np.ones((len(ad_grid), N_SHOCKS)) / (len(ad_grid) * N_SHOCKS)
        ok = False

    # ── Build rho_C and compute utility ──
    v     = np.sqrt(np.maximum(pc, 0.0))
    rho_C = np.outer(v, v)
    u_now = float(np.trace(rho_C @ _Ufull))

    E_a1 = float(p_a1 @ alpha_grid)
    E_a2 = float(p_a2 @ alpha_grid)
    E_ad = np.array([float(p_ad[:, k] @ ad_grid) for k in range(N_SHOCKS)])

    V_new = float(np.clip(u_now + C_jk, -V_MAX, 0.0))
    return V_new, E_a1, E_a2, E_ad, rho_C, ok

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CHUNK WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def process_chunk(idx_list, V_grid, P1g, P2g, Qg, coefs, agg_state):
    interp = make_interp(V_grid)
    out    = []
    for idx in idx_list:
        i0,i1,i2,i3,i4 = _all_idx[idx]
        out.append((idx, solve_node(
            i0, i1, i2, i3, i4,
            interp,
            float(P1g[i0, i1]),
            float(P2g[i0, i1]),
            Qg[i0, i1, :],
            coefs, agg_state)))
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  PRICE GRID FROM KS COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════════

def price_grids(coefs, agg_state):
    """
    Evaluate KS forecasting rule on the (omega,z) grid,
    using aggregate moments from agg_state.
    """
    _, _, mu1, mu2, sig1, sig2, sig12 = agg_state
    P1g = np.zeros((P.N_omega, P.N_z))
    P2g = np.zeros((P.N_omega, P.N_z))
    Qg  = np.zeros((P.N_omega, P.N_z, N_SHOCKS))
    for i0, om in enumerate(omega_grid):
        for i1, z in enumerate(z_grid):
            f = ks_feat(om, z, mu1, mu2, sig1, sig2, sig12)
            P1g[i0, i1] = ks_price(coefs['P1'], f)
            P2g[i0, i1] = ks_price(coefs['P2'], f)
            for k in range(N_SHOCKS):
                Qg[i0, i1, k] = ks_price(coefs[f'Q{k}'], f)
    return P1g, P2g, Qg

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  VFI FOR ONE AGENT  (to convergence, warm-start)
# ═══════════════════════════════════════════════════════════════════════════════

def vfi_agent(V_old, P1g, P2g, Qg, agg_state, coefs,
              n_jobs=N_JOBS, chunk_size=CHUNK_SIZE, verbose=False):
    all_idx = list(range(N_ind))
    chunks  = [all_idx[i:i+chunk_size] for i in range(0, N_ind, chunk_size)]
    V       = V_old.copy()

    for n in range(P.max_vfi):
        Vnew   = np.zeros(ind_shape)
        n_fail = 0

        raw = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_chunk)(ch, V, P1g, P2g, Qg, coefs, agg_state)
            for ch in chunks)

        for chunk_res in raw:
            for idx, (Vj, ea1, ea2, ead, _, ok) in chunk_res:
                i0,i1,i2,i3,i4 = _all_idx[idx]
                Vnew[i0,i1,i2,i3,i4] = (
                    (1 - P.eta) * V[i0,i1,i2,i3,i4] + P.eta * Vj)
                if not ok:
                    n_fail += 1

        dV = float(np.max(np.abs(Vnew - V)))
        V  = Vnew
        if verbose:
            print(f"    VFI {n:3d}  |dV|={dV:.6f}  fails={n_fail}")
        if dV < P.tol_vfi:
            break

    return V, n + 1, dV, n_fail

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  CROSS-SECTIONAL MOMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def xsec_moments(a1s, a2s):
    mu1   = float(np.mean(a1s));  mu2   = float(np.mean(a2s))
    sig1  = float(np.std(a1s))  + 1e-6
    sig2  = float(np.std(a2s))  + 1e-6
    sig12 = float(np.mean((a1s - mu1) * (a2s - mu2)))
    return mu1, mu2, sig1, sig2, sig12

# ═══════════════════════════════════════════════════════════════════════════════
# 12.  WITHIN-PERIOD MARKET CLEARING  (Newton, path-by-path)
# ═══════════════════════════════════════════════════════════════════════════════

def clear_markets(agent_states, V_interps, agg_state, coefs, P1_init, P2_init, Q_init):
    """
    Find P1*, P2*, Q* such that:
      (1/J) sum_j E^j[alpha1'] = ALPHA1_SUPPLY   (equity 1)
      (1/J) sum_j E^j[alpha2'] = ALPHA2_SUPPLY   (equity 2)
      (1/J) sum_j E^j[a'_k]   = 0  for each k   (AD, zero net supply)

    Uses Newton iteration with log-linear step.
    """
    P1 = P1_init;  P2 = P2_init;  Q = Q_init.copy()

    for it in range(P.newton_max):
        ea1s = np.zeros(J);  ea2s = np.zeros(J)
        eads = np.zeros((J, N_SHOCKS))
        rho_Cs = []

        for j, (om,z,a1,a2,ap) in enumerate(agent_states):
            i0 = np.argmin(np.abs(omega_grid - om))
            i1 = np.argmin(np.abs(z_grid     - z))
            i2 = np.argmin(np.abs(alpha_grid  - a1))
            i3 = np.argmin(np.abs(alpha_grid  - a2))
            i4 = np.argmin(np.abs(a_payoff_grid - ap))
            _, ea1, ea2, ead, rC, _ = solve_node(
                i0,i1,i2,i3,i4, V_interps[j], P1, P2, Q, coefs, agg_state)
            ea1s[j] = ea1;  ea2s[j] = ea2;  eads[j] = ead
            rho_Cs.append(rC)

        ED1  = float(np.mean(ea1s)) - P.ALPHA1_SUPPLY
        ED2  = float(np.mean(ea2s)) - P.ALPHA2_SUPPLY
        EDad = np.mean(eads, axis=0)   # should be ~0 (zero net supply)

        if (abs(ED1)  < P.newton_tol and
            abs(ED2)  < P.newton_tol and
            np.all(np.abs(EDad) < P.newton_tol)):
            break

        step = 0.10
        P1 = max(P1 * (1.0 + step * ED1),  0.01)
        P2 = max(P2 * (1.0 + step * ED2),  0.01)
        Q  = np.maximum(Q * (1.0 + step * EDad), 1e-5)

    return P1, P2, Q, ea1s, ea2s, eads, rho_Cs

# ═══════════════════════════════════════════════════════════════════════════════
# 13.  SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(V_agents, coefs, T=P.T_sim, verbose=False):
    V_interps = [make_interp(V_agents[j]) for j in range(J)]

    # Initialise at endowments, central grid nodes
    alphas1 = P.ENDOWMENTS[:, 0].copy()
    alphas2 = P.ENDOWMENTS[:, 1].copy()
    a_pays  = np.zeros(J)
    i_om = P.N_omega // 2;  i_z = P.N_z // 2
    om_cur = omega_grid[i_om];  z_cur = z_grid[i_z]

    mu1,mu2,sig1,sig2,sig12 = xsec_moments(alphas1, alphas2)
    agg = (om_cur, z_cur, mu1, mu2, sig1, sig2, sig12)
    feat = ks_feat(*agg)
    P1_cur = ks_price(coefs['P1'], feat)
    P2_cur = ks_price(coefs['P2'], feat)
    Q_cur  = np.array([ks_price(coefs[f'Q{k}'], feat) for k in range(N_SHOCKS)])

    records      = []
    rhoC_history = [[] for _ in range(J)]  # track rho_C for decoherence

    for t in range(T):
        mu1,mu2,sig1,sig2,sig12 = xsec_moments(alphas1, alphas2)
        agg  = (om_cur, z_cur, mu1, mu2, sig1, sig2, sig12)
        feat = ks_feat(*agg)

        agent_states = [(om_cur, z_cur, alphas1[j], alphas2[j], a_pays[j])
                        for j in range(J)]

        P1_star, P2_star, Q_star, new_a1, new_a2, new_ad, rho_Cs = \
            clear_markets(agent_states, V_interps, agg, coefs,
                          P1_cur, P2_cur, Q_cur)

        # Store rho_C for decoherence diagnostics
        for j in range(J):
            rhoC_history[j].append(rho_Cs[j].copy())

        mu1n = float(np.mean(new_a1));  mu2n = float(np.mean(new_a2))
        records.append({'feat': feat, 'P1': P1_star, 'P2': P2_star,
                        'Q': Q_star.copy(), 'mu1n': mu1n, 'mu2n': mu2n})

        # Shock realisation
        go_up_om = (np.random.rand() < 0.5)
        go_up_z  = (np.random.rand() < 0.5)
        k_star   = (0 if go_up_om else 2) + (0 if go_up_z else 1)

        alphas1 = new_a1.copy()
        alphas2 = new_a2.copy()
        a_pays  = new_ad[:, k_star]  # AD payoff = holdings in realised shock

        i_om = np.argmin(np.abs(omega_grid - om_cur))
        i_z  = np.argmin(np.abs(z_grid     - z_cur))
        i_om = omega_up[i_om] if go_up_om else omega_dn[i_om]
        i_z  = z_up[i_z]     if go_up_z  else z_dn[i_z]
        om_cur = omega_grid[i_om];  z_cur = z_grid[i_z]

        mu1n,mu2n,s1n,s2n,s12n = xsec_moments(alphas1, alphas2)
        agg_n = (om_cur, z_cur, mu1n, mu2n, s1n, s2n, s12n)
        f_n   = ks_feat(*agg_n)
        P1_cur = ks_price(coefs['P1'], f_n)
        P2_cur = ks_price(coefs['P2'], f_n)
        Q_cur  = np.array([ks_price(coefs[f'Q{k}'], f_n) for k in range(N_SHOCKS)])

        if verbose and t % 50 == 0:
            od = np.mean([
                np.sum(np.abs(rho_Cs[j] - np.diag(np.diag(rho_Cs[j]))))
                for j in range(J)])
            print(f"  sim t={t:3d}  P1={P1_star:.3f}  P2={P2_star:.3f}  "
                  f"rho_C offdiag={od:.4f}")

    return records, (alphas1, alphas2, a_pays), rhoC_history

# ═══════════════════════════════════════════════════════════════════════════════
# 14.  KRUSELL-SMITH OUTER LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_ks(n_jobs=N_JOBS, verbose=True):
    coefs    = init_coefs()
    V_agents = [np.zeros(ind_shape) for _ in range(J)]

    # Ergodic aggregate state from endowments
    mu1e,mu2e,s1e,s2e,s12e = xsec_moments(
        P.ENDOWMENTS[:,0], P.ENDOWMENTS[:,1])
    agg_erg = (float(np.mean(omega_grid)), float(np.mean(z_grid)),
               mu1e, mu2e, s1e, s2e, s12e)

    history = []

    for ks in range(P.max_ks):
        t0 = time.time()

        # ── VFI for each agent ────────────────────────────────────────────────
        P1g, P2g, Qg = price_grids(coefs, agg_erg)
        dV_max = 0.0;  fails_tot = 0;  iters_tot = 0

        for j in range(J):
            V_agents[j], nit, dV_j, nf = vfi_agent(
                V_agents[j], P1g, P2g, Qg, agg_erg, coefs,
                n_jobs=n_jobs, verbose=False)
            dV_max    = max(dV_max, dV_j)
            fails_tot += nf;  iters_tot += nit

        # ── Simulate ──────────────────────────────────────────────────────────
        records, final_states, rhoC_hist = simulate(
            V_agents, coefs, T=P.T_sim, verbose=False)

        # Update ergodic aggregate state
        a1f, a2f, _ = final_states
        mu1e,mu2e,s1e,s2e,s12e = xsec_moments(a1f, a2f)
        agg_erg = (float(np.mean(omega_grid)), float(np.mean(z_grid)),
                   mu1e, mu2e, s1e, s2e, s12e)

        # ── Regress ───────────────────────────────────────────────────────────
        coefs_new, r2 = update_coefs(records)
        dist          = coef_dist(coefs_new, coefs)
        coefs         = coefs_new

        # ── Decoherence diagnostic: off-diagonal norm of rho_C ───────────────
        od_mean = np.mean([
            np.mean([np.sum(np.abs(rc - np.diag(np.diag(rc))))
                     for rc in rhoC_hist[j]])
            for j in range(J)])
        od_final = np.mean([
            np.sum(np.abs(rhoC_hist[j][-1] - np.diag(np.diag(rhoC_hist[j][-1]))))
            for j in range(J)])

        elapsed = time.time() - t0
        rec = dict(ks=ks, dV=dV_max, dist=dist, r2=r2,
                   od_mean=od_mean, od_final=od_final,
                   vfi_iters=iters_tot, elapsed=elapsed)
        history.append(rec)

        if verbose:
            print(f"KS {ks:2d}  VFI_iters={iters_tot:3d}  |dV|={dV_max:.5f}  "
                  f"coef_dist={dist:.5f}  R2(P1)={r2['P1']:.4f}  "
                  f"rho_C_offdiag={od_mean:.4f}->{od_final:.4f}  "
                  f"t={elapsed:.0f}s")

        if dist < P.tol_ks:
            print(f"✓ KS converged at iteration {ks}.")
            break

    return V_agents, coefs, history

# ═══════════════════════════════════════════════════════════════════════════════
# 15.  DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def diagnostics(coefs):
    mu1,mu2,s1,s2,s12 = xsec_moments(P.ENDOWMENTS[:,0], P.ENDOWMENTS[:,1])
    rf_grid = np.zeros((P.N_omega, P.N_z))
    ep1     = np.zeros((P.N_omega, P.N_z))
    ep2     = np.zeros((P.N_omega, P.N_z))

    for i0,om in enumerate(omega_grid):
        for i1,z in enumerate(z_grid):
            f  = ks_feat(om, z, mu1, mu2, s1, s2, s12)
            P1 = ks_price(coefs['P1'], f)
            P2 = ks_price(coefs['P2'], f)
            Q  = np.array([ks_price(coefs[f'Q{k}'], f) for k in range(N_SHOCKS)])
            rf_grid[i0,i1] = 1.0/Q.sum() - 1.0

            P1n = 0.25*sum(ks_price(coefs['P1'], ks_feat(
                omega_grid[_om_dirs[k][i0]], z_grid[_z_dirs[k][i1]],
                mu1,mu2,s1,s2,s12)) for k in range(N_SHOCKS))
            P2n = 0.25*sum(ks_price(coefs['P2'], ks_feat(
                omega_grid[_om_dirs[k][i0]], z_grid[_z_dirs[k][i1]],
                mu1,mu2,s1,s2,s12)) for k in range(N_SHOCKS))
            ep1[i0,i1] = P1n/max(P1,1e-8) - 1.0 - rf_grid[i0,i1]
            ep2[i0,i1] = P2n/max(P2,1e-8) - 1.0 - rf_grid[i0,i1]

    return {'r_f': float(rf_grid.mean()),
            'ep1': float(ep1.mean()),
            'ep2': float(ep2.mean()),
            'rf_grid': rf_grid, 'ep1_grid': ep1}

def print_results(d, history):
    print("\n" + "═"*60)
    print("  RESULTS")
    print("═"*60)
    print(f"  Risk-free rate:       {d['r_f']:+.4f}")
    print(f"  Equity premium 1:     {d['ep1']:+.4f}")
    print(f"  Equity premium 2:     {d['ep2']:+.4f}")
    if history:
        h = history[-1]
        print(f"  KS iters:             {len(history)}")
        print(f"  Final coef_dist:      {h['dist']:.6f}")
        print(f"  Final R2(P1):         {h['r2']['P1']:.4f}")
        print(f"  rho_C offdiag (mean): {h['od_mean']:.4f}")
        print(f"  rho_C offdiag (end):  {h['od_final']:.4f}")
    print("═"*60)

# ═══════════════════════════════════════════════════════════════════════════════
# 16.  ENTRY POINT  (Windows: __main__ guard required for loky)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("═"*60)
    print("  Quantum Finance Equilibrium Solver  v4")
    print(f"  J={J}  kappa={P.kappa}  T_sim={P.T_sim}")
    print(f"  Grid {ind_shape}  N_states={N_ind}")
    print(f"  Endowments:\n{P.ENDOWMENTS}")
    print("═"*60)

    t0 = time.time()
    V_agents, coefs, history = run_ks(verbose=True)
    d = diagnostics(coefs)
    print_results(d, history)
    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
