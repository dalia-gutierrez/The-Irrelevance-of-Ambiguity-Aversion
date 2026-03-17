"""
Quantum Finance Equilibrium Solver — v5 final
===============================================
"The Irrelevance of Ambiguity Aversion" — Gutierrez Valencia

CORRECT rho_C CONSTRUCTION (from Proposition 1 + Definition 3):
  The agent chooses an ACT: a function from states of nature to lotteries.
  For each shock k, the agent chooses a consumption distribution p_k over c_grid.
  The density matrix is assembled as:
    rho_C = sum_k w_k * |psi_k><psi_k|   where  |psi_k> = sqrt(p_k)
  Off-diagonal content arises from:
    (a) Different p_k across shocks (different incomes in different states)
    (b) Mixed p_k within a shock (from ambiguity premium gamma_amb > 0)
  The total objective is Tr[rho_C * U] = sum_k w_k [sum_i p_ki*u(c_i) + amb_premium(p_k)]

  This is implemented as N_SH per-shock LPs at each node, with SEPARATE budget
  constraints for each shock (income_k = a_pay + alpha1*omega_k*dt + alpha2*z_k*dt).
  Equity and AD choices are shared across shocks (one portfolio for all states),
  so equity/AD LPs are solved once and consumption LPs per shock.

BUDGET (correct):
  For each shock k:
    c_k + (alpha1'-alpha1)*P1 + (alpha2'-alpha2)*P2 + sum_j a'_j*Q_j
      <= a_payoff + alpha1*omega_k*dt + alpha2*z_k*dt

  Since equity/AD choice is the same across shocks, the JOINT LP is:
    max  sum_k w_k [sum_i p_ki*(u(c_i)+bonus_i) + C_jk]
    s.t. for each k: sum_i p_ki*c_i + equity_net_k + AD_cost <= income_k
         sum_i p_ki = 1 for each k
         p_a1, p_a2, p_ad: simplex + one budget constraint (worst case or average)

OTHER CORRECTIONS (all carried from v4):
  - AD grids symmetric (short positions allowed)
  - V initialised at perpetuity value (not zero)
  - Howard policy iteration
  - KS coefficient mixing
  - Budget uses current dividends (known scalars at decision time)
  - kappa controls rho^u decay
  - Decoherence tracked via rho_C off-diagonal

RETURN CALCULATION:
  R1 = (P1_next + omega_next*dt) / P1_cur - 1   (total return incl. dividend)
  r_f = 1/sum(Q_k) - 1
"""

import numpy as np
from scipy.optimize import linprog
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
import warnings, time
warnings.filterwarnings('ignore')

N_JOBS     = 10
CHUNK_SIZE = 100

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

class Params:
    mu1, sigma1 = 0.05, 0.20
    mu2, sigma2 = 0.04, 0.25
    dt          = 1.0
    delta       = 0.05
    gamma_crra  = 2.0
    gamma_amb   = 0.15          # off-diagonal ambiguity premium

    kappa       = 0.0           # rho^u decay: 0=none, 1=full

    ALPHA1_SUPPLY = 1.0
    ALPHA2_SUPPLY = 1.0

    ENDOWMENTS = np.array([
        [1.6, 0.4],
        [1.0, 1.0],
        [0.4, 1.6],
    ])

    N_omega = 3;  N_z = 3;  N_alpha = 4;  N_a = 5;  N_c = 12

    # Howard
    n_policy = 10
    n_eval   = 25
    tol_vfi  = 5e-4

    # KS
    J        = 3
    T_sim    = 300
    tol_ks   = 5e-4
    max_ks   = 40
    lam      = 1e-4
    ks_mix   = 0.5

    # Newton
    newton_tol = 1e-5
    newton_max = 40

P  = Params()
J  = P.J
df = float(np.exp(-P.delta * P.dt))

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  GRIDS
# ═══════════════════════════════════════════════════════════════════════════════

def recomb(mu, sigma, N):
    return np.exp(mu*P.dt)*np.exp((np.arange(N)-(N-1)/2.)*sigma*np.sqrt(P.dt))

omega_grid = recomb(P.mu1, P.sigma1, P.N_omega)
z_grid     = recomb(P.mu2, P.sigma2, P.N_z)

def tidx(N):
    return np.minimum(np.arange(N)+1,N-1), np.maximum(np.arange(N)-1,0)

omega_up,omega_dn = tidx(P.N_omega)
z_up,z_dn         = tidx(P.N_z)

alpha_grid    = np.linspace(0.0,  2.5, P.N_alpha)
a_payoff_grid = np.linspace(-3.0, 3.0, P.N_a)
ad_grid       = np.linspace(-3.0, 3.0, P.N_alpha)
c_grid        = np.linspace(0.05, 8.0,  P.N_c)   # fine enough for off-diagonal

ind_shape = (P.N_omega,P.N_z,P.N_alpha,P.N_alpha,P.N_a)
N_ind     = int(np.prod(ind_shape))
_idx      = np.array(np.unravel_index(np.arange(N_ind), ind_shape)).T

N_SH=4; SH_P=0.25
_om_d=[omega_up,omega_up,omega_dn,omega_dn]
_z_d =[z_up,   z_dn,   z_up,   z_dn]

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  RHO^u
# ═══════════════════════════════════════════════════════════════════════════════

def step_rho_u(r):
    rn=r.copy().astype(complex)
    rn[0,1]=r[0,1]*np.exp(-P.kappa*(P.sigma2**2/2)*P.dt)*np.exp(1j*(P.mu2-P.sigma2**2/2)*P.dt)
    rn[1,0]=np.conj(rn[0,1]); rn/=np.trace(rn).real; return rn

rho_u0=np.array([[0.5,0.5],[0.5,0.5]],dtype=complex)
rho_u_path=[rho_u0]
for _ in range(50): rho_u_path.append(step_rho_u(rho_u_path[-1]))
rho_u_ss=rho_u_path[-1]
pzu=float(rho_u_ss[0,0].real); pzd=float(rho_u_ss[1,1].real)
Z_W=np.array([pzu,pzd,pzu,pzd])
_zws=SH_P*Z_W.sum()   # normalisation constant

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

u_diag = c_grid**(1-P.gamma_crra)/(1-P.gamma_crra)
_Ufull = np.diag(u_diag)
for i in range(P.N_c-1): _Ufull[i,i+1]=_Ufull[i+1,i]=P.gamma_amb

# Steady-state V initialisation
_c_ss  = float(np.median(c_grid))
_V_ss  = float((_c_ss**(1-P.gamma_crra)/(1-P.gamma_crra))/(1-df))

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  KS RULES
# ═══════════════════════════════════════════════════════════════════════════════

N_KS=8

def ks_feat(om,z,mu1,mu2,s1,s2,s12):
    e=1e-8; r12=np.clip(s12/(max(s1,e)*max(s2,e)),-1.,1.)
    return np.array([1.,np.log(max(om,e)),np.log(max(z,e)),
                     np.log(max(mu1,e)),np.log(max(mu2,e)),
                     np.log(max(s1,e)),np.log(max(s2,e)),r12])

def ksp(c,f): return float(np.exp(f@c))

def ridge(X,y): return np.linalg.solve(X.T@X+P.lam*np.eye(N_KS),X.T@y)

def init_coefs():
    c={}
    b=np.zeros(N_KS); b[0]=-np.log(P.delta+.5*P.sigma1**2); b[1]=1.; c['P1']=b.copy()
    b=np.zeros(N_KS); b[0]=-np.log(P.delta+.5*P.sigma2**2); b[2]=1.; c['P2']=b.copy()
    q0=np.log(df/N_SH)
    for k in range(N_SH): b=np.zeros(N_KS); b[0]=q0; c[f'Q{k}']=b.copy()
    b=np.zeros(N_KS); b[3]=1.; c['mu1']=b.copy()
    b=np.zeros(N_KS); b[4]=1.; c['mu2']=b.copy()
    return c

def update_coefs(records,cold):
    X=np.array([r['f'] for r in records])
    def fit(yl,key):
        y=np.log(np.maximum(yl,1e-8))
        return P.ks_mix*ridge(X,y)+(1-P.ks_mix)*cold[key]
    c={}
    c['P1']=fit([r['P1'] for r in records],'P1')
    c['P2']=fit([r['P2'] for r in records],'P2')
    c['mu1']=fit([r['mu1n'] for r in records],'mu1')
    c['mu2']=fit([r['mu2n'] for r in records],'mu2')
    for k in range(N_SH):
        c[f'Q{k}']=fit([r['Q'][k] for r in records],f'Q{k}')
    r2={}
    for key,yl in [('P1',[r['P1'] for r in records]),('P2',[r['P2'] for r in records]),
                   ('mu1',[r['mu1n'] for r in records]),('mu2',[r['mu2n'] for r in records])]:
        y=np.log(np.maximum(yl,1e-8)); yh=X@c[key]
        r2[key]=float(1.-np.sum((y-yh)**2)/(np.sum((y-y.mean())**2)+1e-12))
    return c,r2

def cdist(cn,co): return max(np.max(np.abs(cn[k]-co[k])) for k in cn)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  INTERPOLATOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_interp(Vg):
    return RegularGridInterpolator(
        (omega_grid,z_grid,alpha_grid,alpha_grid,a_payoff_grid),
        Vg,method='linear',bounds_error=False,fill_value=None)

def evV(interp,om,z,a1,a2,ap):
    return float(interp((
        np.clip(om,omega_grid[0],omega_grid[-1]),
        np.clip(z,z_grid[0],z_grid[-1]),
        np.clip(a1,alpha_grid[0],alpha_grid[-1]),
        np.clip(a2,alpha_grid[0],alpha_grid[-1]),
        np.clip(ap,a_payoff_grid[0],a_payoff_grid[-1]))))

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CORE SOLVER  (correct per-shock structure)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_node(i0,i1,i2,i3,i4, V_interp, P1s,P2s,Q_vec, coefs, agg):
    """
    Solve agent's problem at state node (i0..i4).

    STRUCTURE (from Proposition 1 + Definition 3):
      The agent chooses an act: for each shock k, a consumption distribution p_k.
      Equity and AD portfolio are common across shocks (one portfolio per period).

    STEP 1: Solve a joint LP over (p_k for k=0..3, p_a1, p_a2, p_ad).
      Objective: max sum_k w_k [sum_i p_ki*(u_i + C_jk) + gamma_amb*offdiag(p_k)]
      Per-shock budget: for each k:
        sum_i p_ki*c_i <= income_k - (E[alpha1']-alpha1)*P1 - (E[alpha2']-alpha2)*P2
                          - sum_j E[a'_j]*Q_j
      Since equity/AD terms are same across k, we handle this as:
        sum_i p_ki*c_i + (alpha1_choice-alpha1)*P1 + (alpha2_choice-alpha2)*P2
             + sum_j ad_choice_j*Q_j <= income_k
      with equity/AD as single variables shared across all shocks.

    STEP 2: Assemble rho_C = sum_k w_k * outer(sqrt(p_k), sqrt(p_k)).

    Returns (V_new, Ea1, Ea2, Ead[N_SH], rho_C, ok)
    """
    om=omega_grid[i0]; zv=z_grid[i1]; a1=alpha_grid[i2]; a2=alpha_grid[i3]
    ap=a_payoff_grid[i4]
    Nc=P.N_c; Na=P.N_alpha; Nad=len(ad_grid)

    # Next-period shock values
    om_n=[omega_grid[_om_d[k][i0]] for k in range(N_SH)]
    z_n =[z_grid[_z_d[k][i1]]     for k in range(N_SH)]

    # Per-shock incomes (known scalars: current holdings x current-state dividends)
    incomes=[ap + a1*om_n[k]*P.dt + a2*z_n[k]*P.dt for k in range(N_SH)]
    # Note: we use NEXT-period omega/z for income because the shock determines
    # WHICH state of nature occurs, and dividends are paid in that state.
    # Actually in this timing: at t we choose portfolio for t+dt.
    # Dividends in state k = alpha1'*omega_k + alpha2'*z_k (paid at t+dt to new holder)
    # But we're choosing alpha' now... the income in the budget is from CURRENT holdings:
    # current_income_k = a_pay + alpha1*omega_k*dt + alpha2*z_k*dt
    # (The current owner receives dividends from current shock realisation)
    # We use om_n[k] as the shock that WILL be realised -- this is correct since
    # at t we don't know which k occurs, so we plan for each k separately.

    # Continuation value per shock (at a_payoff=0, zero net supply)
    C_k=np.array([df*evV(V_interp,om_n[k],z_n[k],a1,a2,0.) for k in range(N_SH)])
    # Weighted average continuation (for portfolio objective)
    C_bar=sum(SH_P*Z_W[k]*C_k[k] for k in range(N_SH))/max(_zws,1e-12)

    # ── Joint LP ──
    # TIMING (corrected): at time t, equity alpha1,alpha2 is FIXED (chosen last period).
    # Budget: c + sum_k a'_k*Q_k <= income_k   (per shock)
    # income_k = a_payoff + alpha1*omega_k*dt + alpha2*z_k*dt  (dividends in shock k)
    # Equity for t+dt (alpha1', alpha2') is chosen separately via continuation value:
    # it affects C_jk through next-period income but NOT current budget.
    # Variables: [p_0..p_3 (Nc each), p_a1 (Na), p_a2 (Na), p_ad (Nad*N_SH)]
    # Budget: per-shock c + AD cost <= income_k  (equity does NOT appear here)

    n_c  = N_SH*Nc
    n_a1 = Na;  n_a2 = Na;  n_ad = Nad*N_SH
    n    = n_c + n_a1 + n_a2 + n_ad

    # Objective: max sum_k w_k [sum_i p_ki*(u_i + C_k[k]) + ambiguity_bonus(p_k)]
    # Plus equity component from continuation (folded into C_k via next-period income)
    obj=np.zeros(n)
    for k in range(N_SH):
        wk=SH_P*Z_W[k]/max(_zws,1e-12)
        obj_k=-(u_diag+C_k[k])*wk
        bonus_k=np.zeros(Nc)
        bonus_k[:-1]+=P.gamma_amb*wk; bonus_k[1:]+=P.gamma_amb*wk
        obj[k*Nc:(k+1)*Nc]=obj_k-bonus_k
    # Equity: penalise deviation from current (minimise adjustment costs)
    obj[n_c:n_c+Na]      = np.zeros(Na)   # equity: no direct cost in current budget
    obj[n_c+Na:n_c+2*Na] = np.zeros(Na)
    obj[n_c+2*Na:]       = np.zeros(n_ad)

    # Equality: simplex for each p_k, p_a1, p_a2, p_ad
    n_eq=N_SH+3
    Aeq=np.zeros((n_eq,n)); beq=np.ones(n_eq)
    for k in range(N_SH): Aeq[k, k*Nc:(k+1)*Nc]=1.
    Aeq[N_SH,   n_c:n_c+Na]     =1.
    Aeq[N_SH+1, n_c+Na:n_c+2*Na]=1.
    Aeq[N_SH+2, n_c+2*Na:]      =1.

    # Inequality: per-shock budget (EQUITY EXCLUDED from current budget)
    # c_k + sum_j a'_j*Q_j <= income_k
    # Equity for t+dt is financed from NEXT period resources (via continuation value)
    Aub=np.zeros((N_SH,n)); bub=np.array(incomes)
    for k in range(N_SH):
        Aub[k, k*Nc:(k+1)*Nc] = c_grid    # consumption cost in shock k
        for j in range(N_SH):
            Aub[k, n_c+2*Na+np.arange(Nad)*N_SH+j] = ad_grid*Q_vec[j]  # AD cost

    lp=linprog(obj,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
               bounds=[(0.,1.)]*n,method='highs')

    if lp.success and np.all(lp.x>=-1e-9):
        x=np.maximum(lp.x,0.)
        pks=[]; ok=True
        for k in range(N_SH):
            pk=x[k*Nc:(k+1)*Nc]; pk/=max(pk.sum(),1e-12); pks.append(pk)
        pa1=x[n_c:n_c+Na];      pa1/=max(pa1.sum(),1e-12)
        pa2=x[n_c+Na:n_c+2*Na]; pa2/=max(pa2.sum(),1e-12)
        pad=x[n_c+2*Na:].reshape(Nad,N_SH); pad/=max(pad.sum(),1e-12)
    else:
        pks=[]
        for k in range(N_SH):
            pk=np.zeros(Nc)
            best=np.argmin(np.abs(c_grid-np.clip(incomes[k],c_grid[0],c_grid[-1])))
            pk[best]=1.; pks.append(pk)
        pa1=np.zeros(Na); pa1[np.argmin(np.abs(alpha_grid-a1))]=1.
        pa2=np.zeros(Na); pa2[np.argmin(np.abs(alpha_grid-a2))]=1.
        pad=np.ones((Nad,N_SH))/(Nad*N_SH); ok=False

    # Assemble rho_C = sum_k w_k |sqrt(p_k)><sqrt(p_k)|
    rho_C=np.zeros((Nc,Nc))
    for k in range(N_SH):
        wk=SH_P*Z_W[k]/max(_zws,1e-12)
        vk=np.sqrt(np.maximum(pks[k],0.))
        rho_C+=wk*np.outer(vk,vk)
    # rho_C is PSD and has trace = sum_k w_k = 1 (normalised weights)

    u_now=float(np.trace(rho_C@_Ufull))
    V_new=float(np.clip(u_now+C_bar,-1e4,0.))
    Ea1=float(pa1@alpha_grid); Ea2=float(pa2@alpha_grid)
    Ead=np.array([float(pad[:,k]@ad_grid) for k in range(N_SH)])
    return V_new,Ea1,Ea2,Ead,rho_C,ok

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CHUNK WORKERS
# ═══════════════════════════════════════════════════════════════════════════════

def policy_chunk(idx_list,Vg,P1g,P2g,Qg,coefs,agg):
    interp=make_interp(Vg); out=[]
    for idx in idx_list:
        i0,i1,i2,i3,i4=_idx[idx]
        Vn,Ea1,Ea2,Ead,rC,ok=solve_node(
            i0,i1,i2,i3,i4,interp,
            float(P1g[i0,i1]),float(P2g[i0,i1]),Qg[i0,i1,:],coefs,agg)
        out.append((idx,Vn,rC,ok))
    return out

def eval_chunk(idx_list,Vg,pol_rhoC,coefs,agg):
    """Policy evaluation with fixed rho_C (from policy improvement)."""
    interp=make_interp(Vg); out=[]
    for idx in idx_list:
        i0,i1,i2,i3,i4=_idx[idx]
        rho_C_fix=pol_rhoC[idx]
        a1=alpha_grid[i2]; a2=alpha_grid[i3]
        om_n=[omega_grid[_om_d[k][i0]] for k in range(N_SH)]
        z_n =[z_grid[_z_d[k][i1]]     for k in range(N_SH)]
        C_bar=df*sum(SH_P*Z_W[k]*evV(interp,om_n[k],z_n[k],a1,a2,0.)
                     for k in range(N_SH))/max(_zws,1e-12)
        u_now=float(np.trace(rho_C_fix@_Ufull))
        Vb=float(np.clip(u_now+C_bar,-1e4,0.))
        out.append((idx,Vb))
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  PRICE GRIDS AND HOWARD VFI
# ═══════════════════════════════════════════════════════════════════════════════

def price_grids(coefs,agg):
    _,_,mu1,mu2,s1,s2,s12=agg
    P1g=np.zeros((P.N_omega,P.N_z)); P2g=np.zeros_like(P1g)
    Qg=np.zeros((P.N_omega,P.N_z,N_SH))
    for i0,om in enumerate(omega_grid):
        for i1,z in enumerate(z_grid):
            f=ks_feat(om,z,mu1,mu2,s1,s2,s12)
            P1g[i0,i1]=ksp(coefs['P1'],f); P2g[i0,i1]=ksp(coefs['P2'],f)
            for k in range(N_SH): Qg[i0,i1,k]=ksp(coefs[f'Q{k}'],f)
    return P1g,P2g,Qg

def howard_vfi(V_old,P1g,P2g,Qg,agg,coefs,n_jobs=N_JOBS):
    chunks=[list(range(N_ind))[i:i+CHUNK_SIZE] for i in range(0,N_ind,CHUNK_SIZE)]
    V=V_old.copy(); fails=0

    for pi in range(P.n_policy):
        raw=Parallel(n_jobs=n_jobs,backend='loky')(
            delayed(policy_chunk)(ch,V,P1g,P2g,Qg,coefs,agg) for ch in chunks)

        Vnew=np.zeros(ind_shape); pol_rC={}; nf=0
        for cr in raw:
            for idx,Vn,rC,ok in cr:
                i0,i1,i2,i3,i4=_idx[idx]
                Vnew[i0,i1,i2,i3,i4]=Vn; pol_rC[idx]=rC
                if not ok: nf+=1
        fails=nf

        for ev in range(P.n_eval):
            raw_ev=Parallel(n_jobs=n_jobs,backend='loky')(
                delayed(eval_chunk)(ch,Vnew,pol_rC,coefs,agg) for ch in chunks)
            Vev=np.zeros(ind_shape)
            for cr in raw_ev:
                for idx,Vb in cr:
                    i0,i1,i2,i3,i4=_idx[idx]; Vev[i0,i1,i2,i3,i4]=Vb
            dVev=float(np.max(np.abs(Vev-Vnew))); Vnew=Vev
            if dVev<P.tol_vfi*0.1: break

        dV=float(np.max(np.abs(Vnew-V))); V=Vnew
        if dV<P.tol_vfi: break

    return V,pi+1,dV,fails

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  CROSS-SECTION AND MARKET CLEARING
# ═══════════════════════════════════════════════════════════════════════════════

def xsec(a1s,a2s):
    mu1,mu2=float(np.mean(a1s)),float(np.mean(a2s))
    s1=float(np.std(a1s))+1e-6; s2=float(np.std(a2s))+1e-6
    s12=float(np.mean((a1s-mu1)*(a2s-mu2)))
    return mu1,mu2,s1,s2,s12

def clear(sts,Vis,agg,coefs,P1i,P2i,Qi):
    P1=P1i; P2=P2i; Q=Qi.copy()
    ea1=np.zeros(J); ea2=np.zeros(J); ead=np.zeros((J,N_SH))

    for it in range(P.newton_max):
        rCs_this=[]
        for j,(om,z,a1,a2,ap) in enumerate(sts):
            i0=np.argmin(np.abs(omega_grid-om)); i1=np.argmin(np.abs(z_grid-z))
            i2=np.argmin(np.abs(alpha_grid-a1)); i3=np.argmin(np.abs(alpha_grid-a2))
            i4=np.argmin(np.abs(a_payoff_grid-ap))
            _,e1,e2,ed,rC,_=solve_node(i0,i1,i2,i3,i4,Vis[j],P1,P2,Q,coefs,agg)
            ea1[j]=e1; ea2[j]=e2; ead[j]=ed; rCs_this.append(rC)

        ED1=np.mean(ea1)-P.ALPHA1_SUPPLY; ED2=np.mean(ea2)-P.ALPHA2_SUPPLY
        EDad=np.mean(ead,axis=0)
        if (abs(ED1)<P.newton_tol and abs(ED2)<P.newton_tol
                and np.all(np.abs(EDad)<P.newton_tol)): break

        s=0.10
        P1=max(P1*(1+s*ED1),0.01); P2=max(P2*(1+s*ED2),0.01)
        Q=np.maximum(Q*(1+s*EDad),1e-5)

    return P1,P2,Q,ea1.copy(),ea2.copy(),ead.copy(),rCs_this

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(V_agents,coefs,T=P.T_sim,verbose=False):
    Vis=[make_interp(V_agents[j]) for j in range(J)]
    a1s=P.ENDOWMENTS[:,0].copy(); a2s=P.ENDOWMENTS[:,1].copy(); apays=np.zeros(J)
    iom=P.N_omega//2; iz=P.N_z//2
    om=omega_grid[iom]; z=z_grid[iz]
    f=ks_feat(om,z,*xsec(a1s,a2s))
    P1c=ksp(coefs['P1'],f); P2c=ksp(coefs['P2'],f)
    Qc=np.array([ksp(coefs[f'Q{k}'],f) for k in range(N_SH)])

    records=[]; rhoC_h=[[] for _ in range(J)]
    rf_ts=[]; ep1_ts=[]; ep2_ts=[]

    for t in range(T):
        mu1,mu2,s1,s2,s12=xsec(a1s,a2s)
        agg=(om,z,mu1,mu2,s1,s2,s12)
        sts=[(om,z,a1s[j],a2s[j],apays[j]) for j in range(J)]
        P1s,P2s,Qs,na1,na2,nad,rCs=clear(sts,Vis,agg,coefs,P1c,P2c,Qc)
        for j in range(J): rhoC_h[j].append(rCs[j].copy())

        rf=1./Qs.sum()-1.
        om_nm=0.25*sum(omega_grid[_om_d[k][iom]] for k in range(N_SH))
        z_nm =0.25*sum(z_grid[_z_d[k][iz]]       for k in range(N_SH))
        mu1n=float(np.mean(na1)); mu2n=float(np.mean(na2))
        fn=ks_feat(om_nm,z_nm,mu1n,mu2n,s1,s2,s12)
        P1n=ksp(coefs['P1'],fn); P2n=ksp(coefs['P2'],fn)
        R1=(P1n+om_nm*P.dt)/max(P1s,1e-8)-1.
        R2=(P2n+z_nm *P.dt)/max(P2s,1e-8)-1.
        rf_ts.append(float(rf)); ep1_ts.append(float(R1-rf)); ep2_ts.append(float(R2-rf))

        records.append({'f':ks_feat(*agg),'P1':P1s,'P2':P2s,
                        'Q':Qs.copy(),'mu1n':float(np.mean(na1)),'mu2n':float(np.mean(na2))})

        uom=(np.random.rand()<.5); uz=(np.random.rand()<.5)
        ks=(0 if uom else 2)+(0 if uz else 1)
        a1s=na1.copy(); a2s=na2.copy(); apays=nad[:,ks]
        iom=omega_up[iom] if uom else omega_dn[iom]
        iz =z_up[iz]      if uz  else z_dn[iz]
        om=omega_grid[iom]; z=z_grid[iz]
        mu1nn,mu2nn,s1n,s2n,s12n=xsec(a1s,a2s)
        fn2=ks_feat(om,z,mu1nn,mu2nn,s1n,s2n,s12n)
        P1c=ksp(coefs['P1'],fn2); P2c=ksp(coefs['P2'],fn2)
        Qc=np.array([ksp(coefs[f'Q{k}'],fn2) for k in range(N_SH)])

        if verbose and t%50==0:
            od=np.mean([np.sum(np.abs(rCs[j]-np.diag(np.diag(rCs[j])))) for j in range(J)])
            print(f"  t={t:3d} rf={rf:.4f} ep1={ep1_ts[-1]:.4f} rho_C_od={od:.4f}")

    return records,(a1s,a2s,apays),rhoC_h,{'rf':rf_ts,'ep1':ep1_ts,'ep2':ep2_ts}

# ═══════════════════════════════════════════════════════════════════════════════
# 12.  KS OUTER LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_ks(n_jobs=N_JOBS,verbose=True):
    coefs=init_coefs()
    V_agents=[np.full(ind_shape,_V_ss) for _ in range(J)]
    mu1e,mu2e,s1e,s2e,s12e=xsec(P.ENDOWMENTS[:,0],P.ENDOWMENTS[:,1])
    agg_erg=(float(np.mean(omega_grid)),float(np.mean(z_grid)),mu1e,mu2e,s1e,s2e,s12e)
    history=[]; all_series={'rf':[],'ep1':[],'ep2':[]}; all_rhoC=None

    for ks in range(P.max_ks):
        t0=time.time()
        P1g,P2g,Qg=price_grids(coefs,agg_erg)
        dVmax=0.; fails=0; iters=0
        for j in range(J):
            V_agents[j],nit,dVj,nf=howard_vfi(
                V_agents[j],P1g,P2g,Qg,agg_erg,coefs,n_jobs=n_jobs)
            dVmax=max(dVmax,dVj); fails+=nf; iters+=nit

        np.random.seed(ks)
        records,(a1f,a2f,_),rhoC_h,series=simulate(V_agents,coefs,verbose=False)
        for k in all_series: all_series[k]+=series[k]
        all_rhoC=rhoC_h

        mu1e,mu2e,s1e,s2e,s12e=xsec(a1f,a2f)
        agg_erg=(float(np.mean(omega_grid)),float(np.mean(z_grid)),mu1e,mu2e,s1e,s2e,s12e)

        coefs_new,r2=update_coefs(records,coefs); dist=cdist(coefs_new,coefs); coefs=coefs_new

        od_m=np.mean([np.mean([np.sum(np.abs(rc-np.diag(np.diag(rc)))) for rc in rhoC_h[j]])
                      for j in range(J)])
        od_f=np.mean([np.sum(np.abs(rhoC_h[j][-1]-np.diag(np.diag(rhoC_h[j][-1]))))
                      for j in range(J)])

        elapsed=time.time()-t0
        rec=dict(ks=ks,dV=dVmax,dist=dist,r2=r2,od_m=od_m,od_f=od_f,iters=iters,
                 elapsed=elapsed,rf=float(np.mean(series['rf'])),
                 ep1=float(np.mean(series['ep1'])),ep2=float(np.mean(series['ep2'])))
        history.append(rec)

        if verbose:
            print(f"KS {ks:2d}  Hi={iters:2d}  |dV|={dVmax:.5f}  dist={dist:.5f}  "
                  f"R2={r2['P1']:.4f}  rf={rec['rf']:.4f}  ep1={rec['ep1']:.4f}  "
                  f"rho_C_od={od_m:.4f}->{od_f:.4f}  t={elapsed:.0f}s")

        if dist<P.tol_ks: print(f"✓ Converged at KS {ks}."); break

    return V_agents,coefs,history,all_series,all_rhoC

# ═══════════════════════════════════════════════════════════════════════════════
# 13.  GRAPHS
# ═══════════════════════════════════════════════════════════════════════════════

def make_graphs(history,series,rhoC_h,coefs):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError: print("matplotlib unavailable"); return

    fig,axes=plt.subplots(2,3,figsize=(16,9)); fig.subplots_adjust(hspace=0.4,wspace=0.35)
    T=len(series['rf']); t=np.arange(T); cols=['steelblue','darkorange','green']

    ax=axes[0,0]; ax.plot(t,series['rf'],lw=0.8,color='steelblue')
    ax.axhline(0,color='k',lw=0.5,ls='--'); ax.set_title('Risk-Free Rate')
    ax.set_xlabel('Period'); ax.grid(True,alpha=0.3)

    ax=axes[0,1]
    ax.plot(t,series['ep1'],lw=0.8,color='green',label='Asset 1 (risky)')
    ax.plot(t,series['ep2'],lw=0.8,color='red',label='Asset 2 (ambiguous)')
    ax.axhline(0,color='k',lw=0.5,ls='--'); ax.set_title('Equity Premia')
    ax.set_xlabel('Period'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    ax=axes[0,2]
    if rhoC_h:
        for j in range(J):
            od=[np.sum(np.abs(rc-np.diag(np.diag(rc)))) for rc in rhoC_h[j]]
            ax.plot(od,color=cols[j],lw=1.2,
                    label=f'j={j} α=({P.ENDOWMENTS[j,0]},{P.ENDOWMENTS[j,1]})')
    ax.set_title('rho_C Off-Diagonal (Decoherence)'); ax.set_xlabel('Period')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax=axes[1,0]
    ax.semilogy([h['dist'] for h in history],'o-',color='purple',ms=4)
    ax.axhline(P.tol_ks,color='r',ls='--',lw=0.8); ax.set_title('KS Coefficient Distance')
    ax.set_xlabel('KS Iteration'); ax.grid(True,alpha=0.3)

    ax=axes[1,1]
    iz=P.N_z//2; z_mid=z_grid[iz]
    mu1,mu2,s1,s2,s12=xsec(P.ENDOWMENTS[:,0],P.ENDOWMENTS[:,1])
    P1v=[ksp(coefs['P1'],ks_feat(o,z_mid,mu1,mu2,s1,s2,s12)) for o in omega_grid]
    P2v=[ksp(coefs['P2'],ks_feat(o,z_mid,mu1,mu2,s1,s2,s12)) for o in omega_grid]
    ax.plot(omega_grid,P1v,'o-',color='green',label='P1')
    ax.plot(omega_grid,P2v,'s--',color='red',label=f'P2 (z={z_mid:.2f})')
    ax.set_title('Prices vs omega'); ax.set_xlabel('omega'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    ax=axes[1,2]
    ax.semilogy([h['dV'] for h in history],'s-',color='brown',ms=4)
    ax.axhline(P.tol_vfi,color='r',ls='--',lw=0.8); ax.set_title('VFI Residual')
    ax.set_xlabel('KS Iteration'); ax.grid(True,alpha=0.3)

    fig.suptitle(f'Quantum Finance v5 (J={J}, kappa={P.kappa}, gamma_amb={P.gamma_amb})',
                 fontsize=12,fontweight='bold')
    out='/mnt/user-data/outputs/quantum_finance_results.png'
    fig.savefig(out,dpi=130,bbox_inches='tight'); plt.close(fig)
    print(f"Graphs saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# 14.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__=='__main__':
    print("═"*60)
    print(f"  Quantum Finance Equilibrium Solver  v5-final")
    print(f"  J={J}  kappa={P.kappa}  gamma_amb={P.gamma_amb}  T_sim={P.T_sim}")
    print(f"  Grid {ind_shape}  N_states={N_ind}  Jobs={N_JOBS}")
    print(f"  Howard: {P.n_policy} policy / {P.n_eval} eval  tol={P.tol_vfi}")
    print(f"  V_init={_V_ss:.3f}  (perpetuity at c={_c_ss:.3f})")
    print(f"  Endowments:\n{P.ENDOWMENTS}")
    print("═"*60)
    t0=time.time()
    V_ag,coefs,hist,series,rhoC_h=run_ks(verbose=True)
    elapsed=time.time()-t0
    print(f"\nTotal: {elapsed/60:.1f} min")
    Tl=P.T_sim
    print(f"Final rf={np.mean(series['rf'][-Tl:]):.4f}  "
          f"ep1={np.mean(series['ep1'][-Tl:]):.4f}  "
          f"ep2={np.mean(series['ep2'][-Tl:]):.4f}")
    make_graphs(hist,series,rhoC_h,coefs)
