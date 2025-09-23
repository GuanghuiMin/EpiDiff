import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import nbinom


def eta_to_theta(eta):
    """eta = (log R0, log Tinf)  →  theta = (beta, gamma)"""
    logR0, logTinf = float(eta[0]), float(eta[1])
    R0   = np.exp(logR0)
    Tinf = np.exp(logTinf)
    Tinf = np.clip(Tinf, 1e-3, 1e6)
    beta  = R0 / Tinf
    gamma = 1.0 / Tinf
    return np.array([beta, gamma], dtype=float)

def clamp_eta(eta, bounds_logR0_logTinf):
    (lb_R0, ub_R0), (lb_T, ub_T) = bounds_logR0_logTinf
    return np.array([np.clip(eta[0], lb_R0, ub_R0),
                     np.clip(eta[1], lb_T,  ub_T )], dtype=float)

def sir_ode(state, t, beta, gamma):
    S, I, R = state
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR], dtype=float)

def _sanitize_state(S0, I0, R0):
    S0 = max(float(S0), 1e-10)
    I0 = max(float(I0), 1e-12)
    R0 = max(float(R0), 0.0)
    total = S0 + I0 + R0
    if total <= 0:
        return np.array([0.99, 0.01, 0.0], dtype=float)
    return np.array([S0/total, I0/total, R0/total], dtype=float)

def forward_incidence_given_init(k, x0_norm, N, theta):
    beta, gamma = map(float, theta)
    if k <= 0: return np.array([])
    t = np.arange(k, dtype=float)
    sol = odeint(sir_ode, x0_norm, t, args=(beta, gamma), rtol=1e-6, atol=1e-8, mxstep=5000)
    S_t = np.clip(sol[:, 0], 1e-12, 1.0)
    I_t = np.clip(sol[:, 1], 1e-12, 1.0)
    mu = beta * S_t * I_t * float(N)
    return np.maximum(mu, 1e-12)

def forward_incidence_future(init_state, pop, h, theta):
    beta, gamma = map(float, theta)
    if h <= 0: return np.array([])
    x0 = _sanitize_state(*init_state)
    t = np.arange(h, dtype=float)
    sol = odeint(sir_ode, x0, t, args=(beta, gamma), rtol=1e-6, atol=1e-8, mxstep=5000)
    S_t = np.clip(sol[:, 0], 1e-10, 1.0)
    I_t = np.clip(sol[:, 1], 1e-12, 1.0)
    mu = beta * S_t * I_t * pop
    return np.maximum(mu, 0.0)


def nb_nll(y, mu, alpha):
    if alpha <= 0: return 1e10
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        y = np.round(y).astype(int)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-10, 1e12)
    p  = np.clip(alpha / (alpha + mu), 1e-8, 1-1e-8)
    try:
        nll = -np.sum(nbinom.logpmf(y, alpha, p))
        return float(nll) if np.isfinite(nll) else 1e10
    except Exception:
        return 1e10

def neg_log_prior_eta(eta, mu, sd):
    z = (eta - mu) / sd
    return 0.5*np.sum(z**2) + np.sum(np.log(sd)) + len(eta)*0.5*np.log(2*np.pi)


def map_estimate_eta_fixed_init(
    y_hist, x0_norm, N, alpha_nb=5.0,
    prior_mu=None, prior_sd=None,
    bounds_logR0_logTinf=((np.log(0.8), np.log(5.0)), (np.log(2.0), np.log(10.0))),
    maxiter=150, ftol=1e-6
):
    if prior_mu is None or prior_sd is None:
        (lbR, ubR), (lbT, ubT) = bounds_logR0_logTinf
        mu_R = 0.5*(lbR+ubR); mu_T = 0.5*(lbT+ubT)
        sd_R = max((ubR-lbR)/2.0, 0.5)
        sd_T = max((ubT-lbT)/2.0, 0.5)
        prior_mu = np.array([mu_R, mu_T], dtype=float)
        prior_sd = np.array([sd_R, sd_T], dtype=float)

    def objective(eta):
        eta_c = clamp_eta(eta, bounds_logR0_logTinf)
        theta = eta_to_theta(eta_c)
        mu_tr = forward_incidence_given_init(len(y_hist), x0_norm, N, theta)
        return nb_nll(y_hist, mu_tr, alpha_nb) + neg_log_prior_eta(eta_c, prior_mu, prior_sd)

    inits = [
        np.array([0.5*(bounds_logR0_logTinf[0][0]+bounds_logR0_logTinf[0][1]),
                  0.5*(bounds_logR0_logTinf[1][0]+bounds_logR0_logTinf[1][1])]),
        np.array([np.log(1.5), np.log(5.0)]),
        np.array([np.log(2.0), np.log(4.0)]),
        np.array([np.log(3.0), np.log(3.0)]),
    ]
    best, best_val = None, np.inf
    for x0 in inits:
        res = minimize(objective, x0, method='L-BFGS-B',
                       bounds=bounds_logR0_logTinf,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success and np.isfinite(res.fun) and res.fun < best_val:
            best, best_val = res, float(res.fun)

    eta_hat = clamp_eta(best.x, bounds_logR0_logTinf) if best is not None else np.array([np.log(2.0), np.log(5.0)])
    return eta_hat, prior_mu, prior_sd

def numerical_hessian(f, x, eps=5e-4):
    x = np.asarray(x, dtype=float); d = len(x)
    H = np.zeros((d, d), dtype=float)
    f0 = float(f(x))
    for i in range(d):
        hi = eps * (1 + abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += hi; xm[i] -= hi
        H[i, i] = (f(xp) - 2*f0 + f(xm)) / (hi*hi)
    for i in range(d):
        for j in range(i+1, d):
            hi = eps*(1+abs(x[i])); hj = eps*(1+abs(x[j]))
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i]+=hi; xpp[j]+=hj
            xpm[i]+=hi; xpm[j]-=hj
            xmp[i]-=hi; xmp[j]+=hj
            xmm[i]-=hi; xmm[j]-=hj
            H[i,j] = (f(xpp)-f(xpm)-f(xmp)+f(xmm))/(4*hi*hj); H[j,i]=H[i,j]
    return H

def laplace_cov_eta_fixed_init(y_hist, x0_norm, N, eta_hat, alpha_nb=5.0, prior_mu=None, prior_sd=None, hess_eps=5e-4):
    def objective(eta):
        theta = eta_to_theta(eta)
        mu_tr = forward_incidence_given_init(len(y_hist), x0_norm, N, theta)
        return nb_nll(y_hist, mu_tr, alpha_nb) + neg_log_prior_eta(eta, prior_mu, prior_sd)

    H = numerical_hessian(objective, eta_hat, eps=hess_eps)
    w, V = np.linalg.eigh(H)
    w = np.clip(w, 1e-8, None)
    H_pd = V @ np.diag(w) @ V.T
    try:
        Sigma = np.linalg.inv(H_pd)
    except np.linalg.LinAlgError:
        diag = np.clip(np.diag(H_pd), 1e-8, None)
        Sigma = np.diag(1.0/diag)
    w2, V2 = np.linalg.eigh(Sigma)
    w2 = np.clip(w2, 1e-10, 10.0)
    Sigma = V2 @ np.diag(w2) @ V2.T
    return Sigma

def ut_sigma_points(mu, Sigma, alpha=0.2, beta=2.0, kappa=None):
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    L = len(mu)
    if kappa is None: kappa = 3.0 - L
    lam = alpha**2 * (L + kappa) - L

    e, U = np.linalg.eigh(Sigma)
    e = np.maximum(e, 1e-10)
    Sigma_pd = U @ np.diag(e) @ U.T

    scale = L + lam
    if scale <= 0: scale = 1e-6
    try:
        S = np.linalg.cholesky(scale * Sigma_pd)
    except np.linalg.LinAlgError:
        S = U @ np.diag(np.sqrt(e * scale))

    pts = np.zeros((2*L + 1, L)); pts[0] = mu
    for i in range(L):
        pts[i+1]   = mu + S[:, i]
        pts[i+1+L] = mu - S[:, i]

    wm = np.zeros(2*L + 1); wc = np.zeros(2*L + 1)
    wm[0] = lam / (L + lam)
    wc[0] = wm[0] + (1.0 - alpha**2 + beta)
    wm[1:] = wc[1:] = 1.0/(2.0*(L+lam))
    return pts, wm, wc

def robust_filter_trajectories(samples, trim_frac=0.15, max_neg_frac=0.0):
    n, h = samples.shape
    neg_mask = (samples < 0).sum(axis=1) / max(1, h)
    ok_neg = neg_mask <= max_neg_frac
    X = np.log1p(np.maximum(samples, 0.0))
    med = np.median(X, axis=0, keepdims=True)
    dist = np.sqrt(np.sum((X - med)**2, axis=1))
    keep = np.ones(n, dtype=bool)
    k_trim = int(np.floor(trim_frac * n))
    if k_trim > 0:
        idx = np.argsort(dist)[::-1]
        keep[idx[:k_trim]] = False
    keep = keep & ok_neg
    if keep.sum() == 0:
        mid = np.argsort(dist)[0]
        mask = np.zeros(n, dtype=bool); mask[mid] = True
        return mask
    return keep

def weighted_geometric_mean(samples, weights):
    samples = np.maximum(samples, 0.0)
    W = np.clip(weights, 0.0, None)
    if W.sum() <= 0: W = np.ones_like(W)
    W = W / W.sum()
    L = np.log1p(samples)
    m = np.sum(W[:, None] * L, axis=0)
    return np.expm1(m)

def _fallback_from_hist_mean(y_hist, h, alpha_nb=5.0):
    if len(y_hist) == 0 or not np.all(np.isfinite(y_hist)):
        mu_hist = 0.0
    else:
        mu_hist = float(np.mean(np.clip(y_hist, 0.0, None)))
    mu_hist = max(mu_hist, 0.0)
    y_hat_fb = np.full(h, mu_hist, dtype=float)
    u_fb = mu_hist + (mu_hist * mu_hist) / max(alpha_nb, 1e-6)
    u_fb = np.full(h, u_fb, dtype=float)
    return y_hat_fb, u_fb

def forecast_point_and_uncertainty_robust(
    y_hist,                 # (T_h-1,)
    *,
    y_future,               # (T_p,)
    sir_hist,               # (T_h,3)
    alpha_nb = 5.0,
    bounds_logR0_logTinf=((np.log(0.8), np.log(5.0)), (np.log(2.0), np.log(10.0))),
    prior_mu=None, prior_sd=None,
    ut_alpha=0.2, ut_beta=2.0, ut_kappa=None,
    trim_frac=0.15, max_neg_frac=0.0,
    clip_nonneg=True,
    calib_s_min=0.3, calib_s_max=3.0,
    sir_future_sequence=None,
    max_inc_val=1e6,
    max_ratio_vs_hist=100.0,
    compute_sigma_every=4,
    hess_eps=5e-4,
    lbfgs_maxiter=150,
    lbfgs_ftol=1e-6,
    hist_smooth_k=1
):
    def smooth_hist(y, k):
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)
        for i in range(len(y)):
            L = max(0, i-k+1)
            out[i] = y[L:i+1].mean()
        return out

    y_hist   = np.asarray(y_hist,   dtype=float)
    y_future = np.asarray(y_future, dtype=float)
    sir_hist = np.asarray(sir_hist, dtype=float)
    assert sir_hist.ndim == 2 and sir_hist.shape[1] == 3
    assert np.all(y_hist >= 0) and np.all(y_future >= 0)

    S0_tr, I0_tr, R0_tr = sir_hist[0]
    N_tr = max(S0_tr + I0_tr + R0_tr, 1e-8)
    x0_hist = _sanitize_state(S0_tr/N_tr, I0_tr/N_tr, R0_tr/N_tr)

    S_last, I_last, R_last = sir_hist[-1]
    N_last = max(S_last + I_last + R_last, 1e-8)
    x0_fore_base = _sanitize_state(S_last/N_last, I_last/N_last, R_last/N_last)

    T_p = len(y_future)
    y_hat_all = np.zeros(T_p, dtype=float)
    u_all     = np.zeros(T_p, dtype=float)

    y_hist_roll = y_hist.copy()

    hist_mean_base = float(np.mean(y_hist)) if len(y_hist) > 0 else 0.0
    hist_mean_base = max(hist_mean_base, 1e-8)

    Sigma_eta_cached = None

    for step in range(T_p):
        if hist_smooth_k > 1:
            y_hist_for_est = smooth_hist(y_hist_roll, hist_smooth_k)
        else:
            y_hist_for_est = y_hist_roll

        eta_hat, pr_mu, pr_sd = map_estimate_eta_fixed_init(
            y_hist_for_est, x0_hist, N_tr, alpha_nb=alpha_nb,
            prior_mu=prior_mu, prior_sd=prior_sd,
            bounds_logR0_logTinf=bounds_logR0_logTinf,
            maxiter=lbfgs_maxiter, ftol=lbfgs_ftol
        )

        need_sigma = (compute_sigma_every in (0, 1)) or (step % compute_sigma_every == 0) or (Sigma_eta_cached is None)
        if need_sigma:
            Sigma_eta = laplace_cov_eta_fixed_init(
                y_hist_for_est, x0_hist, N_tr, eta_hat,
                alpha_nb=alpha_nb, prior_mu=pr_mu, prior_sd=pr_sd, hess_eps=hess_eps
            )
            Sigma_eta_cached = Sigma_eta
        else:
            Sigma_eta = Sigma_eta_cached

        pts, wm, _ = ut_sigma_points(eta_hat, Sigma_eta, alpha=ut_alpha, beta=ut_beta, kappa=ut_kappa)

        theta_mean = eta_to_theta(eta_hat)
        mu_tr_mean = forward_incidence_given_init(len(y_hist_for_est), x0_hist, N_tr, theta_mean)
        mu_last_bar = float(mu_tr_mean[-1] if len(mu_tr_mean) > 0 else 0.0)
        y_last_obs  = float(y_hist_for_est[-1]) if len(y_hist_for_est) > 0 else mu_last_bar
        s = y_last_obs / max(1e-8, mu_last_bar) if mu_last_bar > 0 else 1.0
        s = float(np.clip(s, calib_s_min, calib_s_max))

        if sir_future_sequence is not None:
            S_f, I_f, R_f = sir_future_sequence[step]
            N_f = max(S_f + I_f + R_f, 1e-8)
            x0_fore = _sanitize_state(S_f/N_f, I_f/N_f, R_f/N_f)
            pop_fore = N_f
        else:
            x0_fore  = x0_fore_base
            pop_fore = N_last

        samples = []
        for eta in pts:
            theta = eta_to_theta(eta)
            inc1  = forward_incidence_future(x0_fore, pop_fore, 1, theta)
            yhat1 = (inc1[0] if len(inc1)>0 else 0.0) * s
            samples.append(yhat1)
        samples = np.asarray(samples, dtype=float).reshape(-1, 1)

        bad = False
        if not np.all(np.isfinite(samples)):
            bad = True
        if (samples < 0).any():
            bad = True
        if np.nanmax(samples) > max_inc_val:
            bad = True
        if np.nanmax(samples) > max_ratio_vs_hist * hist_mean_base:
            bad = True

        if bad:
            y_hat_step, u_step = _fallback_from_hist_mean(y_hist_roll, 1, alpha_nb=alpha_nb)
            y_hat_all[step] = y_hat_step[0]
            u_all[step]     = u_step[0]
            y_hist_roll = np.concatenate([y_hist_roll, [y_future[step]]])
            continue

        keep_mask = robust_filter_trajectories(samples, trim_frac=trim_frac, max_neg_frac=max_neg_frac)
        kept   = samples[keep_mask][:, 0]
        w_kept = np.clip(wm[keep_mask], 0.0, None)
        if w_kept.sum() <= 0 or kept.size == 0:
            y_hat_step, u_step = _fallback_from_hist_mean(y_hist_roll, 1, alpha_nb=alpha_nb)
            y_hat_all[step] = y_hat_step[0]
            u_all[step]     = u_step[0]
            y_hist_roll = np.concatenate([y_hist_roll, [y_future[step]]])
            continue

        y_hat_step = weighted_geometric_mean(kept.reshape(-1,1), w_kept).item()
        if clip_nonneg: y_hat_step = max(0.0, y_hat_step)

        if (y_hat_step > max_inc_val) or (y_hat_step > max_ratio_vs_hist * hist_mean_base):
            y_hat_step, _ = _fallback_from_hist_mean(y_hist_roll, 1, alpha_nb=alpha_nb)
            y_hat_step = y_hat_step[0]

        diff  = y_future[step] - kept
        w_norm = w_kept / w_kept.sum()
        u_step = float(np.sum(w_norm * (diff**2)))
        u_step = max(u_step, 0.0)

        y_hat_all[step] = y_hat_step
        u_all[step]     = u_step

        y_hist_roll = np.concatenate([y_hist_roll, [y_future[step]]])

    return y_hat_all, u_all
