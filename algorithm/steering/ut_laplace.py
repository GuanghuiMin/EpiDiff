import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import nbinom
import matplotlib.pyplot as plt  # kept as-is per your constraint (no plotting used)

# ---------------- SIR ODE ----------------
def sir_ode(state, t, beta, gamma):
    """Standard SIR ODE with mass-action incidence (states are normalized by population)."""
    S, I, R = state
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR], dtype=float)

def _sanitize_state(S0, I0, R0):
    """Ensure valid, positive, normalized initial state in [0, 1] with S+I+R=1."""
    S0 = max(float(S0), 1e-10)
    I0 = max(float(I0), 1e-12)
    R0 = max(float(R0), 0.0)
    total = S0 + I0 + R0
    if total <= 0:
        return np.array([0.99, 0.01, 0.0], dtype=float)
    return np.array([S0/total, I0/total, R0/total], dtype=float)

# ---------------- Forward models ----------------
def forward_incidence_train(y_hist, pop, theta):
    """
    Forward incidence on the observed window for likelihood computation.
    Heuristic init from y_hist; used only when sir_hist is NOT provided.
    """
    beta, gamma = map(float, theta)
    k = len(y_hist)
    if k == 0:
        return np.array([])

    gamma_eff = max(gamma, 1e-6)
    I0 = np.clip(float(y_hist[0]) / (pop * gamma_eff), 1e-8, 0.2)
    S0 = max(1.0 - I0, 1e-6)
    R0 = 0.0
    x0 = _sanitize_state(S0, I0, R0)

    t = np.arange(k, dtype=float)
    sol = odeint(sir_ode, x0, t, args=(beta, gamma), rtol=1e-6, atol=1e-8, mxstep=5000)
    S_t = np.clip(sol[:, 0], 1e-10, 1.0)
    I_t = np.clip(sol[:, 1], 1e-12, 1.0)
    mu = beta * S_t * I_t * pop
    return np.maximum(mu, 1e-8)

def forward_incidence_future(init_state, pop, h, theta):
    """Forward incidence on the future window of length h."""
    beta, gamma = map(float, theta)
    if h <= 0:
        return np.array([])
    x0 = _sanitize_state(*init_state)
    t = np.arange(h, dtype=float)
    sol = odeint(sir_ode, x0, t, args=(beta, gamma), rtol=1e-6, atol=1e-8, mxstep=5000)
    S_t = np.clip(sol[:, 0], 1e-10, 1.0)
    I_t = np.clip(sol[:, 1], 1e-12, 1.0)
    mu = beta * S_t * I_t * pop
    return np.maximum(mu, 1e-8)

def forward_incidence_given_init(k, x0_norm, N, theta):
    """
    Forward incidence on the observed window with FIXED initial state.
    Used for MAP/Laplace when sir_hist is provided (ensures train/predict consistency).
    """
    beta, gamma = map(float, theta)
    if k <= 0:
        return np.array([])
    t = np.arange(k, dtype=float)
    sol = odeint(sir_ode, x0_norm, t, args=(beta, gamma), rtol=1e-6, atol=1e-8, mxstep=5000)
    S_t = np.clip(sol[:, 0], 1e-12, 1.0)
    I_t = np.clip(sol[:, 1], 1e-12, 1.0)
    mu = beta * S_t * I_t * float(N)
    return np.maximum(mu, 1e-12)

# ---------------- Likelihood & Prior ----------------
def nb_nll(y, mu, alpha):
    """Negative binomial negative log-likelihood with size=alpha, prob=alpha/(alpha+mu)."""
    if alpha <= 0:
        return 1e10
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        y = np.round(y).astype(int)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-8, 1e12)
    p = np.clip(alpha / (alpha + mu), 1e-8, 1 - 1e-8)
    try:
        nll = -np.sum(nbinom.logpmf(y, alpha, p))
        return float(nll) if np.isfinite(nll) else 1e10
    except Exception:
        return 1e10

def neg_log_prior_eta(eta, mu=None, sd=None):
    """Gaussian prior in log-parameter space: eta ~ N(mu, diag(sd^2))."""
    mu = np.zeros(2) if mu is None else mu
    sd = np.ones(2) if sd is None else sd
    z = (eta - mu) / sd
    return 0.5 * np.sum(z**2) + np.sum(np.log(sd)) + len(eta)*0.5*np.log(2*np.pi)

# ---------------- MAP estimation (generic; used when sir_hist is None) ----------------
def map_estimate(y_hist, pop, alpha_nb=5.0,
                 prior_mu=np.log(np.array([0.2, 0.1])),
                 prior_sd=np.array([1.0, 1.0]),
                 bounds_logbeta_loggamma=((-10.0, 1.0), (-10.0, 1.0))):
    """MAP in log-parameter space with heuristic init; used only if sir_hist is NOT provided."""
    def objective(eta):
        theta = np.exp(eta)
        mu_train = forward_incidence_train(y_hist, pop, theta)
        if len(mu_train) != len(y_hist):
            return 1e10
        return nb_nll(y_hist, mu_train, alpha_nb) + neg_log_prior_eta(eta, prior_mu, prior_sd)

    inits = [np.log([0.05, 0.1]), np.log([0.1, 0.1]), np.log([0.2, 0.1]), np.log([0.1, 0.05])]
    best, best_val = None, np.inf
    for x0 in inits:
        res = minimize(objective, x0, method='L-BFGS-B',
                       bounds=bounds_logbeta_loggamma,
                       options={'maxiter': 800, 'ftol': 1e-9})
        if res.success and np.isfinite(res.fun) and res.fun < best_val:
            best, best_val = res, float(res.fun)

    if best is None:
        eta_hat = np.log([0.2, 0.1])  # fallback
        theta_hat = np.exp(eta_hat)
        info = {'success': False, 'obj_value': np.inf}
    else:
        lb0, ub0 = bounds_logbeta_loggamma[0]
        lb1, ub1 = bounds_logbeta_loggamma[1]
        eta_hat = np.array([np.clip(best.x[0], lb0, ub0), np.clip(best.x[1], lb1, ub1)], dtype=float)
        theta_hat = np.exp(eta_hat)
        info = {'success': True, 'obj_value': best_val}
    return eta_hat, theta_hat, info

# ---------------- Numerical Hessian & Laplace ----------------
def numerical_hessian(f, x, eps=1e-4):
    """Symmetric finite-difference Hessian for small-dimensional problems (d=2 here)."""
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
            hi = eps * (1 + abs(x[i]))
            hj = eps * (1 + abs(x[j]))
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i]+=hi; xpp[j]+=hj
            xpm[i]+=hi; xpm[j]-=hj
            xmp[i]-=hi; xmp[j]+=hj
            xmm[i]-=hi; xmm[j]-=hj
            Hij = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*hi*hj)
            H[i, j] = H[j, i] = Hij
    return H

def laplace_covariance(y_hist, pop, eta_hat, alpha_nb=5.0,
                       prior_mu=np.log(np.array([0.2, 0.1])),
                       prior_sd=np.array([1.0, 1.0])):
    """Laplace covariance Sigma ≈ H^{-1} at MAP in log-parameter space (heuristic init version)."""
    def objective(eta):
        theta = np.exp(eta)
        mu_train = forward_incidence_train(y_hist, pop, theta)
        return nb_nll(y_hist, mu_train, alpha_nb) + neg_log_prior_eta(eta, prior_mu, prior_sd)

    H = numerical_hessian(objective, eta_hat, eps=1e-4)
    w, V = np.linalg.eigh(H)
    w = np.clip(w, 1e-8, None)
    H_pd = V @ np.diag(w) @ V.T
    try:
        Sigma = np.linalg.inv(H_pd)
    except np.linalg.LinAlgError:
        diag = np.clip(np.diag(H_pd), 1e-8, None)
        Sigma = np.diag(1.0 / diag)

    w2, V2 = np.linalg.eigh(Sigma)
    w2 = np.clip(w2, 1e-10, 1.0)
    Sigma = V2 @ np.diag(w2) @ V2.T
    return H_pd, Sigma

def laplace_covariance_fixed_init(y_hist, x0_norm, N, eta_hat,
                                  alpha_nb=5.0,
                                  prior_mu=np.log(np.array([0.2, 0.1])),
                                  prior_sd=np.array([1.0, 1.0])):
    """Laplace covariance with FIXED initial state; used when sir_hist is provided."""
    def objective(eta):
        theta = np.exp(eta)
        mu_train = forward_incidence_given_init(len(y_hist), x0_norm, N, theta)
        return nb_nll(y_hist, mu_train, alpha_nb) + neg_log_prior_eta(eta, prior_mu, prior_sd)

    H = numerical_hessian(objective, eta_hat, eps=1e-4)
    w, V = np.linalg.eigh(H)
    w = np.clip(w, 1e-8, None)
    H_pd = V @ np.diag(w) @ V.T
    try:
        Sigma = np.linalg.inv(H_pd)
    except np.linalg.LinAlgError:
        diag = np.clip(np.diag(H_pd), 1e-8, None)
        Sigma = np.diag(1.0 / diag)

    w2, V2 = np.linalg.eigh(Sigma)
    w2 = np.clip(w2, 1e-10, 1.0)
    Sigma = V2 @ np.diag(w2) @ V2.T
    return H_pd, Sigma

# ---------------- Unscented Transform (UT) ----------------
def ut_sigma_points(mu, Sigma, alpha=1e-1, beta=2.0, kappa=None):
    """
    Compute 2L+1 sigma points and UT weights for mean and covariance.
    NOTE: small alpha => possibly negative wm[0]; we handle nonnegativity only when forming 'u'.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    L = len(mu)
    if kappa is None:
        kappa = 3.0 - L
    lam = alpha**2 * (L + kappa) - L

    e, U = np.linalg.eigh(Sigma)
    e = np.maximum(e, 1e-10)
    Sigma_pd = U @ np.diag(e) @ U.T

    scale = L + lam
    if scale <= 0:
        scale = 1e-6
    try:
        S = np.linalg.cholesky(scale * Sigma_pd)
    except np.linalg.LinAlgError:
        S = U @ np.diag(np.sqrt(e * scale))

    pts = np.zeros((2*L + 1, L))
    pts[0] = mu
    for i in range(L):
        pts[i+1]   = mu + S[:, i]
        pts[i+1+L] = mu - S[:, i]

    wm = np.zeros(2*L + 1)
    wc = np.zeros(2*L + 1)
    wm[0] = lam / (L + lam)
    wc[0] = wm[0] + (1.0 - alpha**2 + beta)
    wm[1:] = wc[1:] = 1.0 / (2.0 * (L + lam))
    return pts, wm, wc

def ut_forecast_samples(init_state, pop, h, eta_mu, Sigma_eta, alpha=1e-1, beta=2.0, kappa=None):
    """Generate sigma-point forecasts over the future horizon."""
    pts, wm, wc = ut_sigma_points(eta_mu, Sigma_eta, alpha=alpha, beta=beta, kappa=kappa)
    samples = []
    for eta in pts:
        theta = np.exp(eta)                 # map log-params to linear domain
        inc = forward_incidence_future(init_state, pop, h, theta)
        samples.append(inc)
    samples = np.asarray(samples)           # [2L+1, h]
    return samples, wm, wc

# ---------------- Final public API ----------------
def forecast_point_and_uncertainty(
    y_hist,
    *,
    y_future,               # REQUIRED: ground-truth over the forecast window (length >= h)
    pop=1.0,                # population size (scalar > 0)
    alpha_nb=5.0,
    prior_mu=np.log(np.array([0.2, 0.1])),
    prior_sd=np.array([1.0, 1.0]),
    ut_alpha=1e-1,
    ut_beta=2.0,
    ut_kappa=None,
    sir_hist=None          # optional SIR counts over the history window, shape (T_h, 3)
):
    """
    UT-based forecasting with data-aware uncertainty.
    Implements:
        y_hat_t = sum_j w_j * Phi(theta^(j))_t
        u_t     = sum_j w_j * (y_t - Phi(theta^(j))_t)^2

    If `sir_hist` is provided (counts), we:
        - Fit MAP/Laplace on the HISTORY window starting from sir_hist[0] (first frame),
        - Forecast from sir_hist[-1] (last frame),
        - Use the same constant scale 'pop' (= N_last) for forecasting.
        (This removes the train/predict init mismatch.)
    """
    # ---- Validate inputs ----
    y_hist = np.asarray(y_hist, dtype=float)
    y_future = np.asarray(y_future, dtype=float)
    assert np.all(y_hist >= 0), "y_hist must be nonnegative"
    h = len(y_future)
    assert h > 0, "y_future must have positive length"

    # ---- Branch A: sir_hist is provided -> FIXED-init fitting on the HISTORY START ----
    if sir_hist is not None:
        sir_hist = np.asarray(sir_hist, dtype=float)
        assert sir_hist.ndim == 2 and sir_hist.shape[1] == 3, "sir_hist must be (T_h, 3) with columns S,I,R"

        # A1) for TRAINING over y_hist: start at the FIRST frame of history
        S0_tr, I0_tr, R0_tr = sir_hist[0, 0], sir_hist[0, 1], sir_hist[0, 2]
        N_tr = max(S0_tr + I0_tr + R0_tr, 1e-8)
        x0_hist = _sanitize_state(S0_tr / N_tr, I0_tr / N_tr, R0_tr / N_tr)

        # A2) for FORECASTING: start at the LAST frame of history
        S_last, I_last, R_last = sir_hist[-1, 0], sir_hist[-1, 1], sir_hist[-1, 2]
        N_last = max(S_last + I_last + R_last, 1e-8)
        x0_fore = _sanitize_state(S_last / N_last, I_last / N_last, R_last / N_last)

        # 1) MAP with fixed init (train-start)
        def obj_eta(eta):
            theta = np.exp(eta)
            mu_tr = forward_incidence_given_init(len(y_hist), x0_hist, N_tr, theta)
            return nb_nll(y_hist, mu_tr, alpha_nb) + neg_log_prior_eta(eta, prior_mu, prior_sd)

        inits = [np.log([0.05, 0.1]), np.log([0.1, 0.1]), np.log([0.2, 0.1]), np.log([0.1, 0.05])]
        best, best_val = None, np.inf
        for x0_try in inits:
            res = minimize(obj_eta, x0_try, method='L-BFGS-B',
                           bounds=((-10.0, 1.0), (-10.0, 1.0)),
                           options={'maxiter': 800, 'ftol': 1e-9})
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best, best_val = res, float(res.fun)
        eta_hat = np.array(best.x if best is not None else np.log([0.2, 0.1]), dtype=float)

        # 2) Laplace with fixed init (train-start)
        _, Sigma_eta = laplace_covariance_fixed_init(
            y_hist, x0_hist, N_tr, eta_hat,
            alpha_nb=alpha_nb, prior_mu=prior_mu, prior_sd=prior_sd
        )

        # init for forecasting
        init_state = x0_fore
        pop = N_last

    # ---- Branch B: no sir_hist -> old heuristic path ----
    else:
        eta_hat, theta_hat, _ = map_estimate(
            y_hist, pop, alpha_nb=alpha_nb, prior_mu=prior_mu, prior_sd=prior_sd
        )
        _, Sigma_eta = laplace_covariance(
            y_hist, pop, eta_hat, alpha_nb=alpha_nb, prior_mu=prior_mu, prior_sd=prior_sd
        )
        # heuristic init for forecasting
        tail = max(1, min(3, len(y_hist)))
        y_tail = float(np.mean(y_hist[-tail:]))
        gamma_eff = max(np.exp(eta_hat)[1], 1e-6)
        I0 = np.clip(y_tail / (pop * gamma_eff), 1e-8, 0.2)
        S0 = max(1.0 - I0, 1e-6)
        R0 = 0.0
        init_state = _sanitize_state(S0, I0, R0)

    # ---- Unscented Transform -> sigma points in log-parameter space ----
    pts, wm, _ = ut_sigma_points(eta_hat, Sigma_eta, alpha=ut_alpha, beta=ut_beta, kappa=ut_kappa)
    sig_forecasts = []
    for eta in pts:
        theta = np.exp(eta)  # [beta, gamma]
        inc = forward_incidence_future(init_state, pop, h, theta)  # (h,)
        sig_forecasts.append(inc)
    sig_forecasts = np.asarray(sig_forecasts, dtype=float)  # (n_sigma, h)

    # ---- UT mean and data-aware uncertainty ----
    y_hat = np.sum(wm[:, None] * sig_forecasts, axis=0)     # (h,)
    diff = y_future[None, :] - sig_forecasts                 # (n_sigma, h)

    # Nonnegative renormalized weights ONLY for u (to avoid negative values when alpha is small)
    wm_pos = np.clip(wm, 0.0, None)
    wm_pos = wm_pos / wm_pos.sum()
    u = np.sum(wm_pos[:, None] * (diff**2), axis=0)

    # If you want the strict original definition, use this (may give small negative values):
    # u = np.sum(wm[:, None] * (diff**2), axis=0); u = np.maximum(u, 0.0)

    return y_hat, u

# ---------------- Minimal Example ----------------
if __name__ == "__main__":
    # Load dict with SIR: shape (T, V, 3), channels [S, I, R]
    data = np.load("/home/guanghui/DiffODE/data/dataset/COVID-JP/jp20200401_20210921.npy",
                   allow_pickle=True).item()
    sir = data['SIR']   # (T, V, 3)
    print("SIR data shape:", sir.shape)

    T, V, _ = sir.shape
    v = 30  # choose a node to test

    S = sir[:, :, 0]    # (T, V)
    I = sir[:, :, 1]
    R = sir[:, :, 2]

    # Define windows (example indices; choose from your TEST split)
    T_h = 14
    T_p = 7
    label_start = 450
    hist_start = label_start - T_h
    hist_end   = label_start              # exclusive
    fut_start  = label_start
    fut_end    = label_start + T_p        # exclusive

    # Slice history SIR for node v: shape (T_h, 3)
    sir_hist = np.stack([S[hist_start:hist_end, v],
                         I[hist_start:hist_end, v],
                         R[hist_start:hist_end, v]], axis=-1)

    # Build incidences from S differences (counts domain)
    # y_hist length = T_h-1; y_future length = T_p
    S_hist = S[hist_start:hist_end, v]          # (T_h,)
    S_fut  = S[fut_start-1:fut_end, v]          # include S at t-1 to get T_p diffs
    y_hist = S_hist[:-1] - S_hist[1:]           # (T_h-1,)
    y_future = S_fut[:-1] - S_fut[1:]           # (T_p,)

    # Run UT-based forecast with data-aware uncertainty (sir_hist enforces fixed-init fitting)
    y_hat, u = forecast_point_and_uncertainty(
        y_hist=y_hist,
        y_future=y_future,
        sir_hist=sir_hist,       # train from sir_hist[0], forecast from sir_hist[-1]
        alpha_nb=5.0,
        ut_alpha=1e-1,
        ut_beta=2.0
    )

    print("UT mean forecast (node v=%d):" % v, y_hat)
    print("Data-aware predictive uncertainty (node v=%d):" % v, u)

    # Pretty compare
    h = len(y_future)
    print("\nStep |  True(y)  |  y_hat  |  Uncertainty ")
    print("-----+-----------+-------------------+------------------------------")
    for t in range(h):
        print(f"{t+1:>4d} | {y_future[t]:>9.3f} | {y_hat[t]:>17.3f} | {u[t]:>28.3f}")

    # Summary metrics
    diff = y_hat - y_future
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    print(f"\nMAE over horizon:  {mae:.4f}")
    print(f"RMSE over horizon: {rmse:.4f}")