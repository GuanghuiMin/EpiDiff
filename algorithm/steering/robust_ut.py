def forecast_point_and_uncertainty_robust(
    y_hist,                 # historical incidence, shape (T_h,)
    *,
    horizon,                # forecast horizon T_p
    sir_hist,               # historical SIR states, shape (T_h, 3)
    alpha_nb=5.0,
    bounds_logR0_logTinf=((np.log(0.8), np.log(5.0)), (np.log(2.0), np.log(10.0))),
    prior_mu=None,
    prior_sd=None,
    ut_alpha=0.2,
    ut_beta=2.0,
    ut_kappa=None,
    trim_frac=0.15,
    max_neg_frac=0.0,
    clip_nonneg=True,
    calib_s_min=0.3,
    calib_s_max=3.0,
    sir_future_sequence=None,
    max_inc_val=1e6,
    max_ratio_vs_hist=100.0,
    hess_eps=5e-4,
    lbfgs_maxiter=150,
    lbfgs_ftol=1e-6,
    hist_smooth_k=1,
):
    def smooth_hist(y, k):
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)
        for i in range(len(y)):
            L = max(0, i - k + 1)
            out[i] = y[L:i+1].mean()
        return out

    y_hist = np.asarray(y_hist, dtype=float)
    sir_hist = np.asarray(sir_hist, dtype=float)
    horizon = int(horizon)

    assert horizon > 0
    assert sir_hist.ndim == 2 and sir_hist.shape[1] == 3
    assert np.all(y_hist >= 0)

    if hist_smooth_k > 1:
        y_hist_for_est = smooth_hist(y_hist, hist_smooth_k)
    else:
        y_hist_for_est = y_hist

    S0_tr, I0_tr, R0_tr = sir_hist[0]
    N_tr = max(S0_tr + I0_tr + R0_tr, 1e-8)
    x0_hist = _sanitize_state(S0_tr / N_tr, I0_tr / N_tr, R0_tr / N_tr)

    S_last, I_last, R_last = sir_hist[-1]
    N_last = max(S_last + I_last + R_last, 1e-8)
    x0_fore_base = _sanitize_state(S_last / N_last, I_last / N_last, R_last / N_last)

    hist_mean_base = float(np.mean(y_hist_for_est)) if len(y_hist_for_est) > 0 else 0.0
    hist_mean_base = max(hist_mean_base, 1e-8)

    try:
        eta_hat, pr_mu, pr_sd = map_estimate_eta_fixed_init(
            y_hist_for_est,
            x0_hist,
            N_tr,
            alpha_nb=alpha_nb,
            prior_mu=prior_mu,
            prior_sd=prior_sd,
            bounds_logR0_logTinf=bounds_logR0_logTinf,
            maxiter=lbfgs_maxiter,
            ftol=lbfgs_ftol,
        )

        Sigma_eta = laplace_cov_eta_fixed_init(
            y_hist_for_est,
            x0_hist,
            N_tr,
            eta_hat,
            alpha_nb=alpha_nb,
            prior_mu=pr_mu,
            prior_sd=pr_sd,
            hess_eps=hess_eps,
        )

        pts, wm, _ = ut_sigma_points(
            eta_hat,
            Sigma_eta,
            alpha=ut_alpha,
            beta=ut_beta,
            kappa=ut_kappa,
        )

        theta_mean = eta_to_theta(eta_hat)
        mu_tr_mean = forward_incidence_given_init(
            len(y_hist_for_est),
            x0_hist,
            N_tr,
            theta_mean,
        )

        mu_last_bar = float(mu_tr_mean[-1] if len(mu_tr_mean) > 0 else 0.0)
        y_last_obs = float(y_hist_for_est[-1]) if len(y_hist_for_est) > 0 else mu_last_bar

        s = y_last_obs / max(1e-8, mu_last_bar) if mu_last_bar > 0 else 1.0
        s = float(np.clip(s, calib_s_min, calib_s_max))

        samples = []
        for eta in pts:
            theta = eta_to_theta(eta)

            if sir_future_sequence is None:
                traj = forward_incidence_future(
                    x0_fore_base,
                    N_last,
                    horizon,
                    theta,
                )
            else:
                traj_steps = []
                for step in range(horizon):
                    S_f, I_f, R_f = sir_future_sequence[step]
                    N_f = max(S_f + I_f + R_f, 1e-8)
                    x0_fore = _sanitize_state(S_f / N_f, I_f / N_f, R_f / N_f)
                    inc1 = forward_incidence_future(x0_fore, N_f, 1, theta)
                    traj_steps.append(inc1[0] if len(inc1) > 0 else 0.0)
                traj = np.asarray(traj_steps, dtype=float)

            traj = np.asarray(traj, dtype=float) * s
            samples.append(traj)

        samples = np.asarray(samples, dtype=float)  # (2d+1, horizon)

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
            return _fallback_from_hist_mean(y_hist_for_est, horizon, alpha_nb=alpha_nb)

        keep_mask = robust_filter_trajectories(
            samples,
            trim_frac=trim_frac,
            max_neg_frac=max_neg_frac,
        )

        kept = samples[keep_mask]                       # (K, horizon)
        w_kept = np.clip(wm[keep_mask], 0.0, None)

        if w_kept.sum() <= 0 or kept.size == 0:
            return _fallback_from_hist_mean(y_hist_for_est, horizon, alpha_nb=alpha_nb)

        y_hat = weighted_geometric_mean(kept, w_kept)

        if clip_nonneg:
            y_hat = np.maximum(y_hat, 0.0)

        too_large = (y_hat > max_inc_val) | (y_hat > max_ratio_vs_hist * hist_mean_base)
        if np.any(too_large):
            y_fb, _ = _fallback_from_hist_mean(y_hist_for_est, horizon, alpha_nb=alpha_nb)
            y_hat = np.where(too_large, y_fb, y_hat)

        w_norm = w_kept / w_kept.sum()
        mean_kept = np.sum(w_norm[:, None] * kept, axis=0)
        u = np.sum(w_norm[:, None] * ((kept - mean_kept[None, :]) ** 2), axis=0)
        u = np.maximum(u, 0.0)

        return y_hat, u

    except Exception:
        return _fallback_from_hist_mean(y_hist_for_est, horizon, alpha_nb=alpha_nb)
