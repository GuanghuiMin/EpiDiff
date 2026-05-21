def forecast_point_and_uncertainty_robust(
    y_hist,                 # historical incidence, shape (T_h,)
    *,
    horizon,                # forecast horizon T_p
    sir_hist,               # historical SIR/SI states, shape (T_h, 3)
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
    compute_sigma_every=4,
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
    T_p = int(horizon)

    assert T_p > 0
    assert sir_hist.ndim == 2 and sir_hist.shape[1] == 3
    assert np.all(y_hist >= 0)

    S0_tr, I0_tr, R0_tr = sir_hist[0]
    N_tr = max(S0_tr + I0_tr + R0_tr, 1e-8)
    x0_hist = _sanitize_state(S0_tr / N_tr, I0_tr / N_tr, R0_tr / N_tr)

    S_last, I_last, R_last = sir_hist[-1]
    N_last = max(S_last + I_last + R_last, 1e-8)
    x0_fore_base = _sanitize_state(S_last / N_last, I_last / N_last, R_last / N_last)

    y_hat_all = np.zeros(T_p, dtype=float)
    u_all = np.zeros(T_p, dtype=float)

    y_hist_roll = y_hist.copy()

    hist_mean_base = float(np.mean(y_hist)) if len(y_hist) > 0 else 0.0
    hist_mean_base = max(hist_mean_base, 1e-8)

    Sigma_eta_cached = None

    for step in range(T_p):
        if hist_smooth_k > 1:
            y_hist_for_est = smooth_hist(y_hist_roll, hist_smooth_k)
        else:
            y_hist_for_est = y_hist_roll

        data_consistency = 1.0
        if len(y_hist_for_est) > 3:
            recent_std = np.std(y_hist_for_est[-5:])
            overall_mean = np.mean(y_hist_for_est)
            if overall_mean > 0:
                consistency_score = 1.0 - min(1.0, recent_std / overall_mean)
                data_consistency = 0.5 + 1.5 * consistency_score

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
            adaptive_prior_weight=data_consistency,
        )

        need_sigma = (
            (compute_sigma_every in (0, 1))
            or (step % compute_sigma_every == 0)
            or (Sigma_eta_cached is None)
        )
        if need_sigma:
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
            Sigma_eta_cached = Sigma_eta
        else:
            Sigma_eta = Sigma_eta_cached

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

        if sir_future_sequence is not None:
            S_f, I_f, R_f = sir_future_sequence[step]
            N_f = max(S_f + I_f + R_f, 1e-8)
            x0_fore = _sanitize_state(S_f / N_f, I_f / N_f, R_f / N_f)
            pop_fore = N_f
        else:
            x0_fore = x0_fore_base
            pop_fore = N_last

        samples = []
        for eta in pts:
            theta = eta_to_theta(eta)
            inc1 = forward_incidence_future(x0_fore, pop_fore, 1, theta)
            yhat1 = (inc1[0] if len(inc1) > 0 else 0.0) * s
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
            y_hat_step, u_step = _fallback_from_hist_mean(
                y_hist_roll,
                1,
                alpha_nb=alpha_nb,
            )
            y_hat_all[step] = y_hat_step[0]
            u_all[step] = u_step[0]
            y_hist_roll = np.concatenate([y_hist_roll, [y_hat_step[0]]])
            continue

        keep_mask = robust_filter_trajectories(
            samples,
            trim_frac=trim_frac,
            max_neg_frac=max_neg_frac,
        )
        kept = samples[keep_mask][:, 0]
        w_kept = np.clip(wm[keep_mask], 0.0, None)

        if w_kept.sum() <= 0 or kept.size == 0:
            y_hat_step, u_step = _fallback_from_hist_mean(
                y_hist_roll,
                1,
                alpha_nb=alpha_nb,
            )
            y_hat_all[step] = y_hat_step[0]
            u_all[step] = u_step[0]
            y_hist_roll = np.concatenate([y_hist_roll, [y_hat_step[0]]])
            continue

        y_hat_step = weighted_geometric_mean(kept.reshape(-1, 1), w_kept).item()
        if clip_nonneg:
            y_hat_step = max(0.0, y_hat_step)

        if (y_hat_step > max_inc_val) or (y_hat_step > max_ratio_vs_hist * hist_mean_base):
            y_hat_step, _ = _fallback_from_hist_mean(
                y_hist_roll,
                1,
                alpha_nb=alpha_nb,
            )
            y_hat_step = y_hat_step[0]

        w_norm = w_kept / w_kept.sum()
        mean_kept = np.sum(w_norm * kept)
        u_step = float(np.sum(w_norm * ((kept - mean_kept) ** 2)))
        u_step = max(u_step, 0.0)

        y_hat_all[step] = y_hat_step
        u_all[step] = u_step

        y_hist_roll = np.concatenate([y_hist_roll, [y_hat_step]])

    return y_hat_all, u_all
