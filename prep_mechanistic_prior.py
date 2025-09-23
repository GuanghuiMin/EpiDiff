import os
import numpy as np
from tqdm import tqdm

from algorithm.steering.robust_ut_si import forecast_point_and_uncertainty_robust as forecast_point_and_uncertainty_si
from algorithm.steering.robust_ut import forecast_point_and_uncertainty_robust as forecast_point_and_uncertainty_sir

def run_test_windows_only(
    SIR_PATH: str,
    OUT_PATH: str,
    T_h: int,
    T_p: int,
    test_start_id: int,
    ut_alpha: float = 0.2,
    ut_beta: float = 2.0,
    trim_frac: float = 0.15,
    max_neg_frac: float = 0.0,
    clip_nonneg: bool = True,
    calib_s_min: float = 0.3,
    calib_s_max: float = 3.0,
    max_inc_val: float = 1e6,
    max_ratio_vs_hist: float = 100.0,
    compute_sigma_every: int = 4,
    hess_eps: float = 5e-4,
    lbfgs_maxiter: int = 150,
    lbfgs_ftol: float = 1e-6,
    hist_smooth_k=1,
    alpha_nb: float = 5.0,
    forecast_point_and_uncertainty=None,
    direct_incidence: bool = False,
    I_PATH: str = None,
):
    # ---- Load SIR ----
    data = np.load(SIR_PATH, allow_pickle=True).item()
    sir = data["SIR"]  # (T, V, 3) or (T, V, 2)
    T, V, C = sir.shape
    print(f"SIR data shape: {sir.shape}")

    if I_PATH is not None:
        print(f"Loading I data from: {I_PATH}")
        
        if I_PATH.endswith('.txt'):
            I_data = np.loadtxt(I_PATH, delimiter=',')  # (T, V)
            I = I_data
            print(f"Loaded txt data with shape: {I.shape}")
            
            T, V = I.shape
            print(f"Updated T={T}, V={V} based on I_PATH data")
            
            S = np.full((T, V), 1e6, dtype=float)
            print("Using fixed S values of 1e6 for all nodes and time steps")
            
        else:
            I_data = np.load(I_PATH)  # (T, V, 1)
            if I_data.ndim == 3 and I_data.shape[2] == 1:
                I = I_data[:, :, 0]  # 提取第一列，变成 (T, V)
            else:
                I = I_data
            print(f"Using I data from I_PATH with shape: {I.shape}")
            # 确保形状匹配
            assert I.shape == (T, V), f"I_PATH data shape {I.shape} doesn't match SIR data shape (T={T}, V={V})"
            # 对于非txt文件，使用原来的S数据
            S = sir[:, :, 0]
    else:
        I = sir[:, :, 1]
        S = sir[:, :, 0]
        print("Using I and S data from SIR file")
    
    R = sir[:, :, 2] if C > 2 else np.zeros_like(I)

    # ---- Build window indices ----
    first_all = T_h
    last_exclusive = T - T_p + 1
    if last_exclusive <= first_all:
        raise ValueError("No valid windows: T too small vs T_h/T_p.")
    label_starts_all = np.arange(first_all, last_exclusive, dtype=int)
    assert label_starts_all[-1] == T - T_p, "Window construction bug"

    if not (0 <= test_start_id <= T):
        raise ValueError("test_start_id out of range.")

    is_train_mask = label_starts_all < test_start_id
    label_starts_test = label_starts_all[~is_train_mask]
    N_te = len(label_starts_test)
    print(f"Total TEST  windows: {N_te}   (from {label_starts_test[0] if N_te>0 else 'N/A'} to {label_starts_test[-1] if N_te>0 else 'N/A'})")

    # ---- Allocate outputs (ONLY test) ----
    y_hat_test = np.full((N_te, V, T_p), np.nan, dtype=float)
    uncert_test = np.full((N_te, V, T_p), np.nan, dtype=float)
    y_future_test = np.full((N_te, V, T_p), np.nan, dtype=float)

    # ---- Helper: compute one window/node ----
    def process_one_window_node(label_start: int, v: int):
        hist_start = label_start - T_h
        hist_end = label_start
        fut_start = label_start
        fut_end = label_start + T_p

        if direct_incidence:
            # SIR第二列就是每日新增
            y_hist = I[hist_start:hist_end, v]
            y_future = I[fut_start:fut_end, v]
        else:
            # SIR第二列是累计病例，需要作差
            y_hist = I[hist_start+1:hist_end+1, v] - I[hist_start:hist_end, v]
            if fut_end + 1 <= T:
                y_future = I[fut_start+1:fut_end+1, v] - I[fut_start:fut_end, v]
            else:
                available_end = min(fut_end, T-1)
                if available_end > fut_start:
                    y_future_partial = I[fut_start+1:available_end+1, v] - I[fut_start:available_end, v]
                    last_val = y_future_partial[-1] if len(y_future_partial) > 0 else 0.0
                    remaining_len = fut_end - available_end
                    y_future = np.concatenate([y_future_partial, np.full(remaining_len, last_val)])
                else:
                    current_val = I[fut_start, v] - I[fut_start-1, v] if fut_start > 0 else 0.0
                    y_future = np.full(T_p, max(0.0, current_val))

        y_hist = np.clip(y_hist, 0.0, None)
        y_future = np.clip(y_future, 0.0, None)
        sir_hist = np.stack(
            [S[hist_start:hist_end, v], I[hist_start:hist_end, v], np.zeros(hist_end-hist_start)],
            axis=-1,
        )
        try:
            y_hat, u = forecast_point_and_uncertainty(
                y_hist=y_hist,
                y_future=y_future,
                sir_hist=sir_hist,
                alpha_nb=alpha_nb,
                ut_alpha=ut_alpha,
                ut_beta=ut_beta,
                trim_frac=trim_frac,
                max_neg_frac=max_neg_frac,
                clip_nonneg=clip_nonneg,
                calib_s_min=calib_s_min,
                calib_s_max=calib_s_max,
                max_inc_val=max_inc_val,
                max_ratio_vs_hist=max_ratio_vs_hist,
                compute_sigma_every=compute_sigma_every,
                hess_eps=hess_eps,
                lbfgs_maxiter=lbfgs_maxiter,
                lbfgs_ftol=lbfgs_ftol,
                hist_smooth_k=hist_smooth_k,
            )
        except Exception:
            return None
        return y_hat, u, y_future

    # ---- Compute for TEST ONLY ----
    print("Computing forecasts + uncertainty for TEST windows...")
    for idx_te, label_start in enumerate(tqdm(label_starts_test, desc="Test windows")):
        for v in range(V):
            out = process_one_window_node(label_start, v)
            if out is None:
                continue
            y_hat, u, y_future = out
            y_hat_test[idx_te, v, :] = y_hat
            uncert_test[idx_te, v, :] = u
            y_future_test[idx_te, v, :] = y_future

    # ---- MAE and RMSE for point estimates vs ground truth ----
    valid_mask = ~(np.isnan(y_hat_test) | np.isnan(y_future_test))
    if np.any(valid_mask):
        mae_overall = np.mean(np.abs(y_hat_test[valid_mask] - y_future_test[valid_mask]))
        rmse_overall = np.sqrt(np.mean((y_hat_test[valid_mask] - y_future_test[valid_mask]) ** 2))
        mae_per_step = np.full(T_p, np.nan)
        rmse_per_step = np.full(T_p, np.nan)
        for t in range(T_p):
            step_valid_mask = valid_mask[:, :, t]
            if np.any(step_valid_mask):
                mae_per_step[t] = np.mean(np.abs(y_hat_test[:, :, t][step_valid_mask] - y_future_test[:, :, t][step_valid_mask]))
                rmse_per_step[t] = np.sqrt(np.mean((y_hat_test[:, :, t][step_valid_mask] - y_future_test[:, :, t][step_valid_mask]) ** 2))
        mae_per_node = np.full(V, np.nan)
        rmse_per_node = np.full(V, np.nan)
        for v in range(V):
            node_valid_mask = valid_mask[:, v, :]
            if np.any(node_valid_mask):
                mae_per_node[v] = np.mean(np.abs(y_hat_test[:, v, :][node_valid_mask] - y_future_test[:, v, :][node_valid_mask]))
                rmse_per_node[v] = np.sqrt(np.mean((y_hat_test[:, v, :][node_valid_mask] - y_future_test[:, v, :][node_valid_mask]) ** 2))
        print(f"Overall MAE: {mae_overall:.6f}")
        print(f"Overall RMSE: {rmse_overall:.6f}")
        print(f"MAE per step: {mae_per_step}")
        print(f"RMSE per step: {rmse_per_step}")
    else:
        mae_overall = rmse_overall = np.nan
        mae_per_step = np.full(T_p, np.nan)
        rmse_per_step = np.full(T_p, np.nan)
        mae_per_node = np.full(V, np.nan)
        rmse_per_node = np.full(V, np.nan)
        print("No valid predictions found for MAE/RMSE computation")

    # ---- Save ----
    meta = dict(
        T_h=int(T_h),
        T_p=int(T_p),
        test_start_id=int(test_start_id),
        alpha_nb=float(alpha_nb),
        ut_alpha=float(ut_alpha),
        ut_beta=float(ut_beta),
        trim_frac=float(trim_frac),
        max_neg_frac=float(max_neg_frac),
        clip_nonneg=bool(clip_nonneg),
        calib_s_min=float(calib_s_min),
        calib_s_max=float(calib_s_max),
        max_inc_val=float(max_inc_val),
        max_ratio_vs_hist=float(max_ratio_vs_hist),
        compute_sigma_every=int(compute_sigma_every),
        hess_eps=float(hess_eps),
        lbfgs_maxiter=int(lbfgs_maxiter),
        lbfgs_ftol=float(lbfgs_ftol),
        T=int(T),
        V=int(V),
        mae_overall=float(mae_overall),
        rmse_overall=float(rmse_overall),
        note="y_hat_test/uncert_test/y_future_test cover only TEST windows; label_starts_test provided for downstream split.",
    )
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        y_hat_test=y_hat_test,
        uncert_test=uncert_test,
        y_future_test=y_future_test,
        label_starts_test=label_starts_test,
        mae_per_step=mae_per_step,
        rmse_per_step=rmse_per_step,
        mae_per_node=mae_per_node,
        rmse_per_node=rmse_per_node,
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved: {OUT_PATH}")
    print(f"Shapes -> y_hat_test: {y_hat_test.shape}, uncert_test: {uncert_test.shape}, y_future_test: {y_future_test.shape}")
    print(f"MAE/RMSE per step shape: {mae_per_step.shape}, MAE/RMSE per node shape: {mae_per_node.shape}")

if __name__ == "__main__":
    # SIR_PATH = "/home/guanghui/DiffODE/data/dataset/COVID-US/us20200415_20210415.npy"
    # I_PATH = "/home/guanghui/DiffODE/data/dataset/COVID-US/cases.npy"
    # DATA_NAME = "COVID-US"
    
    # 使用influenza数据
    SIR_PATH = "/home/guanghui/DiffODE/data/dataset/influenza-US/flu_us_20190107_20211227.npy"  # 占位，实际会被I_PATH数据替换
    I_PATH = "/home/guanghui/iclr26/data/us_flu.txt"
    DATA_NAME = "influenza-US"
    T_h = 14
    T_p = 14
    hist_smooth_k = 1
    test_start_id = int(158 * 0.8)  # 158个时间步，80%作为训练
    OUT_PATH = f"uncert_out/{DATA_NAME}_uncert_test_th{T_h}_tp{T_p}_hsk{hist_smooth_k}.npz"

    # 自动切换 SIR/SI/直接新增
    if DATA_NAME in ["COVID-US", "influenza-US"]:
        forecast_point_and_uncertainty = forecast_point_and_uncertainty_si
        direct_incidence = True
        print(f"Using SI model for {DATA_NAME} (incidence = I)")
    else:
        forecast_point_and_uncertainty = forecast_point_and_uncertainty_sir
        direct_incidence = False
        print(f"Using SIR model for {DATA_NAME}")

    run_test_windows_only(
        SIR_PATH=SIR_PATH,
        OUT_PATH=OUT_PATH,
        T_h=T_h,
        T_p=T_p,
        test_start_id=test_start_id,
        ut_alpha=0.2,
        ut_beta=2.0,
        compute_sigma_every=4,
        lbfgs_maxiter=150,
        lbfgs_ftol=1e-6,
        hess_eps=5e-4,
        calib_s_min=0.3,
        calib_s_max=3.0,
        max_inc_val=1e6,
        max_ratio_vs_hist=100.0,
        trim_frac=0.15,
        max_neg_frac=0.0,
        clip_nonneg=True,
        hist_smooth_k=hist_smooth_k,
        forecast_point_and_uncertainty=forecast_point_and_uncertainty,
        direct_incidence=direct_incidence,
        I_PATH=I_PATH,  # 传入I_PATH参数
    )