# run_all_windows_ut_all.py
# 计算所有窗口的点估计 + 不确定性（包含训练/测试窗口），并一并保存。

import os
import numpy as np
from tqdm import tqdm

# 加速版稳健 UT/Laplace 实现（上一条我给你的版本）
from steering.robust_ut import forecast_point_and_uncertainty_robust as forecast_point_and_uncertainty

def run_all_windows(
    SIR_PATH: str,
    OUT_PATH: str,
    T_h: int,
    T_p: int,
    test_start_id: int,
    # --- 原有超参 ---
    alpha_nb: float = 5.0,
    ut_alpha: float = 0.2,
    ut_beta: float  = 2.0,
    # --- 新增：稳健/裁剪/兜底 ---
    trim_frac: float = 0.15,
    max_neg_frac: float = 0.0,
    clip_nonneg: bool = True,
    calib_s_min: float = 0.3,
    calib_s_max: float = 3.0,
    max_inc_val: float = 1e6,
    max_ratio_vs_hist: float = 100.0,
    # --- 新增：加速开关 ---
    compute_sigma_every: int = 4,   # 每隔多少步重算一次协方差；1 表示每步都算
    hess_eps: float = 5e-4,
    lbfgs_maxiter: int = 150,
    lbfgs_ftol: float = 1e-6,
):
    """
    输出：
      - y_hat_all:      (N_all, V, T_p)   # 点估计
      - uncert_all:     (N_all, V, T_p)   # 数据感知不确定性
      - y_future_all:   (N_all, V, T_p)   # 真实未来（由 S 差分）
      - label_starts_all / _train / _test
      - meta: 记录所有关键超参
    """
    # ---- Load SIR ----
    data = np.load(SIR_PATH, allow_pickle=True).item()
    sir = data['SIR']  # (T, V, 3)
    T, V, C = sir.shape
    assert C == 3, "SIR last dim must be 3 = [S, I, R]"
    print(f"SIR data shape: {sir.shape}")

    S = sir[:, :, 0]
    I = sir[:, :, 1]
    R = sir[:, :, 2]

    # ---- Build window indices ----
    first_all = T_h
    last_exclusive = T - T_p
    if last_exclusive <= first_all:
        raise ValueError("No valid windows: T too small vs T_h/T_p.")
    label_starts_all = np.arange(first_all, last_exclusive, dtype=int)

    if not (0 <= test_start_id <= T):
        raise ValueError("test_start_id out of range.")

    is_train_mask = label_starts_all < test_start_id
    label_starts_train = label_starts_all[is_train_mask]
    label_starts_test  = label_starts_all[~is_train_mask]

    N_all = len(label_starts_all)
    N_tr  = len(label_starts_train)
    N_te  = len(label_starts_test)
    print(f"Total ALL windows:   {N_all}  (from {label_starts_all[0]} to {label_starts_all[-1]})")
    print(f"Total TRAIN windows: {N_tr}   (from {label_starts_train[0] if N_tr>0 else 'N/A'} to {label_starts_train[-1] if N_tr>0 else 'N/A'})")
    print(f"Total TEST  windows: {N_te}   (from {label_starts_test[0]  if N_te>0 else 'N/A'} to {label_starts_test[-1]  if N_te>0 else 'N/A'})")

    # ---- Allocate outputs ----
    y_hat_all    = np.full((N_all, V, T_p), np.nan, dtype=float)
    uncert_all   = np.full((N_all, V, T_p), np.nan, dtype=float)
    y_future_all = np.full((N_all, V, T_p), np.nan, dtype=float)

    # ---- Helper: compute for one window+node ----
    def process_one_window_node(label_start:int, v:int):
        hist_start = label_start - T_h
        hist_end   = label_start            # exclusive
        fut_start  = label_start
        fut_end    = label_start + T_p      # exclusive

        # history SIR for node v: shape (T_h, 3)
        sir_hist = np.stack(
            [S[hist_start:hist_end, v],
             I[hist_start:hist_end, v],
             R[hist_start:hist_end, v]],
            axis=-1
        )

        # Build incidence from S differences
        S_hist = S[hist_start:hist_end, v]     # (T_h,)
        S_fut  = S[fut_start-1:fut_end, v]     # include S at t-1 to get T_p diffs

        if (not np.all(np.isfinite(S_hist))) or (not np.all(np.isfinite(S_fut))):
            return None

        y_hist   = np.maximum(S_hist[:-1] - S_hist[1:], 0.0)   # (T_h-1,)
        y_future = np.maximum(S_fut[:-1]  - S_fut[1:],  0.0)   # (T_p,)

        try:
            y_hat, u = forecast_point_and_uncertainty(
                y_hist=y_hist,
                y_future=y_future,     # 测试也用真值做“数据感知”不确定性
                sir_hist=sir_hist,
                alpha_nb=alpha_nb,
                ut_alpha=ut_alpha,
                ut_beta=ut_beta,
                # 透传稳健/兜底/加速参数
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
            )
        except Exception:
            return None

        return y_hat, u, y_future

    # ---- Compute for ALL windows ----
    print("Computing forecasts + uncertainty for ALL windows...")
    for idx_all, label_start in enumerate(tqdm(label_starts_all, desc="All windows")):
        for v in range(V):
            out = process_one_window_node(label_start, v)
            if out is None:
                continue
            y_hat, u, y_future = out
            y_hat_all[idx_all, v, :]    = y_hat
            uncert_all[idx_all, v, :]   = u
            y_future_all[idx_all, v, :] = y_future

    # ---- Save ----
    meta = dict(
        sir_path=SIR_PATH,
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
        T=int(T), V=int(V),
        note=("y_hat_all/uncert_all/y_future_all 覆盖所有窗口；"
              "label_starts_train / label_starts_test 便于后续切分。"
              "不确定性为'数据感知'（使用真实未来构造的参考标签）。"),
    )

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        y_hat_all=y_hat_all,                     # (N_all, V, T_p)
        uncert_all=uncert_all,                   # (N_all, V, T_p)
        y_future_all=y_future_all,               # (N_all, V, T_p)
        label_starts_all=label_starts_all,       # (N_all,)
        label_starts_train=label_starts_train,   # (N_tr,)
        label_starts_test=label_starts_test,     # (N_te,)
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved: {OUT_PATH}")
    print(f"Shapes -> y_hat_all: {y_hat_all.shape}, uncert_all: {uncert_all.shape}, y_future_all: {y_future_all.shape}")

if __name__ == "__main__":
    # ===== 只改这几项即可 =====
    SIR_PATH = "/home/guanghui/DiffODE/data/dataset/COVID-US/us20200415_20210415.npy"
    DATA_NAME = "COVID-US"
    T_h = 14
    T_p = 14
    test_start_id = int(366 * 0.8)
    OUT_PATH = f"./uncert_out/{DATA_NAME}_uncert_th{T_h}_tp{T_p}_ALL.npz"

    run_all_windows(
        SIR_PATH=SIR_PATH,
        OUT_PATH=OUT_PATH,
        T_h=T_h,
        T_p=T_p,
        test_start_id=test_start_id,
        # --- 根据需要调这些加速/稳健参数 ---
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
    )