import os
import numpy as np
from tqdm import tqdm

# Use your fixed-consistency implementation from ut_laplace.py
from steering.ut_laplace import forecast_point_and_uncertainty

def run_all_windows(
    SIR_PATH: str,
    OUT_PATH: str,
    T_h: int,
    T_p: int,
    test_start_id: int,
    alpha_nb: float = 5.0,
    ut_alpha: float = 1e-1,
    ut_beta: float  = 2.0,
):
    """
    Compute UT mean forecasts for ALL valid label_start windows, and
    compute uncertainty ONLY for TRAIN windows (label_start < test_start_id).

    Saves:
      - y_hat_all:        (N_all, V, T_p)   # UT mean forecasts for all windows
      - label_starts_all: (N_all,)          # window start indices for y_hat_all
      - uncert_train:     (N_tr, V, T_p)    # uncertainty for train windows only
      - y_future_train:   (N_tr, V, T_p)    # ground-truth for train windows
      - label_starts_train:(N_tr,)          # train window start indices
      - meta: dict as np.object array
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
    # All valid windows require: label_start - T_h >= 0 and label_start + T_p <= T
    # So label_start in [T_h, T - T_p)
    first_all = T_h
    last_exclusive = T - T_p
    if last_exclusive <= first_all:
        raise ValueError("No valid windows: T too small vs T_h/T_p.")

    label_starts_all = np.arange(first_all, last_exclusive, dtype=int)

    # TRAIN subset: label_start < test_start_id
    if not (0 <= test_start_id <= T):
        raise ValueError("test_start_id out of range.")
    label_starts_train = label_starts_all[label_starts_all < test_start_id]

    N_all = len(label_starts_all)
    N_tr  = len(label_starts_train)
    print(f"Total ALL windows:   {N_all}  (from {label_starts_all[0]} to {label_starts_all[-1]})")
    print(f"Total TRAIN windows: {N_tr}   (from {label_starts_train[0] if N_tr>0 else 'N/A'} to {label_starts_train[-1] if N_tr>0 else 'N/A'})")

    # ---- Allocate outputs ----
    y_hat_all     = np.full((N_all, V, T_p), np.nan, dtype=float)
    uncert_train  = np.full((N_tr, V, T_p),  np.nan, dtype=float) if N_tr > 0 else np.zeros((0, V, T_p))
    y_future_train= np.full((N_tr, V, T_p),  np.nan, dtype=float) if N_tr > 0 else np.zeros((0, V, T_p))

    # ---- Helper: compute y_hat (& u if needed) for a given window and node ----
    def process_one_window_node(label_start:int, v:int, need_uncert:bool):
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

        # Incidence via S diffs
        S_hist = S[hist_start:hist_end, v]     # (T_h,)
        S_fut  = S[fut_start-1:fut_end, v]     # include S at t-1 to get T_p diffs

        if (not np.all(np.isfinite(S_hist))) or (not np.all(np.isfinite(S_fut))):
            return None

        y_hist = np.maximum(S_hist[:-1] - S_hist[1:], 0.0)  # (T_h-1,)
        if need_uncert:
            y_future = np.maximum(S_fut[:-1] - S_fut[1:], 0.0)  # (T_p,)
        else:
            # For non-uncert windows we only need y_hat; provide a dummy y_future
            y_future = np.zeros(T_p, dtype=float)

        try:
            y_hat, u = forecast_point_and_uncertainty(
                y_hist=y_hist,
                y_future=y_future,
                sir_hist=sir_hist,
                alpha_nb=alpha_nb,
                ut_alpha=ut_alpha,
                ut_beta=ut_beta
            )
        except Exception:
            return None

        return y_hat, (u if need_uncert else None), (y_future if need_uncert else None)

    # ---- Fill ALL windows y_hat ----
    print("Computing forecasts for ALL windows...")
    for idx_all, label_start in enumerate(tqdm(label_starts_all, desc="All windows")):
        for v in range(V):
            out = process_one_window_node(label_start, v, need_uncert=False)
            if out is None:
                continue
            y_hat, _, _ = out
            y_hat_all[idx_all, v, :] = y_hat

    # ---- Fill TRAIN windows uncertainty & y_future ----
    if N_tr > 0:
        print("Computing uncertainty for TRAIN windows...")
        for idx_tr, label_start in enumerate(tqdm(label_starts_train, desc="Train windows")):
            for v in range(V):
                out = process_one_window_node(label_start, v, need_uncert=True)
                if out is None:
                    continue
                y_hat, u, y_future = out
                uncert_train[idx_tr, v, :]   = u
                y_future_train[idx_tr, v, :] = y_future

    # ---- Save ----
    meta = dict(
        sir_path=SIR_PATH,
        T_h=int(T_h),
        T_p=int(T_p),
        test_start_id=int(test_start_id),
        alpha_nb=float(alpha_nb),
        ut_alpha=float(ut_alpha),
        ut_beta=float(ut_beta),
        T=int(T), V=int(V),
        note="y_hat_all covers all windows; uncertainty only for train windows (label_start < test_start_id).",
    )

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        y_hat_all=y_hat_all,                       # (N_all, V, T_p)
        label_starts_all=label_starts_all,         # (N_all,)
        uncert_train=uncert_train,                 # (N_tr, V, T_p)
        y_future_train=y_future_train,             # (N_tr, V, T_p)
        label_starts_train=label_starts_train,     # (N_tr,)
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved: {OUT_PATH}")
    print(f"Shapes -> y_hat_all: {y_hat_all.shape}, uncert_train: {uncert_train.shape}, y_future_train: {y_future_train.shape}")

if __name__ == "__main__":
    # ==== Only edit these five ====
    SIR_PATH = "/home/guanghui/DiffODE/data/dataset/COVID-US/us20200415_20210415.npy"
    Data_name = "COVID-US"
    T_h = 14
    T_p = 14
    test_start_id = int(539 * 0.6)
    OUT_PATH = f"./uncert_out/{Data_name}_uncert_th{T_h}_tp{T_p}.npz"

    # Keep defaults unless you have a reason to change
    run_all_windows(
        SIR_PATH=SIR_PATH,
        OUT_PATH=OUT_PATH,
        T_h=T_h,
        T_p=T_p,
        test_start_id=test_start_id
    )