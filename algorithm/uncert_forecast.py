# train_uncert_with_encoder.py
# Train an MLP to predict UT uncertainty from MepoGNN encoder embeddings.
# Uses:
#   - ./uncert_out/COVID-JP_uncert_th14_tp7.npz  (labels + window indices)
#   - /home/guanghui/DiffODE/data/dataset/COVID-JP/jp20200401_20210921.npy (od/node/SIR)
#
# Output:
#   - uncert_mlp_best.pth  (best val)
#   - uncert_mlp_last.pth  (last epoch)
#
# Notes:
#   - If you have a trained MepoGNN backbone checkpoint, set BACKBONE_CHECKPOINT below.
#   - Otherwise也可以训练 MLP（embedding 随机，但能跑通流程；效果不代表真实）。

import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from steering.encoder import mepognn

# =========================
# Config
# =========================
UNCERT_NPZ_PATH = "./uncert_out/COVID-JP_uncert_th14_tp7.npz"
RAW_DATA_PATH   = "/home/guanghui/DiffODE/data/dataset/COVID-JP/jp20200401_20210921.npy"

GLM_TYPE = "Dynamic"         # "Dynamic" (uses od) or "Adaptive"
BACKBONE_CHECKPOINT = None   # e.g., "checkpoints/mepognn_backbone.pth" (optional)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_WINDOWS = 8            # how many windows per batch (encoder batch维)
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN = 256
DROPOUT = 0.1
VAL_FRAC = 0.1               # validation fraction within train windows
SEED = 42                    # reproducibility

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# Models
# =========================
class MLPHead(nn.Module):
    """Tiny MLP to map encoder features -> log-uncertainty."""
    def __init__(self, in_dim, hidden=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        # x: (N, in_dim)
        return self.net(x).squeeze(-1)  # (N,)

# =========================
# IO helpers
# =========================
def load_uncert_outputs(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    y_hat_all = z["y_hat_all"]                 # (N_all, V, T_p)
    label_starts_all = z["label_starts_all"]   # (N_all,)
    uncert_train = z["uncert_train"]           # (N_tr, V, T_p)
    y_future_train = z["y_future_train"]       # (N_tr, V, T_p)  # optional for分析
    label_starts_train = z["label_starts_train"]  # (N_tr,)
    meta = z["meta"].item()
    return (y_hat_all, label_starts_all, uncert_train,
            y_future_train, label_starts_train, meta)

def load_raw_data(npy_path):
    d = np.load(npy_path, allow_pickle=True).item()
    od = d["od"]            # (T, V, V, 1?) or (T, V, V)
    if od.ndim == 3:        # ensure last dim=1 for einsum shape一致
        od = od[..., None]
    node = d["node"]        # (T, V, Cnode)
    SIR = d["SIR"]          # (T, V, 3)
    return od, node, SIR

# =========================
# Window builders
# =========================
def build_window_batch_arrays(od, node, SIR, label_starts, T_h, T_p):
    """
    From a batch of label_starts, build encoder inputs:
      x_node: (B, C_in, V, T_h)
      x_SIR : (B, T_h, V, 3)   (仅保留接口；encoder里不直接使用)
      x_od  : (B, T_h, V, V, 1)  (Dynamic必需；Adaptive可忽略)
    """
    T, V, _ = SIR.shape
    Cnode = node.shape[-1]
    B = len(label_starts)
    x_node = np.zeros((B, Cnode, V, T_h), dtype=np.float32)
    x_SIR  = np.zeros((B, T_h, V, 3), dtype=np.float32)
    x_od   = np.zeros((B, T_h, V, V, 1), dtype=np.float32)

    for bi, ls in enumerate(label_starts):
        hs = ls - T_h
        he = ls
        x_node[bi] = np.transpose(node[hs:he], (2, 1, 0))  # (C_in, V, T_h)
        x_SIR[bi]  = SIR[hs:he]                            # (T_h, V, 3)
        x_od[bi]   = od[hs:he]                             # (T_h, V, V, 1)
    return (torch.from_numpy(x_node).float(),
            torch.from_numpy(x_SIR).float(),
            torch.from_numpy(x_od).float())

# =========================
# Target transform
# =========================
def prepare_targets(uncert_train_np, p_high=99.0):
    """
    Winsorize at p_high, then log1p.
    Returns:
      y_log: (N_tr, V, T_p)  float
      clip_hi: scalar (the p_high percentile), for reference
    """
    flat = uncert_train_np.reshape(-1)
    hi = np.percentile(flat, p_high)
    xw = np.clip(flat, 0.0, hi)
    return np.log1p(xw).reshape(uncert_train_np.shape), hi

# =========================
# Train / Val loop
# =========================
def main():
    # ---- Load UT outputs (labels+indices) ----
    (y_hat_all, label_starts_all, uncert_train,
     y_future_train, label_starts_train, meta) = load_uncert_outputs(UNCERT_NPZ_PATH)

    T_h = int(meta["T_h"])
    T_p = int(meta["T_p"])
    print(f"[NPZ] T_h={T_h}, T_p={T_p}")
    print(f"[NPZ] y_hat_all: {y_hat_all.shape}, uncert_train: {uncert_train.shape}")

    # ---- Prepare targets on train windows ----
    y_train_log, clip_hi = prepare_targets(uncert_train, p_high=99.0)
    print(f"[Target] winsor p99={clip_hi:.3f}, using log1p for training.")

    # ---- Train/Val split（基于 train windows）----
    N_tr = len(label_starts_train)
    n_val = max(1, int(N_tr * VAL_FRAC))
    perm = np.random.permutation(N_tr)
    val_pos = perm[:n_val]
    tr_pos  = perm[n_val:]
    tr_label_starts = label_starts_train[tr_pos]
    val_label_starts= label_starts_train[val_pos]
    print(f"[Split] Train windows: {len(tr_label_starts)} | Val windows: {len(val_label_starts)}")

    # ---- Load raw data for encoder inputs ----
    od_np, node_np, SIR_np = load_raw_data(RAW_DATA_PATH)
    T, V, _ = SIR_np.shape
    Cnode = node_np.shape[-1]

    # ---- Build backbone (encoder) ----
    if GLM_TYPE == "Adaptive":
        adpinit = torch.ones((V, V), dtype=torch.float32)
    else:
        adpinit = torch.ones((V, V), dtype=torch.float32)  # placeholder (Dynamic不使用，但构造器需要)

    backbone = mepognn(
        num_nodes=V, adpinit=adpinit, glm_type=GLM_TYPE,
        in_dim=Cnode, in_len=T_h, out_len=T_p,
        residual_channels=32, dilation_channels=32,
        skip_channels=256, end_channels=512,
        kernel_size=2, blocks=2, layers=3
    ).to(DEVICE).eval()

    if BACKBONE_CHECKPOINT and os.path.isfile(BACKBONE_CHECKPOINT):
        ckpt = torch.load(BACKBONE_CHECKPOINT, map_location=DEVICE)
        backbone.load_state_dict(ckpt, strict=False)
        print(f"[Backbone] Loaded weights from {BACKBONE_CHECKPOINT}")
    else:
        print("[Backbone] WARNING: no checkpoint provided; embeddings are untrained (流程可跑，效果仅供调试).")

    # ---- Derive encoder output dim ----
    with torch.no_grad():
        x_node_t, x_SIR_t, x_od_t = build_window_batch_arrays(
            od_np, node_np, SIR_np, [label_starts_train[0]], T_h, T_p
        )
        feat_t = backbone.encode(
            x_node_t.to(DEVICE),
            x_SIR_t.to(DEVICE),
            od=x_od_t.to(DEVICE) if GLM_TYPE=="Dynamic" else None
        )
        C_ctx = feat_t.shape[1]
    print(f"[Encoder] feature dim C_ctx={C_ctx}")

    # ---- MLP head, opt, sched ----
    head = MLPHead(in_dim=C_ctx, hidden=HIDDEN, drop=DROPOUT).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    mse = nn.MSELoss(reduction="mean")

    def get_targets_subset(label_starts_subset):
        """
        返回与 subset 对应的 y_log targets: (B, V, T_p)（按 label_start升序索引）
        """
        # 将 subset 中每个 label_start 映射到 label_starts_train 的位置
        pos = np.searchsorted(label_starts_train, label_starts_subset)
        return torch.from_numpy(y_train_log[pos]).float()  # (B, V, T_p)

    def epoch_pass(label_starts_subset, is_train=True, desc=""):
        head.train(is_train)
        total_loss, total_n = 0.0, 0

        # 按窗口batch
        for i in tqdm(range(0, len(label_starts_subset), BATCH_WINDOWS), desc=desc):
            batch_ls = label_starts_subset[i:i+BATCH_WINDOWS]
            B = len(batch_ls)

            with torch.no_grad():
                x_node, x_SIR, x_od = build_window_batch_arrays(
                    od_np, node_np, SIR_np, batch_ls, T_h, T_p
                )
                x_node = x_node.to(DEVICE)
                x_SIR  = x_SIR.to(DEVICE)
                x_od   = x_od.to(DEVICE)

                feat = backbone.encode(
                    x_node, x_SIR, od=x_od if GLM_TYPE=="Dynamic" else None
                )                                              # (B, C_ctx, V, T_p)
                # flatten to (B*V*T_p, C_ctx)
                feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C_ctx)

                y_tar = get_targets_subset(batch_ls).to(DEVICE)  # (B, V, T_p)
                y_flat = y_tar.reshape(-1)                        # (B*V*T_p,)

            if is_train:
                opt.zero_grad(set_to_none=True)
                pred = head(feat_flat)                  # (N,)
                loss = mse(pred, y_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                opt.step()
            else:
                with torch.no_grad():
                    pred = head(feat_flat)
                    loss = mse(pred, y_flat)

            total_loss += float(loss) * int(y_flat.numel())
            total_n    += int(y_flat.numel())

        return total_loss / max(1, total_n)

    # ---- Train loop ----
    best_val = math.inf
    for ep in range(1, EPOCHS+1):
        tr_loss = epoch_pass(tr_label_starts, is_train=True,  desc=f"Train[{ep:02d}]")
        if len(val_label_starts) > 0:
            val_loss = epoch_pass(val_label_starts, is_train=False, desc=f" Val [{ep:02d}]")
            print(f"[Epoch {ep:02d}] train MSE={tr_loss:.6f} | val MSE={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(head.state_dict(), "uncert_mlp_best.pth")
        else:
            print(f"[Epoch {ep:02d}] train MSE={tr_loss:.6f}")
        sched.step()

    torch.save(head.state_dict(), "uncert_mlp_last.pth")
    print("[Saved] uncert_mlp_last.pth", "(and uncert_mlp_best.pth if improved).")

    # ---- Quick evaluation preview on first few val windows ----
    if len(val_label_starts) > 0:
        with torch.no_grad():
            preview_ls = val_label_starts[:min(4, len(val_label_starts))]
            x_node, x_SIR, x_od = build_window_batch_arrays(
                od_np, node_np, SIR_np, preview_ls, T_h, T_p
            )
            feat = backbone.encode(
                x_node.to(DEVICE), x_SIR.to(DEVICE), od=x_od.to(DEVICE) if GLM_TYPE=="Dynamic" else None
            )
            pred_log = head(feat.permute(0, 2, 3, 1).reshape(-1, C_ctx)).reshape(len(preview_ls), V, T_p)
            pred_uncert = torch.expm1(pred_log).cpu().numpy()  # inverse of log1p

            # 对齐真实（winsor+log1p之前的原值）做一个直观对比
            # 这里用 npz 中的 uncert_train（train窗口的）；val窗口来自 train split 内部划分
            # 我们把 winsor前的原始值拿来对比（风险：有极端值）。你也可以用 winsor后的再 expm1。
            true_uncert = uncert_train[val_pos[:len(preview_ls)]]

            print("\n[Preview] Pred vs True (val[0]，前5个节点，全 T_p 步):")
            print("Pred (first window, nodes 0..4):")
            print(np.round(pred_uncert[0, :5, :], 3))
            print("True (first window, nodes 0..4):")
            print(np.round(true_uncert[0, :5, :], 3))

if __name__ == "__main__":
    main()