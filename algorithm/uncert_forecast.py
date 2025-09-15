# train_uncert_with_encoder.py
# 预测（训练窗口的）UT 不确定性标签：
# - 标签: ./uncert_out/COVID-JP_uncert_th14_tp7.npz   (只包含 train windows 的 uncertainty)
# - 数据: /home/guanghui/DiffODE/data/dataset/COVID-JP/jp20200401_20210921.npy
#
# 输出:
#   - ./uncert_out/uncert_head_best.pth / uncert_head_last.pth
#   - ./uncert_out/uncert_head_stats.npz  (保存 mu, sigma，便于推理时反标准化)

import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 路径（按需修改）======
UNCERT_NPZ = "./uncert_out/COVID-JP_uncert_th14_tp7.npz"
DATA_PATH  = "/home/guanghui/DiffODE/data/dataset/COVID-JP/jp20200401_20210921.npy"
OUT_DIR    = "./uncert_out"

# ====== 超参 ======
T_h, T_p = 14, 7
VAL_RATIO   = 0.1
LR_HEAD     = 3e-3          # 头部学习率（拉高）
LR_ENC      = 1e-3          # 编码器/特征抽取器学习率（拉高）
EPOCHS      = 60
BATCH_WIN   = 8
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 你的 GNN 编码器 ======
from steering.gnn import mepognn


# ---------- 特征抽取（当 mepognn.encode 不可用时的 fallback） ----------
class FeatureMixer(nn.Module):
    """
    输入:
      x_node: (B, Cn, V, T_h)
      x_od:   (B, T_h, V, V, 1)
      mech:   (B, 1, V, T_p)  # 来自 mepognn.forward 的 I_new，作为附加通道
    输出:
      feat:   (B, C_mix, V, T_p)
    """
    def __init__(self, Cn, V, T_h, T_p, hidden=64):
        super().__init__()
        self.T_h, self.T_p = T_h, T_p
        self.V = V

        self.temporal1 = nn.Conv2d(Cn, hidden, kernel_size=(1,3), padding=(0,1), bias=False)
        self.temporal2 = nn.Conv2d(hidden, hidden, kernel_size=(1,3), padding=(0,1), bias=False)
        self.act = nn.ReLU(inplace=True)
        self.gn1 = nn.GroupNorm(1, hidden)
        self.gn2 = nn.GroupNorm(1, hidden)

        # 输入通道: hidden + 2(in/out度) + 1(mech) -> 128 -> 256
        self.mix = nn.Conv2d(hidden + 2 + 1, 128, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(1, 128)
        self.out = nn.Conv2d(128, 256, kernel_size=1, bias=True)

    def forward(self, x_node, x_od, mech):
        # x_node: (B,Cn,V,T_h)
        z = self.gn1(self.act(self.temporal1(x_node)))
        z = self.gn2(self.act(self.temporal2(z)))     # (B,H,V,T_h)
        z = F.interpolate(z, size=(self.V, self.T_p), mode="bilinear", align_corners=False)

        # x_od: (B,T_h,V,V,1) -> 平均入/出度 (B,2,V,T_p)
        od = x_od.squeeze(-1).float()                # (B,T_h,V,V)
        od_mean = od.mean(1)                         # (B,V,V)
        out_deg = od_mean.sum(-1)                    # (B,V)
        in_deg  = od_mean.sum(-2)                    # (B,V)
        deg = torch.stack([in_deg, out_deg], dim=1)  # (B,2,V)
        deg = deg.unsqueeze(-1).expand(-1,-1,-1,self.T_p).contiguous()  # (B,2,V,T_p)

        x = torch.cat([z, deg, mech], dim=1)         # (B,H+2+1,V,T_p)
        x = self.gn3(self.act(self.mix(x)))
        x = self.out(x)                               # (B,256,V,T_p)
        return x


# ----- 1x1 Conv 头：输入 (B,C,V,T) → 输出 (B,V,T) -----
class UncertHeadConv(nn.Module):
    def __init__(self, C_in=256, hidden=256, dropout=0.1):
        super().__init__()
        self.proj1 = nn.Conv2d(C_in, hidden, kernel_size=1, bias=False)
        self.gn1   = nn.GroupNorm(1, hidden)
        self.act1  = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.proj2 = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.gn2   = nn.GroupNorm(1, hidden)
        self.act2  = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.out   = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

    def forward(self, feat, V_expected=None, T_expected=None):
        # feat: (B, C, V, T)
        y = self.drop1(self.act1(self.gn1(self.proj1(feat))))
        y = self.drop2(self.act2(self.gn2(self.proj2(y)))) + y
        y = self.out(y)           # (B,1,V,T)
        y = y.squeeze(1)          # (B,V,T)

        if V_expected is not None and y.shape[1] != V_expected:
            raise RuntimeError(f"V mismatch: {y.shape[1]} vs expected {V_expected}")
        if T_expected is not None and y.shape[2] != T_expected:
            if y.shape[2] > T_expected:
                y = y[:, :, -T_expected:]
            else:
                pad = T_expected - y.shape[2]
                y = F.pad(y, (pad, 0), value=0.0)
        return y


def make_window_tensors(label_start, T_h, T_p, node, sir, od, device):
    hs, he = label_start - T_h, label_start
    node_win = node[hs:he]   # (T_h, V, Cn)
    sir_win  = sir[hs:he]    # (T_h, V, 3)
    od_win   = od[hs:he]     # (T_h, V, V[,1])

    if od_win.ndim == 3:
        od_win = od_win[..., None]   # (T_h, V, V, 1)

    x_node = torch.from_numpy(node_win).permute(2,1,0).unsqueeze(0).float().to(device)  # (1,Cn,V,T_h)
    x_SIR  = torch.from_numpy(sir_win ).unsqueeze(0).float().to(device)                 # (1,T_h,V,3)
    x_od   = torch.from_numpy(od_win  ).unsqueeze(0).float().to(device)                 # (1,T_h,V,V,1)
    return x_node, x_SIR, x_od


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ==== 读训练不确定性标签（仅 train windows） ====
    nz = np.load(UNCERT_NPZ, allow_pickle=True)
    uncert_train = nz["uncert_train"]             # (N_tr, V, T_p)
    label_starts_train = nz["label_starts_train"] # (N_tr,)
    N_tr, V, T_p_chk = uncert_train.shape
    assert T_p_chk == T_p

    # ---- 在 log 空间拟合，先 log1p + 全局 z-score ----
    y_log = np.log1p(np.maximum(uncert_train, 0.0))  # (N_tr,V,T_p)
    mu = float(y_log.mean())
    sigma = float(y_log.std() + 1e-8)
    np.savez(os.path.join(OUT_DIR, "uncert_head_stats.npz"), mu=mu, sigma=sigma)
    print(f"[Target z-stats] mu={mu:.6f}, sigma={sigma:.6f}, "
          f"log1p(u) min/max=({y_log.min():.6f}, {y_log.max():.6f})")

    # ==== 读原始数据 ====
    d = np.load(DATA_PATH, allow_pickle=True).item()
    od   = d["od"]    # (T, V, V[,1])
    node = d["node"]  # (T, V, Cn)
    sir  = d["SIR"]   # (T, V, 3)
    T, V_data, Cn = node.shape
    assert V_data == V
    print(f"[Data] T={T}, V={V}, Cn={Cn}")

    MAX_OD = float(od[..., 0].max()) if od.ndim == 4 else float(od.max())

    # ==== 构建 mepognn ====
    adpinit_dummy = torch.zeros(V, V)
    backbone = mepognn(num_nodes=V,
                       adpinit=adpinit_dummy,
                       glm_type="Dynamic",
                       in_dim=Cn,
                       in_len=T_h,
                       out_len=T_p).to(DEVICE)

    has_encode = callable(getattr(backbone, "encode", None))

    # 如果没有 encode，用 FeatureMixer + ConvHead
    mixer = FeatureMixer(Cn=Cn, V=V, T_h=T_h, T_p=T_p, hidden=64).to(DEVICE)
    head  = UncertHeadConv(C_in=256, hidden=256, dropout=0.1).to(DEVICE)

    # ---- 优化器（修复：不要重复加入 head 参数）----
    if has_encode:
        enc_params = list(backbone.parameters())
    else:
        enc_params = list(mixer.parameters())

    opt = torch.optim.AdamW(
        [
            {"name": "enc_or_mixer", "params": enc_params,       "lr": LR_ENC,  "weight_decay": WEIGHT_DECAY},
            {"name": "head",         "params": head.parameters(), "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
        ]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=4, verbose=True
    )
    criterion = nn.MSELoss()

    # ==== 训练/验证划分 ====
    perm = np.random.permutation(N_tr)
    n_val = max(1, int(VAL_RATIO * N_tr))
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    def encode_features(x_node, x_SIR, x_od):
        """
        返回 (feat, T_eff)，feat 为 (B,C,V,T_eff)
        """
        if has_encode:
            enc = backbone.encode(x_node, x_SIR, od=x_od)  # (B,V,T) 或 (B,C,V,T)
            if enc is None:
                out = backbone(x_node, x_SIR, x_od, max_od=MAX_OD)  # (B,T_p,V,1)
                mech = out.permute(0,3,2,1).contiguous()            # (B,1,V,T_p)
                feat = mixer(x_node, x_od, mech)                    # (B,256,V,T_p)
                return feat, feat.shape[-1]
            if enc.dim() == 3:
                enc = enc.unsqueeze(1)                              # (B,1,V,T_eff)
            out = backbone(x_node, x_SIR, x_od, max_od=MAX_OD)      # (B,T_p,V,1)
            mech = out.permute(0,3,2,1).contiguous()                # (B,1,V,T_p)
            enc_t = enc
            if enc_t.shape[-1] != T_p:
                enc_t = F.interpolate(enc_t, size=(enc_t.shape[-2], T_p), mode="bilinear", align_corners=False)
            feat = torch.cat([enc_t, mech], dim=1)                  # (B,C+1,V,T_p)
            feat = F.relu(nn.Conv2d(feat.size(1), 256, kernel_size=1).to(feat.device)(feat))
            return feat, feat.shape[-1]
        else:
            out = backbone(x_node, x_SIR, x_od, max_od=MAX_OD)      # (B,T_p,V,1)
            mech = out.permute(0,3,2,1).contiguous()                # (B,1,V,T_p)
            feat = mixer(x_node, x_od, mech)                        # (B,256,V,T_p)
            return feat, feat.shape[-1]

    def run_epoch(which):
        is_train = (which == "train")
        backbone.train(is_train)
        mixer.train(is_train)
        head.train(is_train)

        indices = tr_idx if is_train else val_idx
        losses = []

        pbar = tqdm(range(0, len(indices), BATCH_WIN), desc=f"{which.capitalize()} epoch")
        for k in pbar:
            batch_ids = indices[k:k+BATCH_WIN]

            feat_list, y_list = [], []
            T_eff_batch = None

            for j in batch_ids:
                ls = int(label_starts_train[j])
                x_node, x_SIR, x_od = make_window_tensors(ls, T_h, T_p, node, sir, od, DEVICE)
                feat, T_eff = encode_features(x_node, x_SIR, x_od)            # (1,256,V,T_eff)
                B1, C, V_enc, T_eff = feat.shape
                if V_enc != V:
                    raise RuntimeError(f"V mismatch: {V_enc} vs {V}")

                # 标签: log1p → z-score，并对齐到 T_eff（末尾对齐）
                y_true = y_log[j]                    # (V,T_p)
                if T_eff != T_p:
                    if T_eff < T_p:
                        y_true = y_true[:, -T_eff:]
                    else:
                        y_t = torch.from_numpy(y_true).unsqueeze(0).unsqueeze(0)  # (1,1,V,T_p)
                        y_t = F.interpolate(y_t, size=(V, T_eff), mode="bilinear", align_corners=False)
                        y_true = y_t.squeeze().cpu().numpy()                      # (V,T_eff)

                y_true = (y_true - mu) / sigma
                y_true = torch.from_numpy(y_true).unsqueeze(0).float().to(DEVICE) # (1,V,T_eff)

                feat_list.append(feat)
                y_list.append(y_true)
                T_eff_batch = T_eff if T_eff_batch is None else min(T_eff_batch, T_eff)

            feat_b = torch.cat([f[..., -T_eff_batch:] for f in feat_list], dim=0)  # (B,256,V,Tb)
            y_b    = torch.cat([y[..., -T_eff_batch:] for y in y_list], dim=0)     # (B,V,Tb)

            with torch.set_grad_enabled(is_train):
                y_pred = head(feat_b, V_expected=V, T_expected=y_b.shape[-1])      # (B,V,Tb)
                loss = criterion(y_pred, y_b)

                if is_train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(sum([list(pg['params']) for pg in opt.param_groups], []), GRAD_CLIP)
                    opt.step()

            losses.append(loss.item())
            pbar.set_postfix(mse=f"{np.mean(losses):.6f}")

        # 原尺度 RMSE 监控（取部分样本）
        with torch.no_grad():
            rmse_list = []
            probe = indices[:min(len(indices), 64)]
            for j in probe:
                ls = int(label_starts_train[j])
                x_node, x_SIR, x_od = make_window_tensors(ls, T_h, T_p, node, sir, od, DEVICE)
                feat, T_eff = encode_features(x_node, x_SIR, x_od)
                y_pred_std = head(feat, V_expected=V, T_expected=T_eff)  # (1,V,T_eff)

                y_pred_log = y_pred_std.squeeze(0).cpu().numpy() * sigma + mu
                y_pred_u   = np.expm1(y_pred_log).clip(min=0.0)

                y_true_u = uncert_train[j]
                if T_eff != T_p:
                    if T_eff < T_p:
                        y_true_u = y_true_u[:, -T_eff:]
                    else:
                        t = torch.from_numpy(y_true_u).unsqueeze(0).unsqueeze(0)
                        t = F.interpolate(t, size=(V, T_eff), mode="bilinear", align_corners=False)
                        y_true_u = t.squeeze().cpu().numpy()
                rmse = np.sqrt(np.mean((y_pred_u - y_true_u)**2))
                rmse_list.append(rmse)
            rmse_est = float(np.mean(rmse_list)) if rmse_list else float("nan")

        return float(np.mean(losses)), rmse_est

    best_val = math.inf
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_rmse = run_epoch("train")
        val_loss, val_rmse = run_epoch("val")

        print(f"[Epoch {ep:02d}] train MSE(z) {tr_loss:.6f} | val MSE(z) {val_loss:.6f} | "
              f"val RMSE(u)≈ {val_rmse:.2f}")

        scheduler.step(val_loss)

        torch.save(
            {"backbone": backbone.state_dict(),
             "mixer": mixer.state_dict(),
             "head": head.state_dict(),
             "mu": mu, "sigma": sigma},
            os.path.join(OUT_DIR, "uncert_head_last.pth")
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"backbone": backbone.state_dict(),
                 "mixer": mixer.state_dict(),
                 "head": head.state_dict(),
                 "mu": mu, "sigma": sigma},
                os.path.join(OUT_DIR, "uncert_head_best.pth")
            )
            print("  -> New best model saved")


if __name__ == "__main__":
    main()