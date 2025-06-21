from statsmodels.tsa.ar_model import AutoReg
import numpy as np


def ar(raw_series: np.ndarray, T_h: int, T_p: int):
    """
    对每个节点独立使用 AutoReg 进行滑动窗口预测。
    输入 shape: (T_total, V)
    返回 shape: (B, T_p, V)
    """
    T_total, V = raw_series.shape
    windows = []
    targets = []
    for t in range(T_total - T_h - T_p + 1):
        windows.append(raw_series[t:t+T_h])
        targets.append(raw_series[t+T_h:t+T_h+T_p])
    windows = np.stack(windows)  # (B, T_h, V)
    targets = np.stack(targets)  # (B, T_p, V)

    B = windows.shape[0]
    preds = np.zeros_like(targets)
    for b in range(B):
        for v in range(V):
            try:
                model = AutoReg(windows[b, :, v], lags=1, old_names=False).fit()
                pred = model.predict(start=len(windows[b, :, v]), end=len(windows[b, :, v]) + T_p - 1)
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    pred = np.nan_to_num(pred, nan=windows[b, -1, v], posinf=windows[b, -1, v], neginf=windows[b, -1, v])
            except:
                fallback_value = windows[b, -1, v]
                if np.isnan(fallback_value) or np.isinf(fallback_value):
                    fallback_value = 0.0
                pred = np.tile(fallback_value, T_p)
            preds[b, :, v] = pred
    return preds, targets
