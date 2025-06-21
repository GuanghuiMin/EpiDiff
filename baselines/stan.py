import torch
import numpy as np
from epilearn.models.SpatialTemporal.STAN import STAN

def stan_forecast(data, adj_matrix=None, T_h=14, T_p=14, device="cuda"):
    """
    Forecast future values using STAN model.

    Parameters:
    - data: np.ndarray or torch.Tensor, shape (T, N, F)
    - adj_matrix: np.ndarray or torch.Tensor, shape (N, N)
    - T_h: int, history window size
    - T_p: int, prediction horizon
    - device: str, "cuda" or "cpu"

    Returns:
    - pred: np.ndarray, shape (T-T_h-T_p+1, N)
    - true: np.ndarray, shape (T-T_h-T_p+1, N)
    """
    data = data.feature if hasattr(data, 'feature') else data
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

    T, N, F = data.shape
    model = STAN(
        num_nodes=N,
        num_features=F,
        num_timesteps_input=T_h,
        num_timesteps_output=T_p,
        device=device,
    ).to(device)
    model.eval()

    pred_list = []
    true_list = []

    for t in range(T - T_h - T_p + 1):
        x = data[t:t+T_h].unsqueeze(0).to(device)  # (1, T_h, N, F)
        if x.shape[-1] == 1:
            x = x.repeat(1, 1, 1, 3) ##这里需要解决的，后续需要改进
        y = data[t+T_h:t+T_h+T_p].permute(1, 0, 2)[:, :, 0]  # (N, T_p)
        y = y.permute(1, 0).contiguous()  # (T_p, N)

        with torch.no_grad():
            states = x  # use x as a placeholder for states assuming infection data is in the same structure
            pred, _ = model(x, adj_matrix.to(device), states=states)  # (1, T_p, N, 2)
            pred = pred[0, -1, :, 0]  # take new infection prediction from (1, T_p, N, 2)

        pred_list.append(pred.cpu().numpy())
        true_list.append(y[-1].cpu().numpy())

    pred_array = np.stack(pred_list, axis=0)  # (T-T_h-T_p+1, N)
    true_array = np.stack(true_list, axis=0)  # (T-T_h-T_p+1, N)

    return pred_array, true_array