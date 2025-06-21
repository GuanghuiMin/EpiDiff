import torch
import numpy as np
from epilearn.models.SpatialTemporal.ColaGNN import ColaGNN

def colagnn_forecast(data, adj_matrix=None, T_h=14, T_p=14, device="cuda", epochs=20, lr=1e-3):
    """
    Forecast future values using ColaGNN model with automatic training.

    Parameters:
    - data: np.ndarray or torch.Tensor, shape (T, N, F)
    - adj_matrix: np.ndarray or torch.Tensor, shape (N, N)
    - T_h: int, history window size
    - T_p: int, prediction horizon
    - device: str, "cuda" or "cpu"
    - epochs: int, number of training epochs
    - lr: float, learning rate

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
    model = ColaGNN(
        num_nodes=N,
        num_features=F,
        num_timesteps_input=T_h,
        num_timesteps_output=T_p,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    import torch.nn as nn
    loss_fn = nn.MSELoss()

    # Prepare training data (sliding windows)
    train_X = []
    train_Y = []
    for t in range(T - T_h - T_p + 1):
        x = data[t:t+T_h]  # (T_h, N, F)
        y = data[t+T_h:t+T_h+T_p, :, 0]  # (T_p, N)
        train_X.append(x)
        train_Y.append(y)

    train_X = torch.stack(train_X, dim=0).to(device)  # (B, T_h, N, F)
    train_Y = torch.stack(train_Y, dim=0).to(device)  # (B, T_p, N)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_X, adj_matrix.to(device))  # (B, T_p, N)
        loss = loss_fn(output, train_Y)
        loss.backward()
        optimizer.step()

    # Evaluation (prediction)
    model.eval()
    pred_list = []
    true_list = []
    for t in range(T - T_h - T_p + 1):
        x = data[t:t+T_h].unsqueeze(0).to(device)  # (1, T_h, N, F)
        y = data[t+T_h:t+T_h+T_p].permute(1, 0, 2)[:, :, 0]  # (N, T_p)
        y = y.permute(1, 0).contiguous()  # (T_p, N)
        with torch.no_grad():
            pred = model(x, adj_matrix.to(device))  # (1, T_p, N)
            pred = pred.squeeze(0)[-1]  # (N,)
        pred_list.append(pred.cpu().numpy())
        true_list.append(y[-1].cpu().numpy())

    pred_array = np.stack(pred_list, axis=0)  # (T-T_h-T_p+1, N)
    true_array = np.stack(true_list, axis=0)  # (T-T_h-T_p+1, N)

    return pred_array, true_array