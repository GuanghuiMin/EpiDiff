import torch
import numpy as np
from epilearn.models.SpatialTemporal.CNNRNN_Res import CNNRNN_Res

def cnnrnn_forecast(data, adj_matrix=None, T_h=14, T_p=14, device="cuda", num_epochs=50, lr=1e-3):
    """
    Train and forecast using CNNRNN_Res model.

    Parameters:
    - data: np.ndarray or torch.Tensor, shape (T, N, F)
    - adj_matrix: np.ndarray or torch.Tensor, shape (N, N)
    - T_h: int, history window size
    - T_p: int, prediction horizon
    - device: str, "cuda" or "cpu"
    - num_epochs: int, number of training epochs
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
    model = CNNRNN_Res(
        num_nodes=N,
        num_features=F,
        num_timesteps_input=T_h,
        num_timesteps_output=T_p,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for t in range(T - T_h - T_p + 1):
            x = data[t:t+T_h].unsqueeze(0).to(device)  # (1, T_h, N, F)
            y = data[t+T_h:t+T_h+T_p].permute(1, 0, 2)[:, :, 0]  # (N, T_p)
            y = y.permute(1, 0).contiguous().to(device)  # (T_p, N)

            optimizer.zero_grad()
            pred = model(x, adj_matrix.to(device)).squeeze(0)  # (T_p, N)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / (T - T_h - T_p + 1):.4f}")

    # Forecasting
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