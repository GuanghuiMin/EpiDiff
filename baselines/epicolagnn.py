import torch
import numpy as np
from epilearn.models.SpatialTemporal.EpiColaGNN import EpiColaGNN

def epicolagnn_forecast(data, adj_matrix=None, T_h=14, T_p=14, device="cuda"):
    """
    Forecast future values using ColaGNN model.

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
    model = EpiColaGNN(
        num_nodes=N,
        num_features=F,
        num_timesteps_input=T_h,
        num_timesteps_output=T_p,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Simple training loop
    model.train()
    for epoch in range(5):  # Train for 5 epochs
        total_loss = 0
        for t in range(T - T_h - T_p + 1):
            x = data[t:t+T_h].unsqueeze(0).to(device)  # (1, T_h, N, F)
            y = data[t+T_h:t+T_h+T_p].to(device).permute(1, 0, 2)[:, :, 0]  # (N, T_p)
            y = y.permute(1, 0).contiguous()  # (T_p, N)
            target = y[-1]  # Predict the last step

            pred = model(x, adj_matrix.to(device))
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.squeeze(0)[-1]

            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    model.eval()
    if adj_matrix is not None:
        adj_matrix = adj_matrix.to(device)

    pred_list = []
    true_list = []

    for t in range(T - T_h - T_p + 1):
        x = data[t:t+T_h].unsqueeze(0)
        x = x.to(device)  # (1, T_h, N, F)
        y = data[t+T_h:t+T_h+T_p].to(device).permute(1, 0, 2)[:, :, 0]
        y = y.permute(1, 0).contiguous()  # (T_p, N)

        with torch.no_grad():
            pred = model(x, adj_matrix.to(device))  # (1, T_p, N) or a tuple
            if isinstance(pred, tuple):
                pred = pred[0]  # unpack if model returns a tuple (output, hidden)
            pred = pred.squeeze(0)[-1]  # (N,)

        pred_list.append(pred.cpu().numpy())
        true_list.append(y[-1].cpu().numpy())

    pred_array = np.stack(pred_list, axis=0)  # (T-T_h-T_p+1, N)
    true_array = np.stack(true_list, axis=0)  # (T-T_h-T_p+1, N)

    return pred_array, true_array