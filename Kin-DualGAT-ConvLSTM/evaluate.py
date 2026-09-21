import torch
import yaml
from torch.utils.data import DataLoader
from data.dataset import CBFQADataset, collate_fn
from models.kin_dual_gat_convstm import KinDualGATConvLSTM
from metrics import compute_r2, compute_rmse, compute_mae, compute_plcc, compute_srcc


def evaluate():
    with open("configs/main.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KinDualGATConvLSTM().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    test_loader = DataLoader(CBFQADataset(config['data']['data_path'], "Test"),
                             batch_size=config['data']['batch_size'],
                             collate_fn=collate_fn)

    y_trues, y_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_def, x_off = batch['defender'].to(device), batch['attacker'].to(device)
            y = batch['score']
            y_pred, _, _ = model(x_def, x_off)
            y_trues.extend((y.cpu().numpy() * 100).tolist())  # Denormalize to [0,100]
            y_preds.extend((y_pred.cpu().numpy() * 100).tolist())

    print(f"R2: {compute_r2(y_trues, y_preds):.3f}")
    print(f"RMSE: {compute_rmse(y_trues, y_preds):.3f}")
    print(f"MAE: {compute_mae(y_trues, y_preds):.3f}")
    print(f"PLCC: {compute_plcc(y_trues, y_preds):.3f}")
    print(f"SRCC: {compute_srcc(y_trues, y_preds):.3f}")

    # Save predictions
    with open("predictions.txt", "w") as f:
        for t, p in zip(y_trues, y_preds):
            f.write(f"{t[0]}, {p[0]}\n")


if __name__ == "__main__":
    evaluate()