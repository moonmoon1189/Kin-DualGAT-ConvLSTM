import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import CBFQADataset, collate_fn
from models.kin_dual_gat_convstm import KinDualGATConvLSTM
from losses import total_loss


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(),
                             lr=config['training']['lr'],
                             weight_decay=config['training']['weight_decay'])


def warmup_cosine_scheduler(optimizer, epoch, total_epochs, warmup_epochs, max_lr, min_lr):
    if epoch < warmup_epochs:
        lr = max_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    with open("configs/main.yaml", 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['experiment']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(CBFQADataset(config['data']['data_path'], "Train"),
                              batch_size=config['data']['batch_size'],
                              shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(CBFQADataset(config['data']['data_path'], "Validation"),
                            batch_size=config['data']['batch_size'],
                            collate_fn=collate_fn)

    model = KinDualGATConvLSTM().to(device)
    optimizer = build_optimizer(model, config)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['training']['epochs']):
        warmup_cosine_scheduler(optimizer, epoch, config['training']['epochs'],
                                config['training']['warmup_epochs'],
                                config['training']['lr'], config['training']['min_lr'])
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_def, x_off = batch['defender'].to(device), batch['attacker'].to(device)
            y, lengths = batch['score'].to(device), batch['length'].to(device)

            optimizer.zero_grad()
            y_pred, com_traj, angles = model(x_def, x_off)
            loss = total_loss(y_pred, y, com_traj, angles, lengths, config['losses']['lambda_phy'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            train_loss += loss.item()

        # Validate (Early stopping)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_def, x_off = batch['defender'].to(device), batch['attacker'].to(device)
                y, lengths = batch['score'].to(device), batch['length'].to(device)
                y_pred, com_traj, angles = model(x_def, x_off)
                val_loss += total_loss(y_pred, y, com_traj, angles, lengths, config['losses']['lambda_phy']).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print("Early stopping triggered!")
                break


if __name__ == "__main__":
    train()