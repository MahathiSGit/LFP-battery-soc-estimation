from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data import Session, FeatureScaler


class SessionDataset(Dataset):
    def __init__(self, sessions: Sequence[Session], scaler: FeatureScaler):
        self.sessions = sessions
        self.scaler = scaler

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        s = self.sessions[idx]
        feats = self.scaler.transform(s.features)
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(s.target, dtype=torch.float32),
            len(s.target),
        )


def collate(batch):
    feats, targets, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = lengths.max().item()
    input_size = feats[0].shape[1]

    x = torch.zeros(len(batch), max_len, input_size)
    y = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len)
    for i, (f, t, n) in enumerate(zip(feats, targets, lengths)):
        x[i, :n] = f
        y[i, :n] = t
        mask[i, :n] = 1.0
    return x, y, mask, lengths


def masked_mse(pred, target, mask):
    sq_err = (pred - target) ** 2 * mask
    return sq_err.sum() / mask.sum().clamp_min(1.0)


def masked_mae(pred, target, mask):
    abs_err = (pred - target).abs() * mask
    return abs_err.sum() / mask.sum().clamp_min(1.0)


def train_one_fold(train_sessions, val_sessions, model_cls, hidden_size: int = 16,
                    lr: float = 1e-2, epochs: int = 300, batch_size: int = 4,
                    weight_decay: float = 1e-3, patience: int = 20, loss_fn=masked_mae,
                    verbose: bool = True, model_kwargs: dict | None = None) -> dict:
    scaler = FeatureScaler().fit(train_sessions)

    train_ds = SessionDataset(train_sessions, scaler)
    val_ds = SessionDataset(val_sessions, scaler)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)),
                               shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, collate_fn=collate)

    model = model_cls(hidden_size=hidden_size, **(model_kwargs or {}))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_mae_soc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y, mask, lengths in train_loader:
            opt.zero_grad()
            pred = model(x, lengths)
            loss = loss_fn(pred, y, mask)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(train_ds)
        history["train_loss"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            for x, y, mask, lengths in val_loader:
                pred = model(x, lengths)
                val_loss = loss_fn(pred, y, mask).item()
                mae = (pred - y).abs() * mask
                val_mae = mae.sum().item() / mask.sum().item()
        history["val_loss"].append(val_loss)
        history["val_mae_soc"].append(val_mae)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d}  train_loss={epoch_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_delta_mae={val_mae:.3f} SoC%")

        if epochs_since_improve >= patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "scaler": scaler, "history": history, "best_val_loss": best_val_loss}


def leave_one_session_out(sessions, model_cls, **train_kwargs):
    results = []
    for i in range(len(sessions)):
        val = [sessions[i]]
        train = [s for j, s in enumerate(sessions) if j != i]
        print(f"\n=== Fold {i}: holding out session {i} "
              f"(anchor SoC={sessions[i].anchor_soc:.1f}%, len={len(sessions[i].target)}) ===")
        result = train_one_fold(train, val, model_cls, **train_kwargs)
        results.append(result)
    return results


def predict_session(model, scaler, session):
    model.eval()
    feats = torch.tensor(scaler.transform(session.features), dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([len(session.target)])
    with torch.no_grad():
        delta_pred = model(feats, lengths).squeeze(0).numpy()
    return session.anchor_soc + delta_pred