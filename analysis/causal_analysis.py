import os
import re
import glob
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ==============================================================================
# 0) UTILITAIRES - chargement robuste des logs
# ==============================================================================
def _pick_xyz_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Trouve les colonnes X/Y/Z dans un CSV de drone, de façon robuste."""
    if all(c in df.columns for c in ["x_true", "y_true", "z_true"]):
        return "x_true", "y_true", "z_true"
    if all(c in df.columns for c in ["x", "y", "z"]):
        return "x", "y", "z"

    candidates = [c for c in df.columns if c != "t" and pd.api.types.is_numeric_dtype(df[c])]
    if len(candidates) >= 3:
        return candidates[0], candidates[1], candidates[2]

    raise ValueError("Impossible de trouver des colonnes X,Y,Z dans le CSV.")


def _pick_time_column(df: pd.DataFrame) -> Optional[str]:
    if "t" in df.columns and pd.api.types.is_numeric_dtype(df["t"]):
        return "t"
    return None


def _extract_drone_id_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"drone_(\d+)\.csv$", base)
    if m:
        return int(m.group(1))
    m = re.search(r"drone(\d+)\.csv$", base)
    if m:
        return int(m.group(1))
    return None


def load_swarm_positions(log_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retour:
      pos: (T, N, 3)
      t:   (T,) si présent, sinon t = np.arange(T)
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log dir introuvable: {log_dir}")

    all_csv = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    if not all_csv:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {log_dir}")

    # priorités : drone_*.csv ou droneX.csv
    drone_files = [f for f in all_csv if re.search(r"drone_?\d+\.csv$", os.path.basename(f))]
    files = drone_files if drone_files else all_csv

    # trie par id si possible
    files_with_id = []
    for f in files:
        did = _extract_drone_id_from_filename(f)
        files_with_id.append((did, f))
    if any(did is not None for did, _ in files_with_id):
        files_with_id.sort(key=lambda x: (999999 if x[0] is None else x[0], x[1]))
        files = [f for _, f in files_with_id]

    dfs = []
    time_ref = None
    for f in files:
        try:
            df = pd.read_csv(f)
            xcol, ycol, zcol = _pick_xyz_columns(df)
            tcol = _pick_time_column(df)

            xyz = df[[xcol, ycol, zcol]].astype(float).values
            tt = df[tcol].astype(float).values if tcol is not None else None

            dfs.append((xyz, tt, f))
            if time_ref is None and tt is not None:
                time_ref = tt
        except Exception as e:
            print(f"[WARN] skip {f} ({e})")

    if not dfs:
        raise RuntimeError("Aucun CSV exploitable. Vérifie le format des logs.")

    min_len = min(len(xyz) for xyz, _, _ in dfs)
    if min_len < 50:
        raise RuntimeError(f"Logs trop courts ({min_len} pts). Relance simulation.")

    positions = []
    for xyz, _, _ in dfs:
        positions.append(xyz[:min_len])
    pos = np.stack(positions, axis=1)  # (T, N, 3)

    if time_ref is not None:
        t = time_ref[:min_len]
    else:
        t = np.arange(min_len, dtype=float)

    return pos, t


def finite_difference_velocity(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    """pos: (T,N,3), t:(T,) => vel:(T,N,3)"""
    T = pos.shape[0]
    vel = np.zeros_like(pos)
    dt = np.diff(t)
    dt = np.where(dt <= 1e-9, 1.0, dt)  # robustesse
    for k in range(1, T):
        vel[k] = (pos[k] - pos[k - 1]) / dt[k - 1]
    vel[0] = vel[1]
    return vel


# ==============================================================================
# 1) DÉTECTION DES FAILURES (labels)
# ==============================================================================
@dataclass
class FailureThresholds:
    z_crash: float = 0.10
    d_collision: float = 0.30
    v_low: float = 0.20
    v_drop: float = 0.60


def compute_failure_labels(
    pos: np.ndarray,
    vel: np.ndarray,
    thr: FailureThresholds,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retourne 3 labels booléens par time et drone:
      crash:     (T,N)
      collision: (T,N)
      speedloss: (T,N)
    """
    T, N, _ = pos.shape

    crash = pos[:, :, 2] < thr.z_crash

    collision = np.zeros((T, N), dtype=bool)
    if N >= 2:
        for k in range(T):
            pk = pos[k]  # (N,3)
            diff = pk[:, None, :] - pk[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            min_dist = np.min(dist, axis=1)
            collision[k] = min_dist < thr.d_collision

    speed = np.linalg.norm(vel, axis=-1)
    speedloss = speed < thr.v_low
    if T >= 2:
        drop = (speed[:-1] - speed[1:]) > thr.v_drop
        drop = np.vstack([np.zeros((1, N), dtype=bool), drop])
        speedloss = np.logical_or(speedloss, drop)

    return crash, collision, speedloss


def future_window_any(label: np.ndarray, horizon: int) -> np.ndarray:
    T, N = label.shape
    y = np.zeros((T, N), dtype=bool)
    for t in range(T):
        a = t + 1
        b = min(T, t + horizon + 1)
        if a < b:
            y[t] = np.any(label[a:b], axis=0)
        else:
            y[t] = False
    return y


# ==============================================================================
# 2) DATASET - features ego + pairwise + labels future failures
# ==============================================================================
def build_pairwise_features(pos_last: np.ndarray, vel_last: np.ndarray) -> np.ndarray:
    """
    pos_last: (N,3), vel_last:(N,3)
    retourne rel: (N,N,8)
      [dx,dy,dz,dvx,dvy,dvz,dist,closing_speed]
    """
    N = pos_last.shape[0]
    dp = pos_last[None, :, :] - pos_last[:, None, :]  # j - i
    dv = vel_last[None, :, :] - vel_last[:, None, :]

    dist = np.linalg.norm(dp, axis=-1, keepdims=True)
    dist_safe = np.where(dist < 1e-6, 1e-6, dist)
    closing = np.sum(dp * dv, axis=-1, keepdims=True) / dist_safe

    rel = np.concatenate([dp, dv, dist, closing], axis=-1).astype(np.float32)
    for i in range(N):
        rel[i, i, :] = 0.0
    return rel


class SwarmFailureDataset(Dataset):
    def __init__(
        self,
        pos: np.ndarray,      # (T,N,3)
        vel: np.ndarray,      # (T,N,3)
        t: np.ndarray,        # (T,)
        window: int = 10,
        horizon: int = 20,
        thr: FailureThresholds = FailureThresholds(),
    ):
        self.pos = pos.astype(np.float32)
        self.vel = vel.astype(np.float32)
        self.t = t.astype(np.float32)
        self.window = int(window)
        self.horizon = int(horizon)

        T, N, _ = pos.shape
        self.N = N
        self.node_feat_dim = 6  # pos(3)+vel(3)
        self.rel_feat_dim = 8

        crash, collision, speedloss = compute_failure_labels(pos, vel, thr)
        y_crash = future_window_any(crash, horizon)
        y_coll = future_window_any(collision, horizon)
        y_speed = future_window_any(speedloss, horizon)
        self.y = np.stack([y_crash, y_coll, y_speed], axis=-1).astype(np.float32)  # (T,N,3)

        self.indices = []
        for i in range(self.window, T):
            if (i + self.horizon) < T:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        w0 = i - self.window
        w1 = i

        hist_pos = self.pos[w0:w1]  # (W,N,3)
        hist_vel = self.vel[w0:w1]  # (W,N,3)

        hist = np.concatenate([hist_pos, hist_vel], axis=-1)  # (W,N,6)
        hist = hist.transpose(1, 0, 2).reshape(self.N, -1)     # (N, W*6)

        pos_last = self.pos[w1 - 1]
        vel_last = self.vel[w1 - 1]
        rel = build_pairwise_features(pos_last, vel_last)      # (N,N,8)

        target = self.pos[i]           # (N,3)
        labels = self.y[w1 - 1]        # (N,3)

        return (
            torch.from_numpy(hist),      # (N, W*6)
            torch.from_numpy(rel),       # (N, N, 8)
            torch.from_numpy(target),    # (N, 3)
            torch.from_numpy(labels),    # (N, 3)
        )


# ==============================================================================
# 3) MODÈLE : NRI + prédiction état + tête risque failures
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphFailureNRI(nn.Module):
    def __init__(self, num_drones: int, node_in_dim: int, rel_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.N = num_drones
        self.node_in_dim = node_in_dim
        self.rel_dim = rel_dim

        self.edge_enc = MLP(node_in_dim * 2 + rel_dim, hidden, 1, dropout=dropout)
        self.msg_net = MLP(node_in_dim, hidden, hidden, dropout=dropout)
        self.state_out = MLP(node_in_dim + hidden, hidden, 3, dropout=dropout)
        self.risk_out = MLP(node_in_dim + hidden, hidden, 3, dropout=dropout)

    def edge_prediction(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        B, N, Fdim = x.shape
        assert N == self.N

        xi = x.unsqueeze(2).expand(-1, -1, N, -1)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)

        enc_in = torch.cat([xi, xj, rel], dim=-1)
        logits = self.edge_enc(enc_in).squeeze(-1)
        adj = torch.sigmoid(logits)

        eye = torch.eye(N, device=x.device).unsqueeze(0)
        adj = adj * (1.0 - eye)
        return adj

    def forward(self, x: torch.Tensor, rel: torch.Tensor, adj_override: Optional[torch.Tensor] = None):
        if adj_override is None:
            adj = self.edge_prediction(x, rel)
        else:
            adj = adj_override

        msgs = self.msg_net(x)
        agg = torch.matmul(adj, msgs)

        dec_in = torch.cat([x, agg], dim=-1)
        pred_pos = self.state_out(dec_in)
        risk_logits = self.risk_out(dec_in)
        risk = torch.sigmoid(risk_logits)

        return pred_pos, risk, adj


# ==============================================================================
# 4) ENTRAÎNEMENT + ÉVALUATION + ABLATION D'ARÊTES
# ==============================================================================
@dataclass
class TrainConfig:
    window: int = 10
    horizon: int = 20
    hidden: int = 128
    dropout: float = 0.1
    epochs: int = 120
    batch_size: int = 128
    lr: float = 1e-3
    w_state: float = 1.0
    w_risk: float = 3.0
    w_sparsity: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(log_dir: str, thr: FailureThresholds, cfg: TrainConfig):
    pos, t = load_swarm_positions(log_dir)
    vel = finite_difference_velocity(pos, t)

    ds = SwarmFailureDataset(pos, vel, t, window=cfg.window, horizon=cfg.horizon, thr=thr)
    N = ds.N
    node_in_dim = cfg.window * ds.node_feat_dim
    rel_dim = ds.rel_feat_dim

    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = GraphFailureNRI(N, node_in_dim, rel_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def run_epoch(loader, train: bool):
        model.train(train)
        total = total_state = total_risk = total_sparse = 0.0
        count = 0

        for hist, rel, target, labels in loader:
            hist = hist.to(cfg.device)
            rel = rel.to(cfg.device)
            target = target.to(cfg.device)
            labels = labels.to(cfg.device)

            if train:
                opt.zero_grad()

            pred_pos, risk, adj = model(hist, rel)

            loss_state = F.mse_loss(pred_pos, target)
            loss_risk = F.binary_cross_entropy(risk, labels)
            loss_sparse = torch.mean(torch.abs(adj))
            loss = cfg.w_state * loss_state + cfg.w_risk * loss_risk + cfg.w_sparsity * loss_sparse

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            bs = hist.size(0)
            total += loss.item() * bs
            total_state += loss_state.item() * bs
            total_risk += loss_risk.item() * bs
            total_sparse += loss_sparse.item() * bs
            count += bs

        denom = max(count, 1)
        return total / denom, total_state / denom, total_risk / denom, total_sparse / denom

    history = {"train": [], "val": []}
    for ep in range(cfg.epochs):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        history["train"].append(tr)
        history["val"].append(va)
        if ep % 10 == 0 or ep == cfg.epochs - 1:
            print(f"Epoch {ep:03d}/{cfg.epochs} | "
                  f"train loss={tr[0]:.4f} (state={tr[1]:.4f}, risk={tr[2]:.4f}) | "
                  f"val loss={va[0]:.4f} (state={va[1]:.4f}, risk={va[2]:.4f})")

    return model, ds, history


@torch.no_grad()
def mean_adjacency(model: GraphFailureNRI, ds: SwarmFailureDataset, cfg: TrainConfig, max_batches: int = 50):
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    model.eval()

    A_sum = None
    count = 0
    for b, (hist, rel, _, _) in enumerate(loader):
        hist = hist.to(cfg.device)
        rel = rel.to(cfg.device)
        _, _, adj = model(hist, rel)

        A = adj.mean(dim=0).detach().cpu().numpy()
        A_sum = A if A_sum is None else (A_sum + A)
        count += 1
        if b + 1 >= max_batches:
            break

    return A_sum / max(count, 1)


@torch.no_grad()
def mean_adjacency_prefail_vs_normal(model: GraphFailureNRI, ds: SwarmFailureDataset, cfg: TrainConfig, label_idx: int = 1):
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    model.eval()

    A_fail_sum = None
    A_norm_sum = None
    c_fail = 0
    c_norm = 0

    for hist, rel, _, labels in loader:
        hist = hist.to(cfg.device)
        rel = rel.to(cfg.device)
        labels = labels.to(cfg.device)

        _, _, adj = model(hist, rel)
        is_fail = (labels[..., label_idx] > 0.5).any(dim=1)  # (B,)
        is_norm = ~is_fail

        if is_fail.any():
            A = adj[is_fail].mean(dim=0).detach().cpu().numpy()
            A_fail_sum = A if A_fail_sum is None else (A_fail_sum + A)
            c_fail += 1

        if is_norm.any():
            A = adj[is_norm].mean(dim=0).detach().cpu().numpy()
            A_norm_sum = A if A_norm_sum is None else (A_norm_sum + A)
            c_norm += 1

    A_fail = A_fail_sum / max(c_fail, 1) if A_fail_sum is not None else None
    A_norm = A_norm_sum / max(c_norm, 1) if A_norm_sum is not None else None
    return A_fail, A_norm


@torch.no_grad()
def edge_ablation_attribution(
    model: GraphFailureNRI,
    sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: TrainConfig,
    label_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    hist, rel, _, _ = sample
    hist = hist.unsqueeze(0).to(cfg.device)
    rel = rel.unsqueeze(0).to(cfg.device)

    model.eval()
    _, base_risk_full, adj = model(hist, rel)
    base_risk = base_risk_full[0, :, label_idx].detach().cpu().numpy()

    N = base_risk.shape[0]
    delta = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            adj2 = adj.clone()
            adj2[0, i, j] = 0.0
            _, risk2, _ = model(hist, rel, adj_override=adj2)
            r2 = risk2[0, i, label_idx].item()
            delta[i, j] = base_risk[i] - r2

    return base_risk, delta


def print_top_edges(delta: np.ndarray, topk: int = 10, label: str = "collision"):
    N = delta.shape[0]
    edges = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            edges.append((delta[i, j], i, j))
    edges.sort(reverse=True, key=lambda x: x[0])

    print(f"\nTop {topk} edges responsables (ablation) pour {label}:")
    for k in range(min(topk, len(edges))):
        d, i, j = edges[k]
        print(f"  {k+1:02d}. edge {j} -> {i}  | Δrisk={d:+.4f}")


# ==============================================================================
# 5) PLOTS - une seule figure, tous les graphes dedans
# ==============================================================================
def _plot_training_on_ax(ax, history: Dict[str, List[Tuple[float, float, float, float]]]):
    tr = np.array(history["train"])
    va = np.array(history["val"])

    ax.plot(tr[:, 0], label="train total")
    ax.plot(va[:, 0], label="val total")
    ax.plot(tr[:, 1], label="train state")
    ax.plot(tr[:, 2], label="train risk")
    ax.set_title("Training curves")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()


def _plot_matrix_on_ax(fig, ax, M: Optional[np.ndarray], title: str):
    ax.set_title(title)
    ax.set_xlabel("SOURCE j")
    ax.set_ylabel("TARGET i")

    if M is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    im = ax.imshow(M, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def run_full_analysis(log_dir: str):
    thr = FailureThresholds(
        z_crash=0.10,
        d_collision=0.30,
        v_low=0.20,
        v_drop=0.60,
    )

    cfg = TrainConfig(
        window=10,
        horizon=20,
        hidden=128,
        dropout=0.1,
        epochs=120,
        batch_size=128,
        lr=1e-3,
        w_state=1.0,
        w_risk=3.0,
        w_sparsity=1e-4,
    )

    model, ds, history = train_model(log_dir, thr=thr, cfg=cfg)

    A_mean = mean_adjacency(model, ds, cfg)
    A_fail, A_norm = mean_adjacency_prefail_vs_normal(model, ds, cfg, label_idx=1)  # collision
    A_diff = (A_fail - A_norm) if (A_fail is not None and A_norm is not None) else None

    # trouver un sample "fail" (collision future)
    fail_idx = None
    for k in range(len(ds)):
        _, _, _, labels = ds[k]
        if (labels[:, 1] > 0.5).any():
            fail_idx = k
            break

    base_risk = None
    delta = None
    if fail_idx is not None:
        sample = ds[fail_idx]
        base_risk, delta = edge_ablation_attribution(model, sample, cfg, label_idx=1)
        print("\nBase risk collision par drone:", np.round(base_risk, 3))
        print_top_edges(delta, topk=12, label="collision")
    else:
        print("\n[INFO] Aucun exemple avec collision future détecté selon tes seuils.")
        print("=> augmente horizon, augmente d_collision, ou vérifie tes logs.")

    # === Une seule figure, tous les plots ensemble ===
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    _plot_training_on_ax(axs[0, 0], history)
    _plot_matrix_on_ax(fig, axs[0, 1], A_mean, "Adjacency moyen (global)")
    _plot_matrix_on_ax(fig, axs[0, 2], A_diff, "Diff adjacency (pre-fail - normal)")

    _plot_matrix_on_ax(fig, axs[1, 0], A_fail, "Adjacency pre-failure (collision)")
    _plot_matrix_on_ax(fig, axs[1, 1], A_norm, "Adjacency normal")
    _plot_matrix_on_ax(fig, axs[1, 2], delta, "Attribution ablation Δrisk (collision)")

    # info texte sur la figure
    if base_risk is not None:
        txt = "Base risk collision: " + ", ".join([f"d{i}={base_risk[i]:.2f}" for i in range(len(base_risk))])
        fig.suptitle(txt, fontsize=10)

    plt.show()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(here, "../logs")
    run_full_analysis(log_dir)
