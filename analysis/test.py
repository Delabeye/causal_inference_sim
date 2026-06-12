import pandas as pd
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import argparse
import json

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_torch_threads(n: int) -> None:
    """
    PyTorch peut être TRÈS lent sur des tenseurs minuscules si beaucoup de threads CPU sont activés.
    Mettre n=1 (ou 2) est souvent 10x-1000x plus rapide dans ce type de pipeline NRI.
    """
    n = int(max(1, n))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(n)
    except Exception:
        pass

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def softmax_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + eps)


def robust_zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Robust z-score using median and MAD."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + eps
    return 0.6745 * (x - med) / mad


@dataclass
class SwarmLogs:
    times: np.ndarray                 # [T]
    drone_names: List[str]            # [N]
    data: Dict[str, List[np.ndarray]]       # key -> nb_run*[T, N, dim] or [T, N]
    dt: float


def _find_drone_logs(log_dir: str) -> List[str]:
    paths = []
    for fn in os.listdir(log_dir):
        if fn.startswith("run_") and fn.endswith(".csv"):
            paths.append(os.path.join(log_dir, fn))
    return sorted(paths)


def load_swarm_logs(log_dir: str, downsample: int = 1) -> SwarmLogs:
    """
    Load drone logs, align by time, return structured arrays.

    Expected columns (from entities/uav.py):
      time, gt_x,gt_y,gt_z, gt_vx,gt_vy,gt_vz, meas_x,meas_y,meas_z,
      gnss_error_mag, wind_x,wind_y,wind_z, wind_mag, rep_force_mag,
      nearest_neighbor_dist, target_x,target_y,target_z, tracking_error_mag, collision_flag
    """
    paths = _find_drone_logs(log_dir)
    if not paths:
        raise FileNotFoundError(
            f"Aucun fichier logs 'run_*.csv' trouvé dans: {log_dir}"
        )
    
    # Changement dans la lecture des logs, regroupement des logs dans un dictionnaire pour un même drone
    drone_groups = {}
    # drone_groups : dict[drone_x, df] de taille [nb drones, nb run]
    # On fait en sorte que chaque run soit bien l'index des valeurs du dictionnaire
    # Par construction cela ne devrait pas causer de problème
    # E.G : dict["drone_0"]["1"] -> run 1

    for p in paths:
        brut_name = os.path.splitext(os.path.basename(p))[0]
        match = re.search(r'(drone_\d+)', brut_name)
        # Je récupère 'drone_x' de la run_y avec x, y croissants
        drone_id = match.group(1) if match else brut_name

        df = pd.read_csv(p)
        # Clean and basic checks
        if "time" not in df.columns:
            raise ValueError(f"Colonne 'time' absente dans {p}. Colonnes: {list(df.columns)}")
        df = df.copy()
        df.sort_values("time", inplace=True)
        if downsample > 1:
            df = df.iloc[::downsample].reset_index(drop=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        if drone_id not in drone_groups:
            drone_groups[drone_id] = []
        drone_groups[drone_id].append(df)
        
        
    # Permet de trier les indices des drones dans l'ordre croissant
    names = sorted(drone_groups.keys(), key= lambda x: int(re.search(r'\d+', x).group()))
        

    # Align by time (inner join on rounded time)
    # times are already rounded to 0.001 in logs; still, we align robustly.
    run_drone_0_list = drone_groups["drone_0"]
    df_run = run_drone_0_list[0]
    base = df_run[["time"]].copy()
    base["time"] = base["time"].round(3)

    # On récupère l'intersection de tous les temps de tous les CSV de chaque drone
    aligned_time = base["time"].tolist()
    for drone_id in names:
        for df in drone_groups[drone_id]:
            ensemble_temps = list(set(aligned_time) & set(df["time"].round(3)))
            aligned_time = ensemble_temps

    aligned_time = sorted(aligned_time) # list
    aligned_time = np.array(aligned_time, dtype=np.float32)
    
    # On traite les lignes en trop pour que chaque CSV aient les mêmes timestamps
    # Pour l'instant je pars du principe que tous les temps sont alignés en gros il faut finir
    # Le dfs2 c'est une run avec tous les drones qui ont été réindexés

    
    if len(aligned_time) >= 2:
        dt = float(np.median(np.diff(aligned_time)))
    else:
        dt = 0.1


    nb_runs = len(drone_groups[names[0]])

    
    def stack_cols(cols : List[str]) -> List[np.ndarray]: 
        all_runs = []

        for r in range(nb_runs):
            arrs = []

            for drone_id in names:
                df = drone_groups[drone_id][r]
                arr = df[cols].values.astype(np.float32)
                arrs.append(arr)

            matrix_run = np.stack(arrs, axis = 1)
            all_runs.append(matrix_run)
        
        # List of nb_runs of matrixes [T, N, len(cols)]
        return(all_runs)
    

    def stack_col(col: str) -> List[np.ndarray]:
        all_runs = []

        for r in range(nb_runs):
            arrs = []

            for drone_id in names:
                df = drone_groups[drone_id][r]
                arr = df[col].values.astype(np.float32)
                arrs.append(arr)

            matrix_run = np.stack(arrs, axis = 1)
            all_runs.append(matrix_run)
        
        # List of nb_runs of matrixes [T, N]
        return(all_runs)
    
    data: dict[str, List[np.ndarray]] = {}
    # La liste représente le nombre de run et l'array possède le nombre de drone
    # Donc ici data est devenu un dictionnaire avec les features en clés

    data["gt_pos"] = stack_cols(["gt_x", "gt_y", "gt_z"])
    data["gt_vel"] = stack_cols(["gt_vx", "gt_vy", "gt_vz"])
    data["meas_pos"] = stack_cols(["meas_x", "meas_y", "meas_z"])
    data["wind"] = stack_cols(["wind_x", "wind_y", "wind_z"])
    data["wind_mag"] = stack_col("wind_mag")
    data["gnss_error_mag"] = stack_col("gnss_error_mag")
    data["rep_force_mag"] = stack_col("rep_force_mag")
    data["nearest_neighbor_dist"] = stack_col("nearest_neighbor_dist")
    data["target_pos"] = stack_cols(["target_x", "target_y", "target_z"])
    data["tracking_error_mag"] = stack_col("tracking_error_mag")
    data["collision_flag_log"] = stack_col("collision_flag")  # from pybullet contact points

    return SwarmLogs(times=aligned_time, drone_names=names, data=data, dt=dt)


# ---------------------------
# Failure detection (labels)
# ---------------------------

@dataclass
class FailureLabels:
    # Each is [T, N] (binary)
    collision: np.ndarray
    formation_loss: np.ndarray
    gnss_degradation: np.ndarray
    wind_loss: np.ndarray
    suboptimal_traj: np.ndarray

    # aux signals
    formation_error: np.ndarray      # [T, N]
    min_pairwise_dist: np.ndarray    # [T, N]
    speed: np.ndarray                # [T, N]


def compute_pairwise_dists(pos: np.ndarray) -> np.ndarray:
    """pos: [T,N,3] -> dists: [T,N,N]"""
    T, N, _ = pos.shape
    d = pos[:, :, None, :] - pos[:, None, :, :]
    d2 = np.sum(d * d, axis=-1)
    dist = np.sqrt(np.maximum(d2, 0.0))
    return dist


def detect_failures(
    logs: SwarmLogs,
    run_idx: int,
    leader_index: int = 0,
    collision_dist: float = 0.6,
    formation_thresh: float = 1.0,
    gnss_quantile: float = 0.95,
    wind_quantile: float = 0.95,
    suboptimal_quantile: float = 0.95,
    min_persist_steps: int = 5,
) -> FailureLabels:
    """
    Heuristiques de détection:
    - collision: min distance inter-drones < collision_dist
    - formation_loss: écart à la formation de référence > formation_thresh
      (référence = offsets initiaux par rapport au leader)
    - gnss_degradation: gnss_error_mag > quantile (par drone)
    - wind_loss: wind_mag > quantile ET dérivée de tracking_error positive
    - suboptimal_traj: tracking_error_mag > quantile (persistant)
    """
    pos = logs.data["gt_pos"][run_idx]  # [T,N,3]
    vel = logs.data["gt_vel"][run_idx]
    wind_mag = logs.data["wind_mag"][run_idx]
    gnss_err = logs.data["gnss_error_mag"][run_idx]
    track_err = logs.data["tracking_error_mag"][run_idx]

    T, N, _ = pos.shape

    # Pairwise distances
    dists = compute_pairwise_dists(pos)  # [T,N,N]
    # avoid self by setting diag to +inf
    for t in range(T):
        np.fill_diagonal(dists[t], np.inf)
    min_dist = np.min(dists, axis=-1)  # [T,N]

    collision = (min_dist < collision_dist).astype(np.int32)

    # Formation error relative to leader, reference = initial offsets
    leader_pos0 = pos[0, leader_index].copy()
    offsets0 = pos[0] - leader_pos0[None, :]  # [N,3]
    rel = pos - pos[:, leader_index:leader_index+1, :]  # [T,N,3]
    formation_error = np.linalg.norm(rel - offsets0[None, :, :], axis=-1)  # [T,N]
    # leader is always 0 formation error by definition
    formation_error[:, leader_index] = 0.0

    formation_loss = (formation_error > formation_thresh).astype(np.int32)

    # GNSS degradation per drone quantile
    gnss_thr = np.quantile(gnss_err, gnss_quantile, axis=0)  # [N]
    gnss_degradation = (gnss_err > gnss_thr[None, :]).astype(np.int32)

    # Wind loss: high wind + tracking error increasing sharply
    wind_thr = np.quantile(wind_mag, wind_quantile, axis=0)
    d_track = np.zeros_like(track_err)
    d_track[1:] = (track_err[1:] - track_err[:-1]) / max(logs.dt, 1e-6)
    wind_loss = ((wind_mag > wind_thr[None, :]) & (d_track > 0.5)).astype(np.int32)  # 0.5 m/s as default slope

    # Suboptimal trajectory: high tracking error persistent
    sub_thr = np.quantile(track_err, suboptimal_quantile, axis=0)
    suboptimal = (track_err > sub_thr[None, :]).astype(np.int32)
    # persistence: require min_persist_steps consecutive 1s
    if min_persist_steps > 1:
        suboptimal_p = np.zeros_like(suboptimal)
        for i in range(N):
            run = 0
            for t in range(T):
                if suboptimal[t, i] == 1:
                    run += 1
                else:
                    run = 0
                if run >= min_persist_steps:
                    suboptimal_p[t, i] = 1
        suboptimal = suboptimal_p

    speed = np.linalg.norm(vel, axis=-1)  # [T,N]

    return FailureLabels(
        collision=collision,
        formation_loss=formation_loss,
        gnss_degradation=gnss_degradation,
        wind_loss=wind_loss,
        suboptimal_traj=suboptimal,
        formation_error=formation_error.astype(np.float32),
        min_pairwise_dist=min_dist.astype(np.float32),
        speed=speed.astype(np.float32),
    )






@dataclass
class SequenceDataset:
    s: List[np.ndarray]  # List of matrix nb_run * [T,N,S]
    u: List[np.ndarray]  # List of matrix [T,N,U]
    times: np.ndarray  # [T]

    def sample_windows(self, seq_len: int, n_samples: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:

        s_w, u_w = [], []
        nb_runs = len(self.s)

        for _ in range(n_samples):
            # On tire une run au hasard
            run_idx = rng.randint(0, nb_runs)
            T_run = self.s[run_idx].shape[0]

            if T_run <= seq_len:
                continue

            start = rng.randint(0, T_run - seq_len)

            s_w.append(self.s[run_idx][start:start+seq_len])
            u_w.append(self.u[run_idx][start:start+seq_len])
        
        return np.stack(s_w), np.stack(u_w)
    

def build_sequence_dataset(logs: SwarmLogs, use_measured: bool = False) -> SequenceDataset:

    s_list, u_list = [], []

    for nb_run in range(len(logs.data["meas_pos"])):

        if use_measured:
            pos = logs.data["meas_pos"][nb_run]
            vel = np.zeros_like(pos)
            vel[1:] = (pos[1:] - pos[:-1]) / max(logs.dt, 1e-6)
        else:
            pos = logs.data["gt_pos"][nb_run]
            vel = logs.data["gt_vel"][nb_run]
        
        
        s = np.concatenate([pos, vel], axis = -1)
        s_list.append(s.astype(np.float32))
        wind = logs.data["wind"][nb_run]
        gnss = logs.data["gnss_error_mag"][nb_run][..., None]
        u = np.concatenate([wind, gnss], axis = -1)
        u_list.append(u.astype(np.float32))

    return SequenceDataset(s = s_list, u = u_list, times = logs.times.astype(np.float32))




# ---------------------------
# NRI model (PyTorch)
# ---------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def gumbel_softmax_sample(logits, tau=1.0, eps=1e-10):
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    y = logits + g
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1.0, hard=False):
    y = gumbel_softmax_sample(logits, tau=tau)
    if not hard:
        return y
    # straight-through
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y = (y_hard - y).detach() + y
    return y


def build_offdiag_indices(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return sender/receiver indices for all directed edges excluding self edges.
    """
    offdiag = np.ones((n, n)) - np.eye(n)
    receivers, senders = np.where(offdiag)
    receivers = torch.tensor(receivers, dtype=torch.long, device=device)
    senders = torch.tensor(senders, dtype=torch.long, device=device)
    return senders, receivers  # E each


class NRIEncoder(nn.Module):
    """
    Classic NRI-ish encoder: encode node trajectories, then infer edge type logits.

    Input x: [B, L, N, D_in]
    """
    def __init__(self, n_nodes: int, d_in: int, hidden: int, n_edge_types: int, dropout: float = 0.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_in = d_in
        self.hidden = hidden
        self.n_edge_types = n_edge_types
        # Node embedding from flattened temporal window
        self.node_mlp = MLP(in_dim=d_in, hidden_dim=hidden, out_dim=hidden, dropout=dropout)
        # Edge MLP from concatenated sender/receiver node embeddings + difference
        self.edge_mlp = MLP(in_dim=3*hidden, hidden_dim=hidden, out_dim=n_edge_types, dropout=dropout)

    def forward(self, x_flat: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor):
        """
        x_flat: [B, N, D_in] (already aggregated across time, e.g. mean/flatten+MLP)
        """
        B, N, D = x_flat.shape
        assert N == self.n_nodes

        h = self.node_mlp(x_flat)  # [B,N,H]
        h_send = h[:, senders, :]  # [B,E,H]
        h_recv = h[:, receivers, :]
        h_diff = h_send - h_recv
        edge_in = torch.cat([h_send, h_recv, h_diff], dim=-1)  # [B,E,3H]
        logits = self.edge_mlp(edge_in)  # [B,E,K]
        return logits


class NRIDecoder(nn.Module):
    """
    Message passing decoder for next-step prediction.

    Given current state s_t, exog u_t, and edges z_ij (soft one-hot over K),
    predict delta (residual) to produce s_{t+1}.
    """
    def __init__(self, n_nodes: int, state_dim: int, exog_dim: int, hidden: int, n_edge_types: int, dropout: float = 0.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.state_dim = state_dim
        self.exog_dim = exog_dim
        self.hidden = hidden
        self.n_edge_types = n_edge_types

        # Per-edge-type message networks
        self.msg_mlps = nn.ModuleList([
            MLP(in_dim=2*state_dim, hidden_dim=hidden, out_dim=hidden, dropout=dropout)
            for _ in range(n_edge_types)
        ])
        # Node update
        self.node_mlp = MLP(in_dim=state_dim + exog_dim + hidden, hidden_dim=hidden, out_dim=state_dim, dropout=dropout)

    def forward(self, s: torch.Tensor, u: torch.Tensor, z: torch.Tensor,
                senders: torch.Tensor, receivers: torch.Tensor) -> torch.Tensor:
        """
        s: [B,N,S]
        u: [B,N,U]
        z: [B,E,K] soft edges over K types
        -> s_next_pred: [B,N,S]
        """
        B, N, S = s.shape
        E = senders.shape[0]

        s_send = s[:, senders, :]  # [B,E,S]
        s_recv = s[:, receivers, :]  # [B,E,S]
        edge_feat = torch.cat([s_send, s_recv], dim=-1)  # [B,E,2S]

        # Compute messages for each edge type, weighted by z
        msg_all = 0.0
        for k in range(self.n_edge_types):
            msg_k = self.msg_mlps[k](edge_feat)  # [B,E,H]
            w = z[..., k:k+1]  # [B,E,1]
            msg_all = msg_all + w * msg_k

        # Aggregate messages per receiver
        agg = torch.zeros((B, N, self.hidden), device=s.device, dtype=s.dtype)  # [B,N,H]
        agg.index_add_(1, receivers, msg_all)  # sum over incoming edges

        node_in = torch.cat([s, u, agg], dim=-1)  # [B,N,S+U+H]
        delta = self.node_mlp(node_in)  # [B,N,S]
        s_next = s + delta
        return s_next


class NRIModel(nn.Module):
    def __init__(self, n_nodes: int, state_dim: int, exog_dim: int,
                 hidden: int = 128, n_edge_types: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.state_dim = state_dim
        self.exog_dim = exog_dim
        self.hidden = hidden
        self.n_edge_types = n_edge_types

        # Encoder takes aggregated temporal node features; we pass mean over window of (pos,vel)
        self.encoder = NRIEncoder(n_nodes=n_nodes, d_in=state_dim, hidden=hidden, n_edge_types=n_edge_types, dropout=dropout)
        self.decoder = NRIDecoder(n_nodes=n_nodes, state_dim=state_dim, exog_dim=exog_dim, hidden=hidden, n_edge_types=n_edge_types, dropout=dropout)

    def infer_edges(self, s_window: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor,
                    tau: float = 0.5, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        s_window: [B,L,N,S] -> aggregate -> [B,N,S]
        returns:
          z: [B,E,K] soft/hard one-hot
          logits: [B,E,K]
        """
        # simple aggregation: mean over time
        s_mean = torch.mean(s_window, dim=1)  # [B,N,S]
        logits = self.encoder(s_mean, senders, receivers)
        z = gumbel_softmax(logits, tau=tau, hard=hard)
        return z, logits


    def forward(self, s_window: torch.Tensor, u_window: torch.Tensor,
                senders: torch.Tensor, receivers: torch.Tensor,
                tau: float = 0.5, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        s_window: [B,L,N,S]
        u_window: [B,L,N,U]
        Predict next steps within window: for t in [0..L-2], predict s_{t+1} from s_t,u_t,edges
        Returns:
          s_pred: [B,L-1,N,S]
          z: [B,E,K]
          logits: [B,E,K]
        """
        z, logits = self.infer_edges(s_window, senders, receivers, tau=tau, hard=hard)
        B, L, N, S = s_window.shape
        preds = []
        s_t = s_window[:, 0, :, :]
        for t in range(L-1):
            u_t = u_window[:, t, :, :]
            s_next = self.decoder(s_t, u_t, z, senders, receivers)
            preds.append(s_next)
            s_t = s_window[:, t+1, :, :]  # teacher forcing
        s_pred = torch.stack(preds, dim=1)  # [B,L-1,N,S]
        return s_pred, z, logits


@dataclass
class NRIResults:
    edge_probs: np.ndarray
    edge_type_probs: np.ndarray
    signed_influence: np.ndarray
    nri_loss_curve: List[float]

def infer_edges_multirun(model, dataset_s_list, senders, receivers, N, n_edge_types, device):
    model.eval()
    all_edge_probs = [] # Va stocker les matrices NxN de chaque run
    all_edge_type_probs = []
    K = n_edge_types

    with torch.no_grad():
        for run_s in dataset_s_list:
            # [1, T, N, S]
            s_tensor = torch.tensor(run_s).unsqueeze(0).to(device)

            s_mean = torch.mean(s_tensor, dim = 1) # [1, N, S]
            
            # L'encodeur recrache la "pile de câbles"
            logits = model.encoder(s_mean, senders, receivers)
            probs = torch.softmax(logits, dim=-1) # Forme: [1, E, K]
            
            # On extrait z_mean (les probabilités de cette run)
            z_mean = probs[0].cpu().numpy() # Forme: [E, K]
            
            # --- LE BLOC TRADUCTEUR QUE TU VIENS DE MONTRER ---
            edge_probs_run = np.zeros((N, N), dtype=np.float32)
            edge_type_probs_run = np.zeros((n_edge_types, N, N), dtype=np.float32)
            E = len(senders)

            send_np = senders.cpu().numpy().astype(int)
            recv_np = receivers.cpu().numpy().astype(int)
            
            for e in range(E):
                j = send_np[e]
                i = recv_np[e]
                for k in range(K):
                    edge_type_probs_run[k, i, j] = float(z_mean[e, k])
                # Probabilité d'interaction = 1 - Probabilité(Type 0)
                edge_probs_run[i, j] = float(1.0 - z_mean[e, 0])
            # ---------------------------------------------------
            
            all_edge_probs.append(edge_probs_run)
            all_edge_type_probs.append(edge_type_probs_run)

    # On retourne la matrice NxN moyenne de toutes les runs
    return np.mean(all_edge_probs, axis=0), np.mean(all_edge_type_probs, axis=0)

def train_nri(
    dataset: SequenceDataset,
    n_nodes: int,
    seq_len: int = 30,
    n_edge_types: int = 2,
    hidden: int = 128,
    dropout: float = 0.0,
    batch_size: int = 64,
    nri_steps: int = 1500,
    lr: float = 3e-4,
    tau_start: float = 1.0,
    tau_end: float = 0.5,
    tau_anneal_steps: int = 1500,
    max_train_windows: int = 5000,
    device: str = "cpu",
    torch_threads: int = 1,
    seed: int = 0,
    verbose_every: int = 100,
) -> NRIResults:
    set_seed(seed)
    dev = torch.device(device)
    set_torch_threads(torch_threads)

    S = dataset.s[0].shape[-1]
    U = dataset.u[0].shape[-1]

    model = NRIModel(n_nodes = n_nodes, state_dim=S, exog_dim=U, hidden=hidden,
                     n_edge_types=n_edge_types, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    senders, receivers = build_offdiag_indices(n_nodes, dev)

    s_concat = np.concatenate(dataset.s, axis = 0) # -> [Somme_T, N, S]
    u_concat = np.concatenate(dataset.u, axis = 0)

    s_flat = s_concat.reshape(-1, S)
    u_flat = u_concat.reshape(-1, U)

    s_scaler = StandardScaler().fit(s_flat)
    u_scaler = StandardScaler().fit(u_flat)

    s_std_list = [
        s_scaler.transform(run.reshape(-1, S)).reshape(run.shape).astype(np.float32)
        for run in dataset.s
    ]
    u_std_list = [
        u_scaler.transform(run.reshape(-1, U)).reshape(run.shape).astype(np.float32)
        for run in dataset.u
    ]

    ds_std = SequenceDataset(s=s_std_list, u=u_std_list, times=dataset.times)

    rng = np.random.RandomState(seed)
    loss_curve: List[float] = []

    n_windows = min(max_train_windows, max(1000, batch_size * 10))

    s_w_all, u_w_all = ds_std.sample_windows(seq_len=seq_len + 1, n_samples=n_windows, rng = rng)

    s_w_all_t = torch.from_numpy(s_w_all).to(dev)
    u_w_all_t = torch.from_numpy(u_w_all).to(dev)

    model.train()
    for step in range(1, nri_steps + 1):
        # anneal tau
        if step <= tau_anneal_steps:
            tau = tau_start + (tau_end - tau_start) * (step / tau_anneal_steps)
        else:
            tau = tau_end

        idx = rng.randint(0, n_windows, size=batch_size)
        s_win = s_w_all_t[idx]  # [B,L+1,N,S]
        u_win = u_w_all_t[idx]

        # Contexte pour inférer le graphe latent
        s_ctx = s_win[:, :seq_len, :, :]   # [B,L,N,S]
        # Exogènes (ici utilisés uniquement au temps t de la transition)
        s_t = s_win[:, seq_len - 1, :, :]  # [B,N,S]
        u_t = u_win[:, seq_len - 1, :, :]  # [B,N,U]
        target = s_win[:, seq_len, :, :]   # [B,N,S]

        z, logits = model.infer_edges(s_ctx, senders, receivers, tau=tau, hard=False)
        s_pred = model.decoder(s_t, u_t, z, senders, receivers)

        loss = F.mse_loss(s_pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        loss_curve.append(float(loss.item()))
        if verbose_every and (step % verbose_every == 0 or step == 1):
            print(f"[NRI] step={step:5d}/{nri_steps}  tau={tau:.3f}  mse={loss.item():.6f}")

    # infer edge probabilities by running encoder over many windows
    model.eval()

    edge_probs, edge_type_probs = infer_edges_multirun(model, ds_std.s, senders, receivers,  N = n_nodes, device = device, n_edge_types=n_edge_types)

    # Signed influence heuristic from data: if i tends to accelerate away from j -> repulsive
    signed = estimate_signed_influence(s_concat, edge_probs, leader_index=0)

    return NRIResults(
        edge_probs=edge_probs,
        edge_type_probs=edge_type_probs,
        signed_influence=signed,
        nri_loss_curve=loss_curve,
    )

def estimate_signed_influence(states: np.ndarray, edge_probs: np.ndarray, leader_index: int = 0) -> np.ndarray:
    """
    Heuristique de "sens" (repulsif / attractif) basée sur la dynamique :
    - states: [T,N,6] = pos(3)+vel(3)
    - on approxime a_i ~ dv_i/dt et on regarde le signe moyen de dot(a_i, (p_i - p_j))
      >0 : accélère en s'éloignant de j (repulsif)
      <0 : accélère vers j (attractif)
    """
    pos = states[..., :3]
    vel = states[..., 3:6]
    T, N, _ = pos.shape

    # finite difference acceleration (dt not known here; sign doesn't need scale)
    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]

    signed = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rel = pos[:, i, :] - pos[:, j, :]  # [T,3]
            dot = np.sum(acc[:, i, :] * rel, axis=-1)  # [T]
            denom = np.sum(rel * rel, axis=-1) + 1e-6
            score = np.nanmean(dot / denom)
            sign = 1.0 if score > 0 else -1.0
            signed[i, j] = sign * float(edge_probs[i, j])
    return signed







@dataclass
class EventModelOutputs:
    # predicted probabilities per event type [T,N]
    proba: Dict[str, np.ndarray]
    # per-event instance explanations
    event_causes: List[Dict]
    # global average cause probs per event type and per drone
    avg_causes: Dict[str, Dict[str, List[float]]]


def build_interaction_pressure(pos: np.ndarray, edge_probs: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Scalar feature per drone representing "pressure" from neighbors weighted by inferred edges.
    pressure_i = sum_j edge_prob[i<-j] / (dist_ij + eps)
    """
    dists = compute_pairwise_dists(pos)  # [T,N,N]
    T, N, _ = dists.shape
    for t in range(T):
        np.fill_diagonal(dists[t], np.inf)
    inv = 1.0 / (dists + eps)
    w = edge_probs[None, :, :]  # [1,N,N] receiver i, sender j
    pressure = np.sum(w * inv, axis=-1)  # [T,N]
    return pressure.astype(np.float32)


def fit_event_models(
    logs: SwarmLogs,
    labels: List[FailureLabels],
    edge_probs: np.ndarray,
    horizon_lag: int = 1,
    max_events_to_explain: int = 2000,
    seed: int = 0,
) -> EventModelOutputs:
    """
    Fit simple per-event logistic regression models to predict event(t) from features(t-lag).
    Then convert coefficient contributions into per-event root-cause probability distribution.
    """
    rng = np.random.RandomState(seed)
    n_runs = len(logs.data["gt_pos"])
    lag = horizon_lag

    all_X_lag = []

    events_all = {
        "collision": [],
        "formation_loss": [],
        "gnss_degradation": [],
        "wind_loss": [],
        "suboptimal_traj": []
    }

    for run in range(n_runs):

        pos = logs.data["gt_pos"][run]
        wind = logs.data["wind_mag"][run]
        gnss = logs.data["gnss_error_mag"][run]
        repf = logs.data["rep_force_mag"][run]
        track = logs.data["tracking_error_mag"][run]

        label = labels[run]
        formerr = label.formation_error
        mindist = label.min_pairwise_dist

        interaction_pressure = build_interaction_pressure(pos, edge_probs)  # [T,N]

        # Features at t-lag
        T, N = wind.shape
        # drop first lag timesteps
        idx_t = np.arange(lag, T)

        events_all["collision"].append(label.collision[idx_t])
        events_all["formation_loss"].append(label.formation_loss[idx_t])
        events_all["gnss_degradation"].append(label.gnss_degradation[idx_t])
        events_all["wind_loss"].append(label.wind_loss[idx_t])
        events_all["suboptimal_traj"].append(label.suboptimal_traj[idx_t])
        
        # lagged features (t-lag)
        X_lag = np.stack([
        wind[idx_t - lag],
        gnss[idx_t - lag],
        repf[idx_t - lag],
        track[idx_t - lag],
        formerr[idx_t - lag],
        mindist[idx_t - lag],
        interaction_pressure[idx_t - lag],
        ], axis=-1)

        all_X_lag.append(X_lag)

        
        # Flatten
        X_lag_global = np.concatenate(all_X_lag, axis=0) # [n_runs*(T-lag), N, F]
        Xf_global = X_lag_global.reshape(-1, X_lag_global.shape[-1]) # [n_runs*(T-lag)*N, F]
        scaler = StandardScaler().fit(Xf_global)
        Xfs = scaler.transform(Xf_global)

        outputs_proba: Dict[str, List[np.ndarray]] = {} # Liste de prédictions par run
        event_causes: List[Dict] = []
        avg_causes: Dict[str, Dict[str, List[float]]] = {}

    

    # base feature matrix per (t,i)
    # order matters for grouping later
    feature_names = [
        "wind_mag",
        "gnss_error_mag",
        "rep_force_mag",
        "tracking_error_mag",
        "formation_error",
        "min_pairwise_dist",
        "interaction_pressure",
    ]

    # Factor grouping for root-cause attribution
    groups = {
        "wind": ["wind_mag"],
        "gnss": ["gnss_error_mag"],
        "interaction": ["rep_force_mag", "min_pairwise_dist", "interaction_pressure"],
        "formation_tracking": ["tracking_error_mag", "formation_error"],
    }
    name_to_idx = {n: k for k, n in enumerate(feature_names)}

    for ev_name in events_all.keys():
        
        y = np.concatenate(events_all[ev_name], axis=0).reshape(-1).astype(int)  # [(T-lag)*N]
        # If event extremely rare, skip
        if y.sum() < 10:
            print(f"[EventModel] '{ev_name}' trop rare ({y.sum()} positives). Skipping model.")
            outputs_proba[ev_name] = np.zeros((T, N), dtype=np.float32)
            continue

        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
        clf.fit(Xfs, y)

        proba = clf.predict_proba(Xfs)[:, 1]
        proba_global = proba.reshape(n_runs, len(idx_t), N)
        outputs_proba[ev_name] = []
        # Predict proba for all times (align)
        for r in range(n_runs):
            proba_all = np.zeros((T, N), dtype=np.float32)
        
            proba_all[idx_t] = proba_global[r]
            outputs_proba[ev_name].append(proba_all)

        # Root-cause explanations for positive instances
        coef = clf.coef_.reshape(-1)  # [F]
        # Build contributions per instance in standardized space
        # contribution = coef_k * x_k
        contrib = Xfs * coef[None, :]  # [M,F]
        contrib = np.maximum(contrib, 0.0)  # only positive evidence

        pos_indices = np.where(y == 1)[0]
        if len(pos_indices) > max_events_to_explain:
            pos_indices = rng.choice(pos_indices, size=max_events_to_explain, replace=False)

        # Aggregate average cause by drone
        avg_causes[ev_name] = {g: [0.0] * N for g in groups.keys()}
        cnt_by_drone = np.zeros(N, dtype=int)

        for idx_flat in pos_indices:
            points_per_run = len(idx_t) * N
            run_idx = idx_flat // points_per_run # Car on a concat toutes les runs dans y
            reste = idx_flat % points_per_run
            t_rel = reste // N
            i = reste % N
            t = int(idx_t[t_rel])  # absolute time index

            # group scores
            g_scores = {}
            for g, names in groups.items():
                s = 0.0
                for nm in names:
                    s += float(contrib[idx_flat, name_to_idx[nm]])
                g_scores[g] = s

            # softmax to probabilities
            vec = np.array(list(g_scores.values()), dtype=np.float32)
            probs = softmax_np(vec, axis=0)
            g_probs = {g: float(probs[k]) for k, g in enumerate(g_scores.keys())}

            # store
            event_causes.append({
                "event": ev_name,
                "run_index": int(run_idx),
                "time": float(logs.times[t]),
                "t_index": int(t),
                "drone": int(i),
                "drone_name": logs.drone_names[i],
                "predicted_event_probability": float(outputs_proba[ev_name][run_idx][t, i]),
                "cause_probabilities": g_probs,
                "raw_group_scores": g_scores,
            })

            # accumulate averages
            cnt_by_drone[i] += 1
            for g in groups.keys():
                avg_causes[ev_name][g][i] += g_probs[g]

        for i in range(N):
            if cnt_by_drone[i] > 0:
                for g in groups.keys():
                    avg_causes[ev_name][g][i] /= float(cnt_by_drone[i])

        print(f"[EventModel] '{ev_name}': trained. Explained events: {len(pos_indices)}.")

    return EventModelOutputs(proba=outputs_proba, event_causes=event_causes, avg_causes=avg_causes)



# ---------------------------
# Systemic impact across the network
# ---------------------------

def compute_systemic_impact(events: np.ndarray, target_errors: np.ndarray, horizon_steps: int = 10) -> np.ndarray:
    """
    events: [T,N] binary (source events)
    target_errors: [T,N] binary (target "impact" indicator, can reuse same event type or a generic failure)
    Returns impact matrix [N,N]:
      impact[i->j] = P(target_error_j within horizon | event_i at t) - P(target_error_j)
    """
    T, N = events.shape
    base = np.mean(target_errors, axis=0)  # [N]

    impact = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        t_events = np.where(events[:, i] == 1)[0]
        if len(t_events) == 0:
            continue
        for j in range(N):
            hit = 0
            total = 0
            for t in t_events:
                t2 = min(T, t + horizon_steps + 1)
                total += 1
                if np.any(target_errors[t:t2, j] == 1):
                    hit += 1
            cond = hit / max(total, 1)
            impact[i, j] = float(cond - base[j])
    return impact


# ---------------------------
# Plotting helpers
# ---------------------------

def plot_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str,
                 xticks: List[str], yticks: List[str], out_path: str,
                 vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticks)), xticks, rotation=45, ha="right")
    plt.yticks(np.arange(len(yticks)), yticks)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_nri_graph(edge_probs: np.ndarray, names: List[str], out_path: str, thresh: float = 0.3) -> None:
    N = len(names)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(names[i])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = float(edge_probs[i, j])
            if w >= thresh:
                G.add_edge(names[j], names[i], weight=w)  # j -> i

    if G.number_of_edges() == 0:
        # still save an empty plot
        plt.figure(figsize=(6, 6))
        plt.title("NRI inferred graph (no edges above threshold)")
        nx.draw(G, with_labels=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    pos = nx.spring_layout(G, seed=0)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    # draw
    plt.figure(figsize=(7, 7))
    plt.title("NRI inferred causal influence graph (directed)")
    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, width=[2 + 4*w for w in weights], alpha=0.8)
    # edge labels
    e_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=e_labels, font_size=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss_curve(losses: List[float], out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.title("NRI training loss (MSE)")
    plt.xlabel("Training step")
    plt.ylabel("MSE")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_event_probas(times: np.ndarray, probas: Dict[str, np.ndarray], names: List[str], out_dir: str, max_events: int = 5) -> None:
    """
    Save one figure per drone with event probabilities time-series.
    """
    events = list(probas.keys())[:max_events]
    T, N = next(iter(probas.values())).shape
    for i in range(N):
        plt.figure(figsize=(10, 4))
        for ev in events:
            plt.plot(times, probas[ev][:, i], label=ev)
        plt.title(f"Predicted failure probabilities - {names[i]}")
        plt.xlabel("time [s]")
        plt.ylabel("P(event)")
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle=":")
        plt.legend(loc="upper right", ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"event_probabilities_{names[i]}.png"), dpi=200)
        plt.close()


def save_matrix_csv(mat: np.ndarray, row_names: List[str], col_names: List[str], out_path: str) -> None:
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    df.to_csv(out_path)








# ---------------------------
# Main pipeline
# ---------------------------

def run_pipeline(args: argparse.Namespace) -> Dict:
    ensure_dir(args.output_dir)

    logs = load_swarm_logs(args.log_dir, downsample=args.downsample)
    names = logs.drone_names
    n_runs = len(logs.data["meas_pos"])
    N = len(names)
    print(f"[Load] N={N} drones, T={len(logs.times)} steps, dt≈{logs.dt:.4f}s, downsample={args.downsample}")

    all_labels = [] # Création de d'une liste de FailureLabels pour chaque run

    for run_idx in range(n_runs):
        labels = detect_failures(
            logs,
            run_idx = run_idx,
            leader_index=args.leader_index,
            collision_dist=args.collision_dist,
            formation_thresh=args.formation_thresh,
            gnss_quantile=args.gnss_quantile,
            wind_quantile=args.wind_quantile,
            suboptimal_quantile=args.suboptimal_quantile,
            min_persist_steps=args.suboptimal_persist,
            )
        all_labels.append(labels)

    # Build dataset for NRI
    ds = build_sequence_dataset(logs, use_measured=args.use_measured)

    # Train NRI (or load precomputed)
    nri_res = train_nri(
        dataset=ds,
        n_nodes=N,
        seq_len=args.seq_len,
        n_edge_types=args.n_edge_types,
        hidden=args.hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        nri_steps=args.nri_steps,
        lr=args.lr,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_anneal_steps=args.tau_anneal_steps,
        max_train_windows=args.max_train_windows,
        device=args.device,
        torch_threads=args.torch_threads,
        seed=args.seed,
        verbose_every=args.verbose_every,
    )

    # Event models & root-cause explanations
    ev_out = fit_event_models(
        logs=logs,
        labels=all_labels,
        edge_probs=nri_res.edge_probs,
        horizon_lag=args.event_lag,
        max_events_to_explain=args.max_events_to_explain,
        seed=args.seed,
    )

    # Systemic impact: define a "generic failure" label as OR of events
    

    impacts = {}
    event_names = ["collision", "formation_loss", "gnss_degradation", "wind_loss", "suboptimal_traj"]
    for ev_name in event_names:
        run_impacts = []
        for r in range(n_runs):
            lab = all_labels[r]
            generic_failure = (lab.collision | lab.formation_loss | lab.gnss_degradation | lab.wind_loss | lab.suboptimal_traj).astype(np.int32)
            ev = getattr(lab, ev_name)        
            imp = compute_systemic_impact(ev, generic_failure, horizon_steps=args.impact_horizon)
            run_impacts.append(imp)

        impacts[ev_name] = np.mean(run_impacts, axis = 0)


     # ------------------ save matrices ------------------
    out = args.output_dir
    np.save(os.path.join(out, "nri_edge_probs.npy"), nri_res.edge_probs)
    np.save(os.path.join(out, "nri_edge_type_probs.npy"), nri_res.edge_type_probs)
    np.save(os.path.join(out, "nri_signed_influence.npy"), nri_res.signed_influence)

    save_matrix_csv(nri_res.edge_probs, row_names=names, col_names=names, out_path=os.path.join(out, "nri_edge_probs.csv"))
    save_matrix_csv(nri_res.signed_influence, row_names=names, col_names=names, out_path=os.path.join(out, "nri_signed_influence.csv"))

    # impacts per event
    for ev_name, mat in impacts.items():
        np.save(os.path.join(out, f"impact_{ev_name}.npy"), mat)
        save_matrix_csv(mat, row_names=names, col_names=names, out_path=os.path.join(out, f"impact_{ev_name}.csv"))

    # save event causes
    with open(os.path.join(out, "event_root_cause_explanations.json"), "w", encoding="utf-8") as f:
        json.dump(ev_out.event_causes, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out, "event_root_cause_averages.json"), "w", encoding="utf-8") as f:
        json.dump(ev_out.avg_causes, f, ensure_ascii=False, indent=2)

    
    # ------------------ figures ------------------
    plot_loss_curve(nri_res.nri_loss_curve, os.path.join(out, "nri_training_loss.png"))

    plot_heatmap(
        nri_res.edge_probs,
        title="NRI inferred interaction probability (receiver i, sender j)",
        xlabel="sender j",
        ylabel="receiver i",
        xticks=names,
        yticks=names,
        out_path=os.path.join(out, "nri_edge_probs_heatmap.png"),
        vmin=0.0,
        vmax=1.0,
    )

    plot_heatmap(
        nri_res.signed_influence,
        title="Signed influence (heuristic) = edge_prob * sign(repulsive/attractive)",
        xlabel="sender j",
        ylabel="receiver i",
        xticks=names,
        yticks=names,
        out_path=os.path.join(out, "nri_signed_influence_heatmap.png"),
        vmin=-1.0,
        vmax=1.0,
    )

    plot_nri_graph(nri_res.edge_probs, names, os.path.join(out, "nri_inferred_graph.png"), thresh=args.graph_thresh)

    # edge type heatmaps
    for k in range(args.n_edge_types):
        plot_heatmap(
            nri_res.edge_type_probs[k],
            title=f"NRI edge type probability k={k} (receiver i, sender j)",
            xlabel="sender j",
            ylabel="receiver i",
            xticks=names,
            yticks=names,
            out_path=os.path.join(out, f"nri_edge_type_{k}_heatmap.png"),
            vmin=0.0,
            vmax=1.0,
        )

    # average root cause probabilities per event type: groups x drones
    for ev_name, gdict in ev_out.avg_causes.items():
        groups = list(gdict.keys())
        mat = np.stack([np.array(gdict[g], dtype=np.float32) for g in groups], axis=0)
        plot_heatmap(
            mat,
            title=f"Average root-cause probabilities per drone (event={ev_name})",
            xlabel="drone",
            ylabel="cause group",
            xticks=names,
            yticks=groups,
            out_path=os.path.join(out, f"avg_root_cause_{ev_name}.png"),
            vmin=0.0,
            vmax=1.0,
        )

    # event probability series
    proba_run_0 = {ev: probas[0] for ev, probas in ev_out.proba.items()}
    plot_event_probas(logs.times, proba_run_0, names, out)

    # impacts
    for ev_name, mat in impacts.items():
        plot_heatmap(
            mat,
            title=f"Systemic impact: {ev_name} (i triggers) -> generic failure (j affected)",
            xlabel="affected drone j",
            ylabel="trigger drone i",
            xticks=names,
            yticks=names,
            out_path=os.path.join(out, f"impact_{ev_name}_heatmap.png"),
            vmin=float(np.min(mat)),
            vmax=float(np.max(mat)),
        )


    # High-level summary for report
    summary = {
        "log_dir": args.log_dir,
        "output_dir": args.output_dir,
        "n_drones": N,
        "n_steps": int(len(logs.times)),
        "dt": logs.dt,
        "nri": {
            "seq_len": args.seq_len,
            "n_edge_types": args.n_edge_types,
            "hidden": args.hidden,
            "nri_steps": args.nri_steps,
            "final_train_mse": float(np.mean(nri_res.nri_loss_curve[-50:])) if len(nri_res.nri_loss_curve) >= 50 else float(np.mean(nri_res.nri_loss_curve)),
        },
        "detection": {
            "collision_dist": args.collision_dist,
            "formation_thresh": args.formation_thresh,
            "gnss_quantile": args.gnss_quantile,
            "wind_quantile": args.wind_quantile,
            "suboptimal_quantile": args.suboptimal_quantile,
        },
        "top_edges": top_edges_from_matrix(nri_res.edge_probs, names, k=10),
    }

    with open(os.path.join(out, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Done] Outputs saved to: {os.path.abspath(out)}")
    return summary


def top_edges_from_matrix(mat: np.ndarray, names: List[str], k: int = 10) -> List[Dict]:
    N = len(names)
    edges = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            edges.append((float(mat[i, j]), j, i))  # w, sender, receiver
    edges.sort(reverse=True, key=lambda x: x[0])
    out = []
    for w, j, i in edges[:k]:
        out.append({"sender": names[j], "receiver": names[i], "weight": w})
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal analysis (NRI + event root cause) for UxS swarm logs.")

    p.add_argument("--log_dir", type=str, default="logs", help="Directory containing drone_*.csv logs.")
    p.add_argument("--output_dir", type=str, default="causal_out", help="Output directory for matrices and figures.")

    # Sampling / speed
    p.add_argument("--downsample", type=int, default=1, help="Downsample logs by keeping every k-th row.")
    p.add_argument("--seed", type=int, default=1, help="Random seed.")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda (if available).")
    p.add_argument("--torch_threads", type=int, default=1, help="Nombre de threads CPU pour PyTorch (mettre 1 pour éviter un énorme overhead sur petites tailles).")

    # Failure detection thresholds
    p.add_argument("--leader_index", type=int, default=0, help="Leader drone index for formation reference.")
    p.add_argument("--collision_dist", type=float, default=0.6, help="Collision distance threshold [m].")
    p.add_argument("--formation_thresh", type=float, default=1.0, help="Formation loss threshold [m].")
    p.add_argument("--gnss_quantile", type=float, default=0.95, help="Quantile threshold for GNSS degradation.")
    p.add_argument("--wind_quantile", type=float, default=0.95, help="Quantile threshold for wind loss.")
    p.add_argument("--suboptimal_quantile", type=float, default=0.95, help="Quantile for suboptimal trajectory.")
    p.add_argument("--suboptimal_persist", type=int, default=5, help="Min consecutive steps for suboptimal label.")

    # NRI training params
    p.add_argument("--use_measured", action="store_true", help="Use measured pos (EKF) instead of ground truth for NRI.")
    p.add_argument("--seq_len", type=int, default=40, help="Sequence length (window) for NRI.")
    p.add_argument("--n_edge_types", type=int, default=3, help="Number of edge types (type 0 assumed no-edge).")
    p.add_argument("--hidden", type=int, default=128, help="Hidden dimension for NRI.")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for NRI training.")
    p.add_argument("--nri_steps", type=int, default=5000, help="Training steps for NRI.")
    p.add_argument("--max_train_windows", type=int, default=5000, help="Max sampled windows for training pool.")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate for NRI.")
    p.add_argument("--tau_start", type=float, default=1.0, help="Initial Gumbel-Softmax temperature.")
    p.add_argument("--tau_end", type=float, default=0.5, help="Final Gumbel-Softmax temperature.")
    p.add_argument("--tau_anneal_steps", type=int, default=1500, help="Anneal steps for tau.")
    p.add_argument("--verbose_every", type=int, default=100, help="Print NRI training status every k steps.")
    p.add_argument("--graph_thresh", type=float, default=0.3, help="Edge threshold for graph visualization.")

    # Event models / causal explanations
    p.add_argument("--event_lag", type=int, default=1, help="Lag (in timesteps) for event prediction features.")
    p.add_argument("--max_events_to_explain", type=int, default=2000, help="Cap number of positive event instances to explain per event.")
    p.add_argument("--granger_maxlag", type=int, default=5, help="Max lag for Granger tests.")
    p.add_argument("--impact_horizon", type=int, default=10, help="Horizon steps for systemic impact estimation.")

    return p



def main():
    args = build_argparser().parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
