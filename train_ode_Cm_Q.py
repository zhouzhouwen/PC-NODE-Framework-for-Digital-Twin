# -*- coding: utf-8 -*-
"""
FINAL++: Hard-constrained monotone model (Scheme 2: trapezoid integration)
ADDED HARD CONSTRAINT: y(dm=0,Cm=0) > y(dm>0,Cm>0) for any fixed other inputs and any t>0

Hard guarantees (structural):
1) Time monotonicity and positivity:
   y(t) = exp(- ∫_0^t k(s,τ) dτ), with k(s,t) >= 0
   => y(t) in (0,1], monotone non-increasing in t

2) Monotonic w.r.t Spray Flow Rate Q, Mist concentration Cm, and Mist mean size dm at any fixed t:
   k(s,t) = k0(s_other,t)
          + kQ(s_other,t)*softplus(Q_scaled)
          + kC(s_other,t)*softplus(Cm_scaled)
          + kD(s_other,t)*softplus(dm_scaled)
   with k0,kQ,kC,kD >= 0
   => increasing Q or Cm or dm increases k => decreases y(t) (hard)

3) Strict baseline dominance for t>0:
   y(Q fixed, dm=0,Cm=0, t) > y(Q fixed, dm>0,Cm>0, t) for any t>0
   Achieved by enforcing kC,kD >= COEF_FLOOR > 0 (strictly positive coefficients).

Missing time entries in Excel:
- If a time column is blank/NaN for a row, that (row,time) pair is skipped (no sample generated).
"""

import os
import csv
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import joblib


# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_seed(SEED)


# --------------------------
# Config
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output files (saved in current working directory unless you set absolute paths)
BEST_MODEL_PATH = "best_aerosol_predictor_monotone_Q_Cm_dm_scheme_ZrO2.pth"
SCALER_PATH = "static_scaler.joblib"
LOSS_LOG_PATH = "training_losses_monotone_Q_Cm_dm_scheme_ZrO2.csv"

# Data
EXCEL_FILE_PATH = "/home/user/ZHOU-Wen/Aerosol_DT/Aerosol_data/Aerosol data 1__filtered.xlsx"
SHEET_NAME = "Sheet1"
NROWS = 1369  # set None to read all rows

TIME_COLS_MAP = {
    "0min": 0.0, "1min": 1.0, "3min": 3.0, "5min": 5.0,
    "10min": 10.0, "15min": 15.0, "20min": 20.0, "30min": 30.0,
    "40min": 40.0, "50min": 50.0, "60min": 60.0
}
TIME_NORMALIZATION_FACTOR = 60.0

STATIC_FEATURE_COLS = [
    "Spray Coverage",
    "Spray Flow Rate (L/min)",                 # Q
    "Droplet mean size  (μm)",
    "Orfice diameter (mm)",
    "Mist mean size (μm)",                     # dm
    "Mist mean concentration (P/cm^3)",        # Cm
    "Aerosol density (kg/m3)",
    "Contact Angle",
    "Aerosol size (μm)"
]
NUM_STATIC_FEATURES = len(STATIC_FEATURE_COLS)

IDX_Q  = STATIC_FEATURE_COLS.index("Spray Flow Rate (L/min)")
IDX_DM = STATIC_FEATURE_COLS.index("Mist mean size (μm)")
IDX_CM = STATIC_FEATURE_COLS.index("Mist mean concentration (P/cm^3)")

# Model hyperparams (your large settings)
STATIC_EMBED_DIM = 1024
TIME_EMBED_DIM = 128
TRANSFORMER_D_MODEL = 2048
N_HEAD = 4
NUM_TRANSFORMER_LAYERS = 7
TRANSFORMER_DROPOUT = 0.02

# Training hyperparams
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1000
BATCH_SIZE = 32
CLIP_VALUE = 1.0

# Split ratios
OVERALL_TRAIN_RATIO = 0.8
VALIDATION_SPLIT_RATIO = 0.08  # within train_val part

# --- NEW: strict positivity floor for coefficients tied to Cm and dm
# Makes y(dm=0,Cm=0,t) strictly larger than y(dm>0,Cm>0,t) for any t>0.
COEF_FLOOR = 1e-6


# --------------------------
# Loss
# --------------------------
class HybridMSE_RMSPELoss(nn.Module):
    """
    Mixed loss:
      - MSE: absolute error
      - RMSPE-like: relative error (clipped) for sensitivity on small values
    """
    def __init__(
        self,
        min_true: float = 1e-5,
        alpha: float = 0.3,
        max_rel: float | None = 10.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.min_true = min_true
        self.alpha = alpha
        self.max_rel = max_rel
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view_as(target)
        mse = torch.mean((pred - target) ** 2)

        target_clamped = torch.clamp(target, min=self.min_true)
        rel = (pred - target_clamped) / (target_clamped + self.eps)
        if self.max_rel is not None:
            rel = torch.clamp(rel, -self.max_rel, self.max_rel)

        rmspe = torch.sqrt(torch.mean(rel ** 2) + self.eps)

        with torch.no_grad():
            scale = mse.detach() / (rmspe.detach() + self.eps)

        return mse + self.alpha * scale * rmspe


# --------------------------
# Embedding blocks
# --------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.embed_dim = embed_dim
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,1] or [B]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        pe = torch.zeros(t.size(0), self.embed_dim, device=t.device)
        pe[:, 0::2] = torch.sin(t * self.div_term)
        pe[:, 1::2] = torch.cos(t * self.div_term)
        return pe


class StaticFeatureEmbedder(nn.Module):
    def __init__(self, num_static_features: int, embed_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(num_static_features, embed_dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(s))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --------------------------
# Rate net: outputs k0,kQ,kC,kD >= 0
# and uses softplus on (Q_scaled, Cm_scaled, dm_scaled) as "effective" nonnegative inputs
# --------------------------
class MonotoneRateNet(nn.Module):
    """
    k(s,t) = k0 + kQ*softplus(Q_scaled) + kC*softplus(Cm_scaled) + kD*softplus(dm_scaled)
    where k0,kQ,kC,kD >= 0 (Softplus), and kC,kD >= COEF_FLOOR for strict dominance.

    Guarantees:
    - k >= 0 always
    - Increasing Q/Cm/dm (raw) increases their scaled values monotonically (StandardScaler is linear monotone),
      then softplus is monotone => k nondecreasing => y nonincreasing.
    - With kC,kD >= COEF_FLOOR > 0: if Cm or dm strictly increases, k strictly increases => for any t>0, y strictly decreases.
    """
    def __init__(
        self,
        num_static_features: int,
        static_embed_dim: int,
        time_embed_dim: int,
        transformer_d_model: int,
        n_head: int,
        num_transformer_layers: int,
        transformer_dropout: float,
        idx_Q: int,
        idx_Cm: int,
        idx_dm: int,
        coef_floor: float = 0.0,
    ):
        super().__init__()
        self.idx_Q = idx_Q
        self.idx_Cm = idx_Cm
        self.idx_dm = idx_dm
        self.coef_floor = float(coef_floor)

        self.static_embedder = StaticFeatureEmbedder(num_static_features, static_embed_dim, transformer_dropout)
        self.time_embedder = SinusoidalTimeEmbedding(time_embed_dim)

        self.proj_static = nn.Linear(static_embed_dim, transformer_d_model) if static_embed_dim != transformer_d_model else nn.Identity()
        self.proj_time = nn.Linear(time_embed_dim, transformer_d_model) if time_embed_dim != transformer_d_model else nn.Identity()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=n_head,
            dropout=transformer_dropout,
            batch_first=True,
            dim_feedforward=transformer_d_model * 4
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        # output 4 numbers -> softplus -> nonnegative
        self.head = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.GELU(),
            nn.Linear(transformer_d_model // 2, 4),
        )
        self.softplus = nn.Softplus()

    def forward(self, t_scalar: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        """
        Return k: [B,1], guaranteed >= 0
        """
        B = static_features.size(0)

        # make t vector [B]
        if t_scalar.numel() == 1:
            t_vec = t_scalar.repeat(B)
        else:
            t_vec = t_scalar
        t_query = t_vec.unsqueeze(-1)  # [B,1]

        # IMPORTANT: coefficients should not depend on Q/Cm/dm
        static_for_coeff = static_features.clone()
        static_for_coeff[:, self.idx_Q] = 0.0
        static_for_coeff[:, self.idx_Cm] = 0.0
        static_for_coeff[:, self.idx_dm] = 0.0

        s_emb = self.static_embedder(static_for_coeff)
        t_emb = self.time_embedder(t_query.float())

        s_proj = self.proj_static(s_emb)
        t_proj = self.proj_time(t_emb)

        seq = torch.stack([s_proj, t_proj], dim=1)  # [B,2,D]
        out = self.encoder(seq)
        time_tok = out[:, 1, :]  # [B,D]

        raw = self.head(time_tok)  # [B,4]
        k0 = self.softplus(raw[:, 0:1])
        kQ = self.softplus(raw[:, 1:2])
        kC = self.softplus(raw[:, 2:3]) + self.coef_floor
        kD = self.softplus(raw[:, 3:4]) + self.coef_floor

        Qs  = static_features[:, self.idx_Q:self.idx_Q+1]
        Cms = static_features[:, self.idx_Cm:self.idx_Cm+1]
        Dms = static_features[:, self.idx_dm:self.idx_dm+1]

        Q_eff  = self.softplus(Qs)
        Cm_eff = self.softplus(Cms)
        Dm_eff = self.softplus(Dms)

        k = k0 + kQ * Q_eff + kC * Cm_eff + kD * Dm_eff
        return k  # [B,1] >= 0


# --------------------------
# Scheme predictor: y = exp(-Integral k dt) via trapezoid rule
# --------------------------
class MonotoneIntegralPredictor(nn.Module):
    """
    y(t)=exp(-∫_0^t k(s,τ)dτ), computed by trapezoid on a time grid.

    Hard properties:
    - k>=0 => integral >=0 => y in (0,1]
    - integral is nondecreasing in t => y is nonincreasing in t
    """
    def __init__(self, rate_net: MonotoneRateNet, base_time_grid: torch.Tensor):
        super().__init__()
        self.rate_net = rate_net
        self.register_buffer("base_time_grid", base_time_grid)

    def forward(self, static_features: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
        """
        static_features: [B,F] (scaled)
        t_query: [B,1] normalized time (e.g. minutes/60)
        return y: [B,1] in (0,1]
        """
        B = static_features.size(0)
        device = static_features.device
        t_flat = t_query.squeeze(-1)  # [B]

        # Build integration grid: include base grid + query times, unique & sorted
        times = torch.unique(torch.cat([self.base_time_grid.to(device), t_flat])).sort()[0]  # [T]
        T = times.numel()

        k_list = []
        for i in range(T):
            k_list.append(self.rate_net(times[i], static_features))  # [B,1]
        k_ts = torch.stack(k_list, dim=0)  # [T,B,1]

        if T == 1:
            Iq = torch.zeros(B, 1, device=device)
        else:
            dts = (times[1:] - times[:-1]).view(T - 1, 1, 1)     # [T-1,1,1]
            avg_k = 0.5 * (k_ts[:-1] + k_ts[1:])                 # [T-1,B,1]
            inc = avg_k * dts                                    # [T-1,B,1] >= 0

            I = torch.zeros(T, B, 1, device=device)
            I[1:] = torch.cumsum(inc, dim=0)                     # [T,B,1], nondecreasing

            idx = torch.searchsorted(times, t_flat)              # [B]
            Iq = I[idx, torch.arange(B, device=device), :]       # [B,1]

        y = torch.exp(-Iq)  # (0,1]
        return y


# --------------------------
# Dataset (skips missing time entries)
# --------------------------
class AerosolQueryDataset(Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        time_columns_map: dict,
        static_feature_cols: list,
        time_normalization_factor: float = 60.0,
        static_feature_scaler=None,
    ):
        static_raw = data_df[static_feature_cols].values.astype(np.float32)
        if np.any(np.isnan(static_raw)) or np.any(np.isinf(static_raw)):
            static_raw = np.nan_to_num(static_raw, nan=0.0, posinf=0.0, neginf=0.0)

        if static_feature_scaler is None:
            self.static_features_all = static_raw.astype(np.float32)
        else:
            self.static_features_all = static_feature_scaler.transform(static_raw).astype(np.float32)

        self.samples = []
        for row_idx, row in data_df.iterrows():
            for col, tmin in time_columns_map.items():
                if col not in data_df.columns:
                    continue
                y = row[col]
                if pd.isna(y) or np.isinf(y):
                    continue
                try:
                    y = float(y)
                except Exception:
                    continue
                if np.isinf(y):
                    continue

                t_norm = float(tmin) / float(time_normalization_factor)
                self.samples.append((row_idx, t_norm, y))

        if len(self.samples) == 0:
            raise ValueError("No valid (row,time)->y samples found. Check TIME_COLS_MAP and Excel columns.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row_idx, t_norm, y = self.samples[idx]
        static = self.static_features_all[row_idx, :]
        return {
            "static_features": torch.tensor(static, dtype=torch.float32),
            "t_query": torch.tensor([t_norm], dtype=torch.float32),
            "y_target": torch.tensor([y], dtype=torch.float32),
        }


# --------------------------
# Train / Eval loop
# --------------------------
def loop_fn(model, dataloader, criterion, device, optimizer=None, clip_value=None, current_epoch=0, loop_type="Train"):
    model.train() if optimizer is not None else model.eval()

    total_loss = 0.0
    num_samples = 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch_idx, batch in enumerate(dataloader):
            static_feats = batch["static_features"].to(device)
            t_query = batch["t_query"].to(device)
            y_target = batch["y_target"].to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            pred = model(static_feats, t_query)

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"{loop_type} warning: Epoch {current_epoch}, batch {batch_idx}, pred has NaN/Inf. Skip.")
                continue

            loss = criterion(pred, y_target)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"{loop_type} error: Epoch {current_epoch}, batch {batch_idx}, loss NaN/Inf. Skip.")
                continue

            if optimizer is not None:
                loss.backward()
                if clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()

            bs = y_target.size(0)
            total_loss += float(loss.item()) * bs
            num_samples += bs

    return total_loss / num_samples if num_samples > 0 else float("inf")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, nrows=NROWS)
    if df.empty:
        raise RuntimeError(f"Excel '{EXCEL_FILE_PATH}' sheet '{SHEET_NAME}' is empty.")

    time_cols_existing = [c for c in TIME_COLS_MAP.keys() if c in df.columns]
    if len(time_cols_existing) == 0:
        raise RuntimeError("None of TIME_COLS_MAP columns exist in the Excel sheet.")

    total_possible = len(df) * len(time_cols_existing)
    valid_count = 0
    for c in time_cols_existing:
        valid_count += df[c].notna().sum()
    missing_count = total_possible - valid_count
    print(f"[Data] rows={len(df)}, time_cols={len(time_cols_existing)}, possible(row,time)={total_possible}")
    print(f"[Data] valid points={valid_count}, missing points={missing_count} (missing are skipped)")

    # Build temporary expanded list for splitting
    temp_samples = []
    for row_idx, row in df.iterrows():
        for col, tmin in TIME_COLS_MAP.items():
            if col not in df.columns:
                continue
            y = row[col]
            if pd.isna(y) or np.isinf(y):
                continue
            try:
                y = float(y)
            except Exception:
                continue
            if np.isinf(y):
                continue
            t_norm = float(tmin) / float(TIME_NORMALIZATION_FACTOR)
            temp_samples.append((row_idx, t_norm, y))

    num_expanded = len(temp_samples)
    if num_expanded == 0:
        raise RuntimeError("No valid expanded samples found after skipping missing values.")
    print(f"[Data] expanded samples used for split/training: {num_expanded}")

    # Split expanded indices into train_val
    num_train_val = int(OVERALL_TRAIN_RATIO * num_expanded)
    gen1 = torch.Generator().manual_seed(42)
    shuffled = torch.randperm(num_expanded, generator=gen1).tolist()
    train_val_item_indices = shuffled[:num_train_val]

    # Split train_val into train and val
    num_val_final = int(VALIDATION_SPLIT_RATIO * len(train_val_item_indices))
    num_train_final = len(train_val_item_indices) - num_val_final
    if num_train_final <= 0 or num_val_final <= 0:
        raise RuntimeError("Train/val split produced empty set. Adjust ratios.")

    gen2 = torch.Generator().manual_seed(43)
    internal = torch.randperm(len(train_val_item_indices), generator=gen2).tolist()

    final_train_item_indices = [train_val_item_indices[i] for i in internal[:num_train_final]]
    val_item_indices = [train_val_item_indices[i] for i in internal[num_train_final:]]

    # Fit scaler on unique training ROW indices
    static_raw_all = df[STATIC_FEATURE_COLS].values.astype(np.float32)
    if np.any(np.isnan(static_raw_all)) or np.any(np.isinf(static_raw_all)):
        static_raw_all = np.nan_to_num(static_raw_all, nan=0.0, posinf=0.0, neginf=0.0)

    train_row_indices = sorted(set([temp_samples[i][0] for i in final_train_item_indices]))
    train_static_for_scaler = static_raw_all[train_row_indices]

    static_scaler = StandardScaler()
    static_scaler.fit(train_static_for_scaler)
    joblib.dump(static_scaler, SCALER_PATH)
    print(f"[Save] Scaler saved to: {os.path.abspath(SCALER_PATH)}")

    # Build full dataset with fitted scaler
    full_dataset = AerosolQueryDataset(
        data_df=df,
        time_columns_map=TIME_COLS_MAP,
        static_feature_cols=STATIC_FEATURE_COLS,
        time_normalization_factor=TIME_NORMALIZATION_FACTOR,
        static_feature_scaler=static_scaler,
    )

    train_dataset = Subset(full_dataset, final_train_item_indices)
    val_dataset = Subset(full_dataset, val_item_indices)

    print(f"[Split] Train size={len(train_dataset)} | Val size={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=True if DEVICE.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    # Base integration grid: include all canonical normalized times (including 0)
    base_times = sorted(set([0.0] + [float(t) / TIME_NORMALIZATION_FACTOR for t in TIME_COLS_MAP.values()]))
    base_time_grid = torch.tensor(base_times, dtype=torch.float32)  # [T]

    # Build model
    rate_net = MonotoneRateNet(
        num_static_features=NUM_STATIC_FEATURES,
        static_embed_dim=STATIC_EMBED_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        transformer_d_model=TRANSFORMER_D_MODEL,
        n_head=N_HEAD,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        transformer_dropout=TRANSFORMER_DROPOUT,
        idx_Q=IDX_Q,
        idx_Cm=IDX_CM,
        idx_dm=IDX_DM,
        coef_floor=COEF_FLOOR,
    ).to(DEVICE)

    model = MonotoneIntegralPredictor(rate_net, base_time_grid=base_time_grid).to(DEVICE)

    criterion = HybridMSE_RMSPELoss(min_true=1e-5, alpha=0.3, max_rel=10.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=50, factor=0.5, min_lr=1e-8
    )

    print("Start training...")
    best_score = float("inf")
    losses_log = []

    for epoch in range(NUM_EPOCHS):
        train_loss = loop_fn(
            model, train_loader, criterion, DEVICE,
            optimizer=optimizer, clip_value=CLIP_VALUE,
            current_epoch=epoch + 1, loop_type="Train"
        )
        val_loss = loop_fn(
            model, val_loader, criterion, DEVICE,
            optimizer=None, current_epoch=epoch + 1, loop_type="Val"
        )

        losses_log.append({"Epoch": epoch + 1, "TrainLoss": train_loss, "ValidationLoss": val_loss})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}]  "
                f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  LR: {optimizer.param_groups[0]['lr']:.1e}"
            )

        score = train_loss + val_loss
        scheduler.step(score)

        if (not math.isinf(score)) and (not math.isnan(score)) and score < best_score:
            best_score = score
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Epoch {epoch+1}: Saved best model. Score(train+val)={best_score:.6f}")

    print("Training finished.")
    print(f"Best score(train+val): {best_score}")

    # Save loss CSV
    try:
        with open(LOSS_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Epoch", "TrainLoss", "ValidationLoss"])
            writer.writeheader()
            for row in losses_log:
                writer.writerow(row)
        print(f"[Save] Loss log saved to: {os.path.abspath(LOSS_LOG_PATH)}")
    except IOError:
        print(f"Error: cannot write loss log to {LOSS_LOG_PATH}")

    print(f"[Save] Best model weights path: {os.path.abspath(BEST_MODEL_PATH)}")