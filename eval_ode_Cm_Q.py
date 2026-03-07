# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import joblib
import matplotlib.pyplot as plt


# --------------------------
# Config (MUST match latest train)
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 改成你“最新训练脚本”保存的权重名（我给的训练脚本默认是这个）
BEST_MODEL_PATH = "best_aerosol_predictor_monotone_Q_Cm_dm_scheme_ZrO2.pth"
SCALER_PATH = "static_scaler.joblib"

EXCEL_FILE_PATH = "/home/user/ZHOU-Wen/Aerosol_DT/Aerosol_data/Aerosol data 1__filtered.xlsx"
SHEET_NAME = "Sheet1"
NROWS = 1361  # None for all

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

# Model hyperparams (must match latest train)
STATIC_EMBED_DIM = 1024
TIME_EMBED_DIM = 128
TRANSFORMER_D_MODEL = 2048
N_HEAD = 4
NUM_TRANSFORMER_LAYERS = 7
TRANSFORMER_DROPOUT = 0.02

# ✅ must match latest train script (for strict baseline dominance)
COEF_FLOOR = 1e-6

# Eval config
EVAL_BATCH_SIZE = 128
TRAIN_SPLIT_RATIO = 0.8  # for making a test split identical style to your old eval
WIDE_BATCH_SIZE = 512    # for wide table generation


# --------------------------
# Plot helpers
# --------------------------
def _savefig(name: str, dpi=300):
    plt.tight_layout()
    plt.savefig(name, dpi=dpi)
    plt.close()
    print(f"✅ Saved: {name}")


# --------------------------
# Metrics export (keep intact)
# --------------------------
def evaluate_metrics(
    y_true,
    y_pred,
    inputs_static,
    inputs_time,
    feature_names,
    threshold=1e-3,
    save_path="evaluation_details.xlsx",
    make_plots=False,
):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    inputs_static = np.array(inputs_static)
    inputs_time = np.array(inputs_time, dtype=float).flatten()

    if not (len(y_true) == len(y_pred) == len(inputs_static) == len(inputs_time)):
        raise ValueError("Length mismatch among y_true, y_pred, inputs_static, inputs_time")

    total_samples = len(y_true)

    # t=0 => treat true as 1.0 for error reporting
    t0_mask = np.isclose(inputs_time, 0.0)
    num_t0 = int(np.sum(t0_mask))
    if num_t0 > 0:
        y_true[t0_mask] = 1.0
        print(f"[Eval] Set y_true=1.0 for {num_t0} samples at t=0 (for error calc).")

    print(f"\n--- Evaluation (N={total_samples}) ---")
    print(f"Threshold: {threshold}")
    print("Generating Excel report" + (" + plots..." if make_plots else "..."))

    y_true_eff = np.maximum(y_true, threshold)
    y_pred_eff = np.maximum(y_pred, threshold)
    clamped_flag = ((y_true < threshold) | (y_pred < threshold)).astype(int)

    abs_diff_raw = np.abs(y_true - y_pred)
    sq_err_raw = (y_true - y_pred) ** 2

    abs_diff_eff = np.abs(y_true_eff - y_pred_eff)
    rel_err_eff = abs_diff_eff / np.abs(y_true_eff)
    rmspe_term = rel_err_eff ** 2

    signed_rel_error_pct = (y_pred_eff - y_true_eff) / y_true_eff * 100.0

    pm_ratios = y_pred_eff / y_true_eff
    pm_mean = np.mean(pm_ratios)
    pm_std = np.std(pm_ratios)

    mape = np.mean(rel_err_eff) * 100.0
    rmspe = np.sqrt(np.mean(rmspe_term))

    sum_abs_true = np.sum(np.abs(y_true))
    if sum_abs_true == 0:
        wmape = float("inf")
        wmape_weight = np.zeros_like(y_true)
    else:
        wmape = (np.sum(abs_diff_raw) / sum_abs_true) * 100.0
        wmape_weight = abs_diff_raw / sum_abs_true

    rmse = np.sqrt(np.mean(sq_err_raw))
    u = np.mean(y_true)
    nrmse = rmse / u if u != 0 else float("inf")

    ss_res = np.sum(sq_err_raw)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    # Save detail excel
    data_dict = {}
    for i, name in enumerate(feature_names):
        data_dict[name] = inputs_static[:, i]
    data_dict["Time (min)"] = inputs_time
    data_dict.update({
        "Measured_Value (True_Used)": y_true,
        "Predicted_Value (Pred)": y_pred,
        "True_Eff": y_true_eff,
        "Pred_Eff": y_pred_eff,
        "Absolute_Diff_Raw": abs_diff_raw,
        "Absolute_Diff_Eff": abs_diff_eff,
        "P/M_Ratio": pm_ratios,
        "Rel_Error_Eff": rel_err_eff,
        "Signed_Rel_Error (%)": signed_rel_error_pct,
        "MAPE_Contribution (%)": rel_err_eff * 100.0,
        "RMSPE_Term": rmspe_term,
        "Squared_Error": sq_err_raw,
        "WMAPE_Weight": wmape_weight,
        "Clamped_Flag": clamped_flag,
    })
    pd.DataFrame(data_dict).to_excel(save_path, index=False)
    print(f"✅ Saved Excel: {save_path}")

    print("-" * 50)
    print(f"R^2:    {r2:.4f}")
    print(f"WMAPE:  {wmape:.2f}%")
    print(f"NRMSE:  {nrmse:.4f}")
    print("-" * 50)
    print(f"MAPE:   {mape:.2f}%")
    print(f"P/M μ:  {pm_mean:.4f}")
    print(f"P/M σ:  {pm_std:.4f}")
    print(f"RMSPE:  {rmspe:.4f}")
    print("-" * 50)

    if make_plots:
        NATURE_RED = "#D62728"  # Nature-style red
        import matplotlib as mpl
        # --- 放大这三张图的字体（临时）---
        _old_rc = mpl.rcParams.copy()
        mpl.rcParams.update({
            "font.size": 24,        # 兜底：文本默认
            "axes.labelsize": 18,   # x/y label
            "xtick.labelsize": 14,  # x tick
            "ytick.labelsize": 14,  # y tick
            "legend.fontsize": 20,  # legend
            "axes.titlesize": 18,   # title（如果用到）
})

        # Fig 1: Pred vs Measured
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, s=15, alpha=0.5, edgecolors="none")
        x_line = np.linspace(0.0, 1.0, 200)
        plt.plot(x_line, x_line, linewidth=2, color="0.15", label="Perfect")
        plt.plot(x_line, 1.10 * x_line, "--", linewidth=2, color=NATURE_RED, label="+10%")  # <-- 改这里
        plt.plot(x_line, 0.90 * x_line, "--", linewidth=2, label="-10%")
        plt.xlabel("Measured")
        plt.ylabel("Predicted")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(loc="upper left")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        _savefig("fig_pred_vs_measured.png", dpi=300)

        # Fig 2: Error histogram
        bins = np.arange(-100, 105, 5)
        centers = (bins[:-1] + bins[1:]) / 2.0
        counts, _ = np.histogram(signed_rel_error_pct, bins=bins)
        perc = counts / total_samples * 100.0
        plt.figure(figsize=(10, 6))
        plt.bar(centers, perc, width=(bins[1]-bins[0]), alpha=0.5, edgecolor="black")
        plt.axvline(0.0, linewidth=2)
        plt.axvline(10.0, linestyle="--", linewidth=2, color=NATURE_RED, label="±10%")  # <-- 改这里
        plt.axvline(-10.0, linestyle="--", linewidth=2, color=NATURE_RED)               # <-- 改这里
        plt.xlabel("Relative error (%)")
        plt.ylabel("% of data")
        plt.grid(True, linestyle="--", linewidth=0.5, axis="y")
        plt.legend(loc="upper right")
        plt.xlim(-100, 100)
        plt.ylim(0.0, perc.max()*1.1 if perc.size else 1.0)
        _savefig("fig_error_histogram.png", dpi=300)

        # Fig 3: error vs each static feature
        n_features = inputs_static.shape[1]
        n_cols = 4
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows), sharey=True)
        axes = np.array(axes).reshape(n_rows, n_cols)
        for i in range(n_features):
            r = i // n_cols
            c = i % n_cols
            ax = axes[r, c]
            ax.scatter(inputs_static[:, i], signed_rel_error_pct, s=10, alpha=0.5, edgecolors="none")
            ax.axhline(0, linewidth=2)
            ax.axhline(30, linestyle="--", linewidth=2, color=NATURE_RED)   # <-- 改这里
            ax.axhline(-30, linestyle="--", linewidth=2, color=NATURE_RED)  # <-- 改这里
            ax.set_xlabel(feature_names[i], fontsize=9)
            if c == 0:
                ax.set_ylabel("Rel error [%]")
            ax.set_ylim(-50, 50)
            ax.grid(True, linestyle="--", linewidth=0.5)
        for j in range(n_features, n_rows*n_cols):
            r = j // n_cols
            c = j % n_cols
            fig.delaxes(axes[r, c])
        _savefig("fig_error_vs_features.png", dpi=300)

    return {
        "P/M Mean": pm_mean,
        "P/M Std": pm_std,
        "RMSPE": rmspe,
        "MAPE": mape,
        "R2": r2,
        "RMSE": rmse,
        "WMAPE": wmape,
        "NRMSE": nrmse,
    }


# --------------------------
# Model blocks (MATCH latest train: Q + Cm + dm monotone)
# --------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even")
        self.embed_dim = embed_dim
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        pe = torch.zeros(t.size(0), self.embed_dim, device=t.device)
        pe[:, 0::2] = torch.sin(t * self.div_term)
        pe[:, 1::2] = torch.cos(t * self.div_term)
        return pe


class StaticFeatureEmbedder(nn.Module):
    def __init__(self, num_static_features, embed_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(num_static_features, embed_dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, s):
        x = self.act(self.fc1(s))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MonotoneRateNet(nn.Module):
    """
    k = k0 + kQ*softplus(Q_scaled) + kC*softplus(Cm_scaled) + kD*softplus(dm_scaled)
    with k0,kQ,kC,kD >= 0 and kC,kD >= COEF_FLOOR for strict dominance.
    """
    def __init__(
        self,
        num_static_features,
        static_embed_dim,
        time_embed_dim,
        transformer_d_model,
        n_head,
        num_transformer_layers,
        transformer_dropout,
        idx_Q,
        idx_Cm,
        idx_dm,
        coef_floor=0.0,
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
            d_model=transformer_d_model, nhead=n_head, dropout=transformer_dropout,
            batch_first=True, dim_feedforward=transformer_d_model * 4
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        # ✅ latest train: head outputs 4 values (k0,kQ,kC,kD)
        self.head = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.GELU(),
            nn.Linear(transformer_d_model // 2, 4)
        )
        self.softplus = nn.Softplus()

    def forward(self, t_scalar, static_features):
        B = static_features.size(0)
        if t_scalar.numel() == 1:
            t_vec = t_scalar.repeat(B)
        else:
            t_vec = t_scalar
        t_query = t_vec.unsqueeze(-1)

        # IMPORTANT: coefficient net must NOT depend on Q/Cm/dm
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
        time_tok = out[:, 1, :]

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
        return k  # >= 0


class MonotoneIntegralPredictor(nn.Module):
    def __init__(self, rate_net: MonotoneRateNet, base_time_grid: torch.Tensor):
        super().__init__()
        self.rate_net = rate_net
        self.register_buffer("base_time_grid", base_time_grid)

    def forward(self, static_features, t_query):
        B = static_features.size(0)
        device = static_features.device
        t_flat = t_query.squeeze(-1)

        times = torch.unique(torch.cat([self.base_time_grid.to(device), t_flat])).sort()[0]
        T = times.numel()

        k_list = []
        for i in range(T):
            k_list.append(self.rate_net(times[i], static_features))  # [B,1]
        k_ts = torch.stack(k_list, dim=0)  # [T,B,1]

        if T == 1:
            Iq = torch.zeros(B, 1, device=device)
        else:
            dts = (times[1:] - times[:-1]).view(T - 1, 1, 1)
            avg_k = 0.5 * (k_ts[:-1] + k_ts[1:])
            inc = avg_k * dts  # >=0

            I = torch.zeros(T, B, 1, device=device)
            I[1:] = torch.cumsum(inc, dim=0)

            idx = torch.searchsorted(times, t_flat)
            Iq = I[idx, torch.arange(B, device=device), :]

        y = torch.exp(-Iq)  # (0,1]
        return y

    @property
    def device(self):
        return next(self.parameters()).device


# --------------------------
# Dataset: same missing handling as train (skip NaNs)
# --------------------------
class AerosolQueryDataset(Dataset):
    def __init__(self, data_df, time_columns_map, static_feature_cols,
                 time_normalization_factor=60.0, static_feature_scaler=None):
        static_raw = data_df[static_feature_cols].values.astype(np.float32)
        if np.any(np.isnan(static_raw)) or np.any(np.isinf(static_raw)):
            static_raw = np.nan_to_num(static_raw, nan=0.0, posinf=0.0, neginf=0.0)

        if static_feature_scaler is None:
            self.static_features_all = static_raw.astype(np.float32)
        else:
            self.static_features_all = static_feature_scaler.transform(static_raw).astype(np.float32)

        self.time_values_all = []  # list of (orig_row_idx, t_norm, y_true)
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
                self.time_values_all.append((row_idx, t_norm, y))

        if len(self.time_values_all) == 0:
            raise ValueError("No valid (row,time)->y samples found (after skipping missing cells).")

    def __len__(self):
        return len(self.time_values_all)

    def __getitem__(self, idx):
        orig_idx, t_norm, y_true = self.time_values_all[idx]
        static = self.static_features_all[orig_idx, :]
        return {
            "static_features": torch.tensor(static, dtype=torch.float32),
            "t_query": torch.tensor([t_norm], dtype=torch.float32),
            "y_target": torch.tensor([y_true], dtype=torch.float32),
        }


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # 1) Load scaler
    print(f"Loading scaler: {SCALER_PATH}")
    static_scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded.")

    # 2) Load Excel once
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, nrows=NROWS)
    if df.empty:
        raise RuntimeError("Excel is empty.")

    # Report missing time cells
    time_cols_existing = [c for c in TIME_COLS_MAP.keys() if c in df.columns]
    total_possible = len(df) * len(time_cols_existing)
    valid_count = int(sum(df[c].notna().sum() for c in time_cols_existing))
    print(f"[Data] rows={len(df)}, time_cols={len(time_cols_existing)}, possible={total_possible}, valid={valid_count}, missing={total_possible-valid_count} (skipped)")

    # 3) Build dataset
    full_dataset = AerosolQueryDataset(
        data_df=df,
        time_columns_map=TIME_COLS_MAP,
        static_feature_cols=STATIC_FEATURE_COLS,
        time_normalization_factor=TIME_NORMALIZATION_FACTOR,
        static_feature_scaler=static_scaler
    )

    # 4) Make a test split (like your old eval)
    num_total = len(full_dataset)
    num_train = int(TRAIN_SPLIT_RATIO * num_total)
    if num_total - num_train <= 0:
        raise RuntimeError("Test set is empty; adjust TRAIN_SPLIT_RATIO.")

    gen = torch.Generator().manual_seed(42)
    shuffled = torch.randperm(num_total, generator=gen).tolist()
    test_indices = shuffled[num_train:]
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    # 5) Build model structure exactly as latest train
    base_times = sorted(set([0.0] + [float(t) / TIME_NORMALIZATION_FACTOR for t in TIME_COLS_MAP.values()]))
    base_time_grid = torch.tensor(base_times, dtype=torch.float32)

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

    # 6) Load weights
    print(f"Loading model weights: {BEST_MODEL_PATH}")
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {BEST_MODEL_PATH}")

    state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)  # ✅ should be strict for architecture match
    print("Model loaded.")
    model.eval()

    # -----------------------
    # A) Evaluate on test set
    # -----------------------
    all_targets, all_preds, all_static, all_times = [], [], [], []

    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            static_feats = batch["static_features"].to(DEVICE)
            t_query = batch["t_query"].to(DEVICE)
            y_true = batch["y_target"].to(DEVICE)

            y_pred = model(static_feats, t_query)  # already (0,1]

            all_targets.append(y_true.cpu().numpy())
            all_preds.append(y_pred.cpu().numpy())
            all_static.append(static_feats.cpu().numpy())
            all_times.append(t_query.cpu().numpy())

    y_true_np = np.concatenate(all_targets).squeeze()
    y_pred_np = np.concatenate(all_preds).squeeze()
    static_scaled_np = np.concatenate(all_static)
    t_norm_np = np.concatenate(all_times).squeeze()

    # inverse transform to physical values for reporting
    static_raw_np = static_scaler.inverse_transform(static_scaled_np)
    t_min_np = t_norm_np * TIME_NORMALIZATION_FACTOR

    evaluate_metrics(
        y_true=y_true_np,
        y_pred=y_pred_np,
        inputs_static=static_raw_np,
        inputs_time=t_min_np,
        feature_names=STATIC_FEATURE_COLS,
        save_path="evaluation_details_test.xlsx",
        make_plots=False
    )

    # -----------------------
    # B) Wide table: test set
    # -----------------------
    print("\nGenerating test wide table: test_predictions_wide.xlsx ...")
    scaled_static_all = full_dataset.static_features_all
    raw_static_all = static_scaler.inverse_transform(scaled_static_all)

    minute_to_label = {int(v): k for k, v in TIME_COLS_MAP.items()}
    pred_map_test = defaultdict(dict)

    for start in range(0, len(test_indices), WIDE_BATCH_SIZE):
        batch_idxs = test_indices[start:start + WIDE_BATCH_SIZE]
        static_list, t_list, orig_list = [], [], []

        for i in batch_idxs:
            orig_idx, t_norm, _ = full_dataset.time_values_all[i]
            static_list.append(scaled_static_all[orig_idx])
            t_list.append(t_norm)
            orig_list.append(orig_idx)

        static_batch = torch.tensor(static_list, dtype=torch.float32, device=DEVICE)
        t_batch = torch.tensor(t_list, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

        with torch.no_grad():
            preds = model(static_batch, t_batch).cpu().numpy().squeeze()

        tmins = np.array(t_list) * TIME_NORMALIZATION_FACTOR
        for orig_idx, tmin, p in zip(orig_list, tmins, preds):
            label = minute_to_label.get(int(round(float(tmin))), None)
            if label is not None:
                pred_map_test[orig_idx][label] = float(p)

    rows = []
    for orig_idx in sorted(pred_map_test.keys()):
        row = {STATIC_FEATURE_COLS[j]: raw_static_all[orig_idx][j] for j in range(len(STATIC_FEATURE_COLS))}
        for label in TIME_COLS_MAP.keys():
            row[label] = pred_map_test[orig_idx].get(label, np.nan)
        rows.append(row)

    pd.DataFrame(rows).to_excel("test_predictions_wide.xlsx", index=False)
    print("✅ Saved: test_predictions_wide.xlsx")

    # -----------------------
    # C) Wide table + metrics: full dataset
    # -----------------------
    print("\nEvaluating full dataset + generating all_predictions_wide.xlsx ...")
    pred_map_all = defaultdict(dict)

    y_true_all, y_pred_all, s_all, t_all = [], [], [], []

    all_indices = list(range(len(full_dataset.time_values_all)))
    for start in range(0, len(all_indices), WIDE_BATCH_SIZE):
        batch_idxs = all_indices[start:start + WIDE_BATCH_SIZE]
        static_list, t_list, y_list, orig_list = [], [], [], []

        for i in batch_idxs:
            orig_idx, t_norm, y_true = full_dataset.time_values_all[i]
            static_list.append(scaled_static_all[orig_idx])
            t_list.append(t_norm)
            y_list.append(y_true)
            orig_list.append(orig_idx)

        static_batch = torch.tensor(static_list, dtype=torch.float32, device=DEVICE)
        t_batch = torch.tensor(t_list, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

        with torch.no_grad():
            preds = model(static_batch, t_batch).cpu().numpy().squeeze()

        tmins = np.array(t_list) * TIME_NORMALIZATION_FACTOR
        for orig_idx, tmin, y_t, y_p in zip(orig_list, tmins, y_list, preds):
            y_true_all.append(float(y_t))
            y_pred_all.append(float(y_p))
            s_all.append(raw_static_all[orig_idx])
            t_all.append(float(tmin))

            label = minute_to_label.get(int(round(float(tmin))), None)
            if label is not None:
                pred_map_all[orig_idx][label] = float(y_p)

    evaluate_metrics(
        y_true=y_true_all,
        y_pred=y_pred_all,
        inputs_static=s_all,
        inputs_time=t_all,
        feature_names=STATIC_FEATURE_COLS,
        save_path="evaluation_details_all.xlsx",
        make_plots=True
    )

    # Build all wide rows (one per original row)
    all_rows = []
    num_rows = raw_static_all.shape[0]
    for orig_idx in range(num_rows):
        row = {STATIC_FEATURE_COLS[j]: raw_static_all[orig_idx][j] for j in range(len(STATIC_FEATURE_COLS))}
        for label in TIME_COLS_MAP.keys():
            row[label] = pred_map_all.get(orig_idx, {}).get(label, np.nan)
        all_rows.append(row)

    pd.DataFrame(all_rows).to_excel("all_predictions_wide.xlsx", index=False)
    print("✅ Saved: all_predictions_wide.xlsx")

    print("\n=== Done ===")