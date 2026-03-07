# -*- coding: utf-8 -*-
"""
Multi-row inference + plot (aligned with training script structure)

What it does:
- Loads Excel: EXCEL_FILE_PATH
- Allows input of multiple row numbers (comma-separated). Default: 2,3,4
  (Excel row numbering as shown on the left; with header_row=1, row=2 is first data row)
- Loads scaler + trained model weights (MonotoneRateNet + MonotoneIntegralPredictor, trapezoid integration)
- Plots (for each selected row, same color):
  - Predicted (Model): curve (legend entry uses Aerosol size to distinguish rows)
  - Measured (Exp): discrete points (no extra legend entry to keep legend clean)
- Styling:
  - Times New Roman font
  - Nature-like palette
  - Big fonts (your rcParams style)
"""

import os
import math
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# --------------------------
# Config (MUST match training)
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_PATH = "/home/user/ZHOU-Wen/Aerosol_DT/best_aerosol_predictor_monotone_Q_Cm_dm_scheme_ZrO2.pth"
SCALER_PATH = "static_scaler.joblib"

EXCEL_FILE_PATH = "/home/user/ZHOU-Wen/Aerosol_DT/Aerosol_data/Aerosol data 1__filtered.xlsx"
SHEET_NAME = "Sheet1"

TIME_COLS_MAP = {
    "0min": 0.0, "1min": 1.0, "3min": 3.0, "5min": 5.0,
    "10min": 10.0, "15min": 15.0, "20min": 20.0, "30min": 30.0,
    "40min": 40.0, "50min": 50.0, "60min": 60.0
}
TIME_COLS = list(TIME_COLS_MAP.keys())
TIME_MINS = np.array([TIME_COLS_MAP[c] for c in TIME_COLS], dtype=float)
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
    "Aerosol size (μm)"                        # for legend label
]
NUM_STATIC_FEATURES = len(STATIC_FEATURE_COLS)

IDX_Q  = STATIC_FEATURE_COLS.index("Spray Flow Rate (L/min)")
IDX_DM = STATIC_FEATURE_COLS.index("Mist mean size (μm)")
IDX_CM = STATIC_FEATURE_COLS.index("Mist mean concentration (P/cm^3)")
IDX_AEROSOL_SIZE = STATIC_FEATURE_COLS.index("Aerosol size (μm)")

# Model hyperparams (must match training)
STATIC_EMBED_DIM = 1024
TIME_EMBED_DIM = 128
TRANSFORMER_D_MODEL = 2048
N_HEAD = 4
NUM_TRANSFORMER_LAYERS = 7
TRANSFORMER_DROPOUT = 0.02
COEF_FLOOR = 1e-6  # should match training

# Plot / inference
MAKE_PLOTS = True
CURVE_POINTS = 250
HEADER_ROW = 1  # your screenshot shows header on row 1

# Nature-ish colors (Okabe-Ito + Nature-like extensions; good contrast)
NATURE_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
]

# Nature red (you used before)
NATURE_RED = "#D62728"


# --------------------------
# Utilities
# --------------------------
def excel_row_to_df_index(excel_row: int, header_row: int = 1) -> int:
    """
    If header_row=1, excel_row=2 => df_index=0 (first data row).
    """
    return int(excel_row - header_row - 1)


def normalize_excel_row_input(user_row: int, header_row: int = 1) -> int:
    """
    Make input more forgiving:
    - If user enters 1, treat as "first data row" -> excel_row = header_row + 1
    - If user enters <= header_row, clamp to first data row
    - Else treat as Excel row number
    """
    if user_row == 1:
        return header_row + 1
    if user_row <= header_row:
        return header_row + 1
    return user_row


def _set_dynamic_ylim(values_list, pad_ratio=0.10):
    vals = np.concatenate([v[np.isfinite(v)] for v in values_list if v is not None and len(v) > 0])
    if vals.size == 0:
        return

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if np.isclose(vmin, vmax):
        pad = 0.05
    else:
        pad = (vmax - vmin) * pad_ratio

    ymin = vmin - pad
    ymax = vmax + pad

    ymin = max(-0.05, ymin)
    ymax = min(1.10, ymax)

    plt.ylim(ymin, ymax)


# --------------------------
# Embedding blocks (same as training)
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
# Rate net (same as training)
# --------------------------
class MonotoneRateNet(nn.Module):
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

        self.head = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.GELU(),
            nn.Linear(transformer_d_model // 2, 4),
        )
        self.softplus = nn.Softplus()

    def forward(self, t_scalar: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        B = static_features.size(0)

        if t_scalar.numel() == 1:
            t_vec = t_scalar.repeat(B)
        else:
            t_vec = t_scalar
        t_query = t_vec.unsqueeze(-1)

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
        return k


# --------------------------
# Predictor: y = exp(-Integral k dt) via trapezoid rule (same as training)
# --------------------------
class MonotoneIntegralPredictor(nn.Module):
    def __init__(self, rate_net: MonotoneRateNet, base_time_grid: torch.Tensor):
        super().__init__()
        self.rate_net = rate_net
        self.register_buffer("base_time_grid", base_time_grid)

    def forward(self, static_features: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
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
            inc = avg_k * dts

            I = torch.zeros(T, B, 1, device=device)
            I[1:] = torch.cumsum(inc, dim=0)

            idx = torch.searchsorted(times, t_flat)
            Iq = I[idx, torch.arange(B, device=device), :]

        y = torch.exp(-Iq)
        return y


# --------------------------
# Load scaler + model
# --------------------------
def load_scaler_and_model():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found: {BEST_MODEL_PATH}")

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

    state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return scaler, model


# --------------------------
# Prediction helper
# --------------------------
def predict_curve_for_one_row(model, scaler, static_raw: np.ndarray, t_minutes: np.ndarray) -> np.ndarray:
    static_scaled = scaler.transform(static_raw.reshape(1, -1)).astype(np.float32)
    t_norm = (t_minutes / TIME_NORMALIZATION_FACTOR).astype(np.float32)

    N = t_norm.shape[0]
    s_batch = np.repeat(static_scaled, repeats=N, axis=0)

    s_tensor = torch.tensor(s_batch, dtype=torch.float32, device=DEVICE)
    t_tensor = torch.tensor(t_norm.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred = model(s_tensor, t_tensor).squeeze(-1)
        pred = torch.clamp(pred, 0.0, 1.0).cpu().numpy()
    return pred


# --------------------------
# Main plotting
# --------------------------
def plot_rows_prediction_vs_experiment(
    excel_path: str,
    sheet_name: str,
    excel_rows: list[int],
    header_row: int = 1,
    curve_points: int = 250,
    make_plots: bool = True,
):
    scaler, model = load_scaler_and_model()

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if df.empty:
        raise RuntimeError(f"Excel '{excel_path}' sheet '{sheet_name}' is empty.")

    df.columns = [str(x).strip() for x in df.columns]

    for c in STATIC_FEATURE_COLS + TIME_COLS:
        if c not in df.columns:
            raise ValueError(f"Excel missing column: {c}")

    # Precompute curve time
    t_curve = np.linspace(0.0, TIME_NORMALIZATION_FACTOR, curve_points)

    # Collect values for y-lim
    all_y_for_ylim = []

    # ---- Style / fonts (your block + Times New Roman) ----
    if make_plots:
        import matplotlib as mpl

        _old_rc = mpl.rcParams.copy()
        mpl.rcParams.update({
            # "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 24,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "axes.titlesize": 18,
        })

        from matplotlib.lines import Line2D

        # --- nicer figure ---
        fig, ax = plt.subplots(figsize=(13.8, 7.2))

        all_y_for_ylim = []
        size_handles = []

        for i, user_row in enumerate(excel_rows):
            excel_row = normalize_excel_row_input(int(user_row), header_row=header_row)
            df_index = excel_row_to_df_index(excel_row, header_row=header_row)
            if df_index < 0 or df_index >= len(df):
                print(f"[Warn] skip excel_row={excel_row} (df_index={df_index}) out of range.")
                continue

            row = df.iloc[df_index]

            static_raw = row[STATIC_FEATURE_COLS].to_numpy(dtype=np.float32)
            if np.any(np.isnan(static_raw)) or np.any(np.isinf(static_raw)):
                static_raw = np.nan_to_num(static_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            aerosol_size_val = static_raw[IDX_AEROSOL_SIZE]
            try:
                size_str = f"{float(aerosol_size_val):g} μm"
            except Exception:
                size_str = f"{aerosol_size_val} μm"

            exp_vals = pd.to_numeric(row[TIME_COLS], errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(exp_vals)

            pred_curve = predict_curve_for_one_row(model, scaler, static_raw, t_curve)

            color = NATURE_PALETTE[i % len(NATURE_PALETTE)]

            # predicted line (below points)
            ax.plot(
                t_curve, pred_curve,
                linewidth=2.8,
                color=color,
                alpha=0.95,
                zorder=2,
            )

            # measured points (above lines)
            if valid_mask.any():
                ax.scatter(
                    TIME_MINS[valid_mask],
                    exp_vals[valid_mask],
                    s=60,
                    color=color,
                    alpha=0.95,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=3,
                )
                all_y_for_ylim.append(exp_vals[valid_mask])

            all_y_for_ylim.append(pred_curve)

            # one legend entry per aerosol size (color)
            size_handles.append(Line2D([0], [0], color=color, lw=3, label=f"Aerosol size = {size_str}"))

        # ----- axes -----
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Remaining aerosol concentration")
        ax.set_title("Measured vs Predicted Curve")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.30)

        # 推荐固定 y 轴（论文风格，且避免动态导致前段更挤）
        ax.set_ylim(-0.02, 1.02)

        # ----- legends: BOTH outside right -----
        # 给右侧留出空白（很关键，否则 legend 会被裁掉/挤压）
        fig.subplots_adjust(right=0.72)

        # 1) aerosol size color legend（上半块）
        leg1 = ax.legend(
            handles=size_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            frameon=True,
            borderaxespad=0.0,
        )
        ax.add_artist(leg1)

        # 2) symbol legend（下半块，短）
        symbol_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                markerfacecolor='0.25', markeredgecolor='white', markeredgewidth=0.9,
                markersize=9, label="Measured (Exp)"),
            Line2D([0], [0], color='0.25', lw=3, label="Predicted (AI)"),
        ]
        ax.legend(
            handles=symbol_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.55),
            frameon=True,
            borderaxespad=0.0,
        )

        # 保存（不 tight 裁剪，避免又挤回去；bbox_inches="tight" 也可以用但会改变边距）
        fig.savefig("paper_3.2_case4.png", dpi=300)
        plt.show()




# --------------------------
# Entry
# --------------------------
if __name__ == "__main__":
    print(f"DEVICE = {DEVICE}")
    print(f"Excel: {EXCEL_FILE_PATH} | sheet: {SHEET_NAME}")
    print("\n输入说明：")
    print(f"  - 逗号分隔多个行号，例如: 2,10,25")
    print(f"  - 默认（直接回车）使用: 2,3,4")
    print(f"  - 如果你输入 1，会被解释为“第一条数据”（即 Excel 行 {HEADER_ROW+1}）\n")

    s = input("请输入行号列表（comma-separated）[default: 2,3,4]: ").strip()
    if s == "":
        rows = [2, 3, 4]
    else:
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip() != ""]
        rows = []
        for p in parts:
            try:
                rows.append(int(p))
            except ValueError:
                print(f"[Warn] cannot parse '{p}', skip.")
        if len(rows) == 0:
            rows = [2, 3, 4]

    plot_rows_prediction_vs_experiment(
        excel_path=EXCEL_FILE_PATH,
        sheet_name=SHEET_NAME,
        excel_rows=rows,
        header_row=HEADER_ROW,
        curve_points=CURVE_POINTS,
        make_plots=MAKE_PLOTS,
    )