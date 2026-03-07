# -*- coding: utf-8 -*-
"""
Constrained + deterministic user baseline (NO fig / NO matplotlib):
✅ Keep ONLY:
  1) Generate Plotly HTML for 2D Pareto: min [L1=Q*T95, T95]  (optional, deterministic)
  2) Save Excel Sheet2 for:
     - user-specified (dm, Cm, Q) under aerosol=0.198
       ✅ baseline T95 computed ONCE (deterministic, no MC)
       ✅ baseline confidence computed ONCE (engineering score; uses OOD + rel_unc=0)
     - optimize ONLY (dm, Cm) with Q fixed, minimize L1_mean = Q*T95_mean
       subject to Engineering constraints (A-scheme):
         τ1: OOD_max|z| <= 2.5
         τ2: OOD_z_norm <= 4.0
         τ3: confidence >= γ * confidence_baseline   (relative-to-baseline threshold)
         τ4: relative_uncertainty u = T95_std / T95_mean <= 0.02
     - write results + saving to Excel (Sheet2)

New additions vs your last version:
- τ3 is now RELATIVE-TO-BASELINE: conf >= CONF_GAMMA * conf_base
  (more aligned with engineering practice; avoids an arbitrary absolute 0.30)

Notes:
- Fix model loading error: ignore checkpoint 'base_time_grid' mismatch
- Fix AMP warning: torch.amp.autocast('cuda', ...)
- Fix flash/effective attention batch limit: auto clamp batch_size in tp_from_curve_direct_batch()
"""

import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import joblib


# =============================================================================
# Device
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Paths
# =============================================================================
BEST_MODEL_PATH = "best_aerosol_predictor_monotone_Q_Cm_dm_scheme.pth"
SCALER_PATH = "static_scaler.joblib"

EXCEL_FILE_PATH = "/home/user/ZHOU-Wen/Aerosol_DT/Aerosol_data/Aerosol data 1__filtered.xlsx"
SHEET_NAME_IN = "Sheet1"
CASE_COL = "Case"
CASE_VALUE = 1

# Output directory + files
SAVE_HTML = True
FIG_DIR = "fig_section_3_3_Q_dm_Cm(case21)"
PARETO_HTML_PREFIX = "Opt_Pareto_2D_L1_T95"

SAVE_SHEET2_EXCEL = True
SHEET2_EXCEL_NAME = "optimization_results_sheet2.xlsx"  # only Sheet2 is written/replaced


# =============================================================================
# User fixed condition for aerosol=0.198 (BASELINE for saving comparison)
# =============================================================================
USER_AEROSOL_FOR_FIXED_OPT = 0.198
USER_SPEC_DM = 0.78       # dm (μm)
USER_SPEC_CM = 2.7e5      # Cm (P/cm^3)
USER_SPEC_Q  = 2.4        # Q (L/min)

FIXEDQ_N_POINTS = 20000   # kept for sheet printing; stage-1 screening size is set in run_fixed... (N_DET)


# =============================================================================
# Extrapolation + Uncertainty settings
# =============================================================================
# expand_ratio expands ranges by width:
# new_width = old_width*(1 + 2*expand_ratio)
EXPAND_RATIO = 0.50

MC_SAMPLES = 50
CI_ALPHA = 0.05            # 95% CI => [2.5%, 97.5%]

# confidence score hyperparams (engineering score)
CONF_A_UNC = 3.0
CONF_B_OOD = 0.7

# relative-to-baseline confidence ratio (γ)
CONF_GAMMA = 0.90          # τ3: conf(x) >= γ * conf_base  (recommend 0.8~0.95)


# =============================================================================
# Engineering thresholds (A-scheme)  -- hard constraints
# =============================================================================
TAU_MAXABSZ = 2.5     # τ1
TAU_ZNORM   = 4.0     # τ2
TAU_RELUNC  = 0.03    # τ4 (u = std/mean)


# =============================================================================
# Feature columns (MUST match training)
# =============================================================================
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

X_NAME = "Mist mean size (μm)"                  # dm
Y_NAME = "Mist mean concentration (P/cm^3)"     # Cm
AEROSOL_NAME = "Aerosol size (μm)"


# =============================================================================
# Time settings (MUST match training)
# =============================================================================
TIME_NORMALIZATION_FACTOR = 60.0
TIME_POINTS_MIN = np.array([0, 0.1, 0.2, 0.5, 0.8, 1, 3, 5, 10, 15, 20, 30, 40, 50, 60], dtype=float)
TP_SEARCH_STEP_MIN = 1.0
TP_SEARCH_MAX_MIN = 360.0
EPS_Y = 1e-8

BASE_TIME_GRID = torch.tensor(
    sorted(set([float(t) / TIME_NORMALIZATION_FACTOR for t in TIME_POINTS_MIN.tolist()])),
    dtype=torch.float32
)

TP_LEVEL = 95


# =============================================================================
# Baseline override
# =============================================================================
BASELINE_OVERRIDE = {
    "Spray Coverage": 0.48,
    "Spray Flow Rate (L/min)": 2.4,
    "Droplet mean size  (μm)": 302.0,
    "Orfice diameter (mm)": 1.2,
    "Aerosol density (kg/m3)": 5600.0,
    "Contact Angle": 66.0,
}


# =============================================================================
# Aerosol sizes for Pareto HTML
# =============================================================================
AEROSOL_SIZES_TO_RUN = [0.198, 0.407, 0.583, 0.836, 0.898, 1.0]


# =============================================================================
# Optimization options (Pareto sampling)
# =============================================================================
DO_PARETO_HTML = True
TP_CONSTRAINT_MIN = 120.0  # set None to disable
COEF_FLOOR = 1e-6


# =============================================================================
# Model definition (MATCH TRAINING)
# =============================================================================
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


class MonotoneRateNet(nn.Module):
    """
    k(s,t) = k0 + kQ*softplus(Q_scaled) + kC*softplus(Cm_scaled) + kD*softplus(dm_scaled)
    Coef net does NOT see Q/Cm/dm (they are zeroed in scaled space).
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
        t_query = t_vec.unsqueeze(-1)  # [B,1]

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


class MonotoneIntegralPredictor(nn.Module):
    """
    y(t)=exp(-∫_0^t k(s,τ)dτ) by trapezoid on merged grid
    """
    def __init__(self, rate_net: MonotoneRateNet, base_time_grid: torch.Tensor):
        super().__init__()
        self.rate_net = rate_net
        self.register_buffer("base_time_grid", base_time_grid)

    def forward(self, static_features: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
        B = static_features.size(0)
        device = static_features.device
        t_flat = t_query.squeeze(-1)  # [B]

        times = torch.unique(torch.cat([self.base_time_grid.to(device), t_flat])).sort()[0]  # [T]
        T = times.numel()

        static_rep = static_features.repeat(T, 1)         # [T*B, F]
        t_rep = times.repeat_interleave(B)                # [T*B]
        k_ts = self.rate_net(t_rep, static_rep).view(T, B, 1)  # [T,B,1]

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


# =============================================================================
# Basic IO / baseline utils
# =============================================================================
def _ensure_cols_exist(df: pd.DataFrame, cols: list, name="df"):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")


def read_excel_df():
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME_IN)
    df.columns = [str(c).strip() for c in df.columns]
    _ensure_cols_exist(df, STATIC_FEATURE_COLS, "Excel")
    return df


def get_case_df(df: pd.DataFrame, case_value: int) -> pd.DataFrame:
    if CASE_COL not in df.columns:
        raise ValueError(f"Excel has no '{CASE_COL}' column.")
    case_filled = pd.to_numeric(df[CASE_COL], errors="coerce").ffill()
    df2 = df.copy()
    df2["_CaseFilled"] = case_filled
    sub = df2[df2["_CaseFilled"] == case_value].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found for case={case_value}.")
    return sub


def build_baseline_from_case(case_df: pd.DataFrame) -> pd.Series:
    return case_df[STATIC_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").median(axis=0)


def apply_baseline_override(base: pd.Series, override: dict) -> pd.Series:
    b = base.copy()
    for k, v in override.items():
        if k in b.index:
            b[k] = float(v)
    return b


def infer_dm_cm_range_from_data(df: pd.DataFrame, dm_col: str, cm_col: str, expand_ratio=0.0):
    dm = pd.to_numeric(df[dm_col], errors="coerce").to_numpy(dtype=float)
    cm = pd.to_numeric(df[cm_col], errors="coerce").to_numpy(dtype=float)
    dm = dm[np.isfinite(dm)]
    cm = cm[np.isfinite(cm)]
    if dm.size == 0 or cm.size == 0:
        raise ValueError("Cannot infer dm/Cm range from data (all NaN).")

    dm_min, dm_max = float(dm.min()), float(dm.max())
    cm_min, cm_max = float(cm.min()), float(cm.max())

    d_dm = (dm_max - dm_min) * expand_ratio if dm_max > dm_min else 1.0
    d_cm = (cm_max - cm_min) * expand_ratio if cm_max > cm_min else 1.0

    dm_min2 = max(0.0, dm_min - d_dm)
    dm_max2 = max(dm_min2 + 1e-12, dm_max + d_dm)

    cm_min2 = max(0.0, cm_min - d_cm)
    cm_max2 = max(cm_min2 + 1e-12, cm_max + d_cm)

    return (dm_min2, dm_max2), (cm_min2, cm_max2), (dm_min, dm_max), (cm_min, cm_max)


# =============================================================================
# Pareto helpers
# =============================================================================
def pareto_mask_minimize_2d(L1: np.ndarray, Tp: np.ndarray) -> np.ndarray:
    L1 = np.asarray(L1, float)
    Tp = np.asarray(Tp, float)
    n = L1.size
    idx = np.argsort(L1, kind="mergesort")
    Tps = Tp[idx]

    best_tp = np.inf
    keep_sorted = np.zeros(n, dtype=bool)
    for k in range(n):
        if Tps[k] < best_tp - 1e-12:
            keep_sorted[k] = True
            best_tp = Tps[k]
    keep = np.zeros(n, dtype=bool)
    keep[idx] = keep_sorted
    return keep


def pick_compromise_point_2obj(o1: np.ndarray, o2: np.ndarray) -> int:
    o1 = np.asarray(o1, float)
    o2 = np.asarray(o2, float)
    o1n = (o1 - o1.min()) / (o1.max() - o1.min() + 1e-12)
    o2n = (o2 - o2.min()) / (o2.max() - o2.min() + 1e-12)
    d = np.sqrt(o1n**2 + o2n**2)
    return int(np.argmin(d))


# =============================================================================
# OOD score (z-score based on scaler)
# =============================================================================
def compute_ood_scores_from_scaler(scaler, static_mat_raw: np.ndarray):
    X = np.asarray(static_mat_raw, dtype=float)
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        Z = scaler.transform(X)
    else:
        mean = np.asarray(scaler.mean_, dtype=float).reshape(1, -1)
        scale = np.asarray(scaler.scale_, dtype=float).reshape(1, -1)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        Z = (X - mean) / scale

    max_abs_z = np.max(np.abs(Z), axis=1)
    z_norm = np.sqrt(np.sum(Z * Z, axis=1))
    return max_abs_z.astype(np.float32), z_norm.astype(np.float32)


def confidence_score_from_unc_ood(tp_mean, tp_std, max_abs_z, a=3.0, b=0.7):
    tp_mean = np.maximum(tp_mean, 1e-6)
    rel_unc = tp_std / tp_mean
    score_u = np.exp(-a * rel_unc)
    score_d = np.exp(-b * max_abs_z)
    score = score_u * score_d
    return np.clip(score, 0.0, 1.0).astype(np.float32)


# =============================================================================
# T{p} direct-from-curve (GPU batch) - deterministic
# =============================================================================
def tp_from_curve_direct_batch(
    model,
    scaler,
    static_mat_raw: np.ndarray,
    tp_level: float,
    t_init_list: np.ndarray = TIME_POINTS_MIN,
    step_min: float = TP_SEARCH_STEP_MIN,
    t_max_min: float = TP_SEARCH_MAX_MIN,
    batch_size: int = 20000,
    use_amp: bool = True
) -> np.ndarray:
    p = tp_level / 100.0
    y_target = max(1.0 - p, 1e-6)

    # safety clamp to avoid SDPA/efficient-attn batch>65535 (transformer limitation)
    T_eff = int(BASE_TIME_GRID.numel()) + 1
    max_safe_batch = max(256, 65535 // max(1, T_eff))
    batch_size = int(min(int(batch_size), int(max_safe_batch)))

    static_scaled = scaler.transform(static_mat_raw).astype(np.float32)
    static_t = torch.from_numpy(static_scaled).to(DEVICE)
    N = static_t.shape[0]

    Tp = torch.full((N,), float("inf"), device=DEVICE, dtype=torch.float32)
    found = torch.zeros((N,), device=DEVICE, dtype=torch.bool)

    times = np.unique(np.asarray(t_init_list, dtype=float))
    times = np.sort(times)

    @torch.no_grad()
    def predict_y_t(t_min: float) -> torch.Tensor:
        t_norm = float(t_min / TIME_NORMALIZATION_FACTOR)

        y_out = torch.empty((N,), device=DEVICE, dtype=torch.float32)
        amp_on = (use_amp and (DEVICE.type == "cuda"))

        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            sb = static_t[s:e]
            tb = torch.full((e - s, 1), t_norm, device=DEVICE, dtype=sb.dtype)

            if amp_on:
                with torch.amp.autocast("cuda"):
                    yb = model(sb, tb).squeeze(-1)
            else:
                yb = model(sb, tb).squeeze(-1)

            y_out[s:e] = yb.clamp(0.0, 1.0).to(torch.float32)

        return y_out

    t_prev = float(times[0])
    if abs(t_prev) < 1e-12:
        y_prev = torch.ones((N,), device=DEVICE, dtype=torch.float32)
    else:
        y_prev = predict_y_t(t_prev).clamp(EPS_Y, 1.0)

    for t_cur in times[1:]:
        t_cur = float(t_cur)
        y_cur = predict_y_t(t_cur).clamp(EPS_Y, 1.0)

        cross = (~found) & (y_prev > y_target) & (y_cur <= y_target)
        if cross.any():
            denom = (y_cur[cross] - y_prev[cross])
            denom = torch.where(denom.abs() < 1e-12, torch.tensor(-1e-12, device=DEVICE), denom)
            alpha = (y_target - y_prev[cross]) / denom
            alpha = alpha.clamp(0.0, 1.0)
            Tp[cross] = t_prev + alpha * (t_cur - t_prev)
            found[cross] = True

        t_prev = t_cur
        y_prev = y_cur
        if found.all():
            return Tp.cpu().numpy().astype(np.float32)

    t_cur = float(t_prev)
    while (not bool(found.all())) and (t_cur < t_max_min):
        t_next = min(t_cur + step_min, t_max_min)
        y_next = predict_y_t(t_next).clamp(EPS_Y, 1.0)

        cross = (~found) & (y_prev > y_target) & (y_next <= y_target)
        if cross.any():
            denom = (y_next[cross] - y_prev[cross])
            denom = torch.where(denom.abs() < 1e-12, torch.tensor(-1e-12, device=DEVICE), denom)
            alpha = (y_target - y_prev[cross]) / denom
            alpha = alpha.clamp(0.0, 1.0)
            Tp[cross] = t_cur + alpha * (t_next - t_cur)
            found[cross] = True

        t_cur = t_next
        y_prev = y_next

    return Tp.cpu().numpy().astype(np.float32)


# =============================================================================
# MC Dropout for candidates ONLY
# =============================================================================
def tp_from_curve_mc_dropout(
    model,
    scaler,
    static_mat_raw: np.ndarray,
    tp_level: float,
    mc_samples: int = 50,
    ci_alpha: float = 0.05,
    seed0: int = 20260228,
    **tp_kwargs
):
    if mc_samples < 2:
        Tp = tp_from_curve_direct_batch(model, scaler, static_mat_raw, tp_level=tp_level, **tp_kwargs).astype(np.float32)
        return Tp, np.zeros_like(Tp), Tp.copy(), Tp.copy()

    was_training = model.training
    rng = np.random.default_rng(seed0)
    seeds = rng.integers(0, 2**31 - 1, size=int(mc_samples), dtype=np.int64)
    Tp_samps = []

    try:
        model.train()  # enable dropout
        for i in range(int(mc_samples)):
            torch.manual_seed(int(seeds[i]))
            if DEVICE.type == "cuda":
                torch.cuda.manual_seed_all(int(seeds[i]))

            Tp_i = tp_from_curve_direct_batch(
                model, scaler, static_mat_raw,
                tp_level=tp_level,
                **tp_kwargs
            )
            Tp_samps.append(Tp_i.astype(np.float32))
    finally:
        model.train(was_training)
        if not was_training:
            model.eval()

    Tp_samps = np.stack(Tp_samps, axis=0)  # [S, N]
    Tp_mean = Tp_samps.mean(axis=0).astype(np.float32)
    Tp_std = Tp_samps.std(axis=0, ddof=1).astype(np.float32)

    q_low = float(ci_alpha / 2.0)
    q_high = float(1.0 - ci_alpha / 2.0)
    Tp_ci_low = np.quantile(Tp_samps, q_low, axis=0).astype(np.float32)
    Tp_ci_high = np.quantile(Tp_samps, q_high, axis=0).astype(np.float32)

    return Tp_mean, Tp_std, Tp_ci_low, Tp_ci_high


# =============================================================================
# Excel writer: ONLY Sheet2 (create or replace)
# =============================================================================
def write_sheet2_only(excel_path: str, df_sheet2: pd.DataFrame, sheet_name: str = "Sheet2"):
    os.makedirs(os.path.dirname(excel_path), exist_ok=True) if os.path.dirname(excel_path) else None
    file_exists = os.path.exists(excel_path)

    if not file_exists:
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
            df_sheet2.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_sheet2.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Saved Excel: {excel_path} (sheet='{sheet_name}')")


# =============================================================================
# Two-stage constrained optimization (FAST) with relative-to-baseline confidence
# =============================================================================
def run_fixedQ_dmCm_constrained_opt(
    model,
    scaler,
    base: pd.Series,
    dm_range: tuple,
    Cm_range: tuple,
    tp_level: float,
    aerosol_val: float,
    dm_user: float,
    Cm_user: float,
    Q_user: float,
    n_points: int = 20000,        # for sheet printing; used if N_DET=None
    mc_samples: int = 50,
    ci_alpha: float = 0.05,
    N_DET: int | None = 200000,   # deterministic screening points
    TOPK: int = 2000,             # MC only on TOPK candidates (recommend >=2000 for feasibility)
    seed0: int = 20260121,
):
    dm_min, dm_max = float(dm_range[0]), float(dm_range[1])
    Cm_min, Cm_max = float(Cm_range[0]), float(Cm_range[1])

    rng = np.random.default_rng(int(seed0))

    def _single_row_with_Q(base_series: pd.Series, dm: float, Cm: float, Q_lpm: float, aerosol_size: float) -> np.ndarray:
        row = base_series[STATIC_FEATURE_COLS].to_numpy(dtype=np.float32).reshape(1, -1)
        row[0, STATIC_FEATURE_COLS.index(X_NAME)] = float(max(0.0, dm))
        row[0, STATIC_FEATURE_COLS.index(Y_NAME)] = float(max(0.0, Cm))
        row[0, STATIC_FEATURE_COLS.index(AEROSOL_NAME)] = float(aerosol_size)
        row[0, IDX_Q] = float(max(0.0, Q_lpm))
        return row

    # ----------------------------
    # Baseline (user): deterministic T95 + baseline OOD + baseline confidence
    # ----------------------------
    model.eval()
    static_user = _single_row_with_Q(base, dm=dm_user, Cm=Cm_user, Q_lpm=Q_user, aerosol_size=aerosol_val)

    Tp_user = float(tp_from_curve_direct_batch(
        model, scaler, static_user,
        tp_level=tp_level,
        t_init_list=TIME_POINTS_MIN,
        step_min=TP_SEARCH_STEP_MIN,
        t_max_min=TP_SEARCH_MAX_MIN,
        batch_size=4096
    )[0])
    L1_user = float(Q_user * Tp_user)

    # baseline OOD
    ood_user_maxz_arr, ood_user_znorm_arr = compute_ood_scores_from_scaler(scaler, static_user)
    ood_user_maxz = float(ood_user_maxz_arr[0])
    ood_user_znorm = float(ood_user_znorm_arr[0])

    # baseline confidence (rel_unc = 0 because baseline is deterministic here)
    conf_user = float(confidence_score_from_unc_ood(
        np.array([Tp_user], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([ood_user_maxz], dtype=np.float32),
        a=CONF_A_UNC, b=CONF_B_OOD
    )[0])

    conf_threshold_abs = float(CONF_GAMMA * conf_user)

    # ----------------------------
    # Stage 1: deterministic screening
    # ----------------------------
    N_det = int(n_points if (N_DET is None) else N_DET)
    TOPK = int(max(1, min(int(TOPK), N_det)))

    dm_samp = rng.uniform(dm_min, dm_max, size=N_det).astype(np.float32)
    Cm_samp = np.exp(
        rng.uniform(
            np.log(max(Cm_min, 1e-12)),
            np.log(max(Cm_max, max(Cm_min, 1e-12) * (1.0 + 1e-12))),
            size=N_det
        )
    ).astype(np.float32)

    base_row = base[STATIC_FEATURE_COLS].to_numpy(dtype=np.float32).reshape(1, -1)
    static_mat = np.repeat(base_row, repeats=N_det, axis=0)
    static_mat[:, STATIC_FEATURE_COLS.index(X_NAME)] = dm_samp
    static_mat[:, STATIC_FEATURE_COLS.index(Y_NAME)] = Cm_samp
    static_mat[:, STATIC_FEATURE_COLS.index(AEROSOL_NAME)] = float(aerosol_val)
    static_mat[:, IDX_Q] = float(Q_user)

    model.eval()
    Tp_det = tp_from_curve_direct_batch(
        model, scaler, static_mat,
        tp_level=tp_level,
        t_init_list=TIME_POINTS_MIN,
        step_min=TP_SEARCH_STEP_MIN,
        t_max_min=TP_SEARCH_MAX_MIN,
        batch_size=4096
    ).astype(np.float32)

    det_ok = np.isfinite(Tp_det)
    if not np.any(det_ok):
        return {
            "note": "No finite deterministic Tp in Stage-1 screening.",
            "aerosol": float(aerosol_val),
            "dm_user": float(dm_user), "Cm_user": float(Cm_user), "Q_user": float(Q_user),
            "Tp_user": float(Tp_user), "Tp_user_std": 0.0, "rel_unc_user": 0.0,
            "L1_user": float(L1_user),
            "ood_user_maxz": float(ood_user_maxz), "ood_user_znorm": float(ood_user_znorm),
            "confidence_user": float(conf_user),
            "confidence_threshold_abs": float(conf_threshold_abs),
            "dm_opt": np.nan, "Cm_opt": np.nan,
            "Tp_opt_mean": np.nan, "Tp_opt_std": np.nan, "rel_unc_opt": np.nan,
            "Tp_opt_ci_low": np.nan, "Tp_opt_ci_high": np.nan,
            "L1_opt_mean": np.nan, "L1_opt_ci_low": np.nan, "L1_opt_ci_high": np.nan,
            "ood_opt_maxz": np.nan, "ood_opt_znorm": np.nan, "confidence_opt": np.nan,
            "saving_L": np.nan, "saving_pct": np.nan,
            "n_points": int(N_det), "n_feasible": 0,
            "tau_maxabsz": float(TAU_MAXABSZ), "tau_znorm": float(TAU_ZNORM),
            "tau_relunc": float(TAU_RELUNC), "tau_conf_gamma": float(CONF_GAMMA),
        }

    L1_det = float(Q_user) * Tp_det
    idx_ok = np.where(det_ok)[0]
    k = min(TOPK, idx_ok.size)
    L1_det_ok = L1_det[idx_ok]
    topk_pos = np.argpartition(L1_det_ok, kth=k - 1)[:k]
    topk_idx = idx_ok[topk_pos]

    # ----------------------------
    # Stage 2: MC dropout only on TOPK
    # ----------------------------
    static_topk = static_mat[topk_idx, :]
    dm_topk = dm_samp[topk_idx].astype(np.float32)
    Cm_topk = Cm_samp[topk_idx].astype(np.float32)

    ood_maxz, ood_znorm = compute_ood_scores_from_scaler(scaler, static_topk)

    Tp_mean, Tp_std, Tp_ci_low, Tp_ci_high = tp_from_curve_mc_dropout(
        model, scaler, static_topk,
        tp_level=tp_level,
        mc_samples=int(mc_samples),
        ci_alpha=float(ci_alpha),
        t_init_list=TIME_POINTS_MIN,
        step_min=TP_SEARCH_STEP_MIN,
        t_max_min=TP_SEARCH_MAX_MIN,
        batch_size=4096
    )

    finite = np.isfinite(Tp_mean) & np.isfinite(Tp_std) & np.isfinite(Tp_ci_high)
    if not np.any(finite):
        best_det_in_topk = int(np.argmin(np.where(np.isfinite(L1_det[topk_idx]), L1_det[topk_idx], np.inf)))
        k_diag = topk_idx[best_det_in_topk]
        return {
            "note": "No finite MC results in Stage-2 (TOPK). Returned best deterministic in TOPK as diagnostic.",
            "aerosol": float(aerosol_val),
            "dm_user": float(dm_user), "Cm_user": float(Cm_user), "Q_user": float(Q_user),
            "Tp_user": float(Tp_user), "Tp_user_std": 0.0, "rel_unc_user": 0.0,
            "L1_user": float(L1_user),
            "ood_user_maxz": float(ood_user_maxz), "ood_user_znorm": float(ood_user_znorm),
            "confidence_user": float(conf_user),
            "confidence_threshold_abs": float(conf_threshold_abs),

            "dm_opt": float(dm_samp[k_diag]), "Cm_opt": float(Cm_samp[k_diag]),
            "Tp_opt_mean": np.nan, "Tp_opt_std": np.nan, "rel_unc_opt": np.nan,
            "Tp_opt_ci_low": np.nan, "Tp_opt_ci_high": np.nan,
            "L1_opt_mean": float(L1_det[k_diag]), "L1_opt_ci_low": np.nan, "L1_opt_ci_high": np.nan,
            "ood_opt_maxz": np.nan, "ood_opt_znorm": np.nan, "confidence_opt": np.nan,
            "saving_L": float(L1_user - float(L1_det[k_diag])),
            "saving_pct": float(100.0 * (L1_user - float(L1_det[k_diag])) / max(L1_user, 1e-12)),
            "n_points": int(N_det), "n_feasible": 0,
            "tau_maxabsz": float(TAU_MAXABSZ), "tau_znorm": float(TAU_ZNORM),
            "tau_relunc": float(TAU_RELUNC), "tau_conf_gamma": float(CONF_GAMMA),
        }

    Tp_mean_safe = np.maximum(Tp_mean, 1e-6)
    rel_unc = (Tp_std / Tp_mean_safe).astype(np.float32)
    conf = confidence_score_from_unc_ood(Tp_mean, Tp_std, ood_maxz, a=CONF_A_UNC, b=CONF_B_OOD)

    feas = (
        finite
        & (ood_maxz <= float(TAU_MAXABSZ))
        & (ood_znorm <= float(TAU_ZNORM))
        & (conf >= float(conf_threshold_abs))         # τ3: relative-to-baseline
        & (rel_unc <= float(TAU_RELUNC))
    )
    n_feas = int(np.sum(feas))

    L1_mean = float(Q_user) * Tp_mean
    L1_low = float(Q_user) * Tp_ci_low
    L1_high = float(Q_user) * Tp_ci_high

    if n_feas == 0:
        k_best_topk = int(np.argmin(np.where(finite, L1_mean, np.inf)))
        saving_L = float(L1_user - float(L1_mean[k_best_topk]))
        saving_pct = float(100.0 * saving_L / max(L1_user, 1e-12))
        return {
            "note": (
                "No feasible points in TOPK under A-scheme constraints (relative confidence). "
                "Returned best finite TOPK by L1_mean as diagnostic (constraints violated). "
                f"Stage1 N_det={N_det}, TOPK={TOPK}; MC={mc_samples}, CI={(1.0-ci_alpha)*100:.0f}%; expand_ratio={EXPAND_RATIO}"
            ),
            "aerosol": float(aerosol_val),

            "dm_user": float(dm_user), "Cm_user": float(Cm_user), "Q_user": float(Q_user),
            "Tp_user": float(Tp_user), "Tp_user_std": 0.0, "rel_unc_user": 0.0,
            "L1_user": float(L1_user),
            "ood_user_maxz": float(ood_user_maxz), "ood_user_znorm": float(ood_user_znorm),
            "confidence_user": float(conf_user),
            "confidence_threshold_abs": float(conf_threshold_abs),

            "dm_opt": float(dm_topk[k_best_topk]),
            "Cm_opt": float(Cm_topk[k_best_topk]),
            "Tp_opt_mean": float(Tp_mean[k_best_topk]),
            "Tp_opt_std": float(Tp_std[k_best_topk]),
            "rel_unc_opt": float(rel_unc[k_best_topk]),
            "Tp_opt_ci_low": float(Tp_ci_low[k_best_topk]),
            "Tp_opt_ci_high": float(Tp_ci_high[k_best_topk]),
            "L1_opt_mean": float(L1_mean[k_best_topk]),
            "L1_opt_ci_low": float(L1_low[k_best_topk]),
            "L1_opt_ci_high": float(L1_high[k_best_topk]),
            "ood_opt_maxz": float(ood_maxz[k_best_topk]),
            "ood_opt_znorm": float(ood_znorm[k_best_topk]),
            "confidence_opt": float(conf[k_best_topk]),

            "saving_L": float(saving_L),
            "saving_pct": float(saving_pct),

            "n_points": int(N_det),
            "n_feasible": int(n_feas),

            "tau_maxabsz": float(TAU_MAXABSZ),
            "tau_znorm": float(TAU_ZNORM),
            "tau_relunc": float(TAU_RELUNC),
            "tau_conf_gamma": float(CONF_GAMMA),
        }

    k_best = int(np.argmin(np.where(feas, L1_mean, np.inf)))

    L1_opt_mean = float(L1_mean[k_best])
    saving_L = float(L1_user - L1_opt_mean)
    saving_pct = float(100.0 * saving_L / max(L1_user, 1e-12))

    return {
        "note": (
            f"Two-stage constrained opt (relative confidence): Stage1 N_det={N_det} -> TOPK={TOPK}; "
            f"Stage2 MC={mc_samples}, CI={(1.0-ci_alpha)*100:.0f}%. "
            f"Objective=min L1_mean s.t. max|z|<={TAU_MAXABSZ}, z_norm<={TAU_ZNORM}, "
            f"conf>=gamma*conf_base (gamma={CONF_GAMMA}), rel_unc<={TAU_RELUNC}. "
            f"Feasible_in_TOPK={n_feas}/{TOPK}; expand_ratio={EXPAND_RATIO}"
        ),

        "aerosol": float(aerosol_val),

        # baseline
        "dm_user": float(dm_user),
        "Cm_user": float(Cm_user),
        "Q_user": float(Q_user),
        "Tp_user": float(Tp_user),
        "Tp_user_std": 0.0,
        "rel_unc_user": 0.0,
        "L1_user": float(L1_user),
        "ood_user_maxz": float(ood_user_maxz),
        "ood_user_znorm": float(ood_user_znorm),
        "confidence_user": float(conf_user),
        "confidence_threshold_abs": float(conf_threshold_abs),

        # optimized (best feasible in TOPK)
        "dm_opt": float(dm_topk[k_best]),
        "Cm_opt": float(Cm_topk[k_best]),
        "Tp_opt_mean": float(Tp_mean[k_best]),
        "Tp_opt_std": float(Tp_std[k_best]),
        "rel_unc_opt": float(rel_unc[k_best]),
        "Tp_opt_ci_low": float(Tp_ci_low[k_best]),
        "Tp_opt_ci_high": float(Tp_ci_high[k_best]),
        "L1_opt_mean": float(L1_mean[k_best]),
        "L1_opt_ci_low": float(L1_low[k_best]),
        "L1_opt_ci_high": float(L1_high[k_best]),
        "ood_opt_maxz": float(ood_maxz[k_best]),
        "ood_opt_znorm": float(ood_znorm[k_best]),
        "confidence_opt": float(conf[k_best]),

        # saving
        "saving_L": float(saving_L),
        "saving_pct": float(saving_pct),

        "n_points": int(N_det),
        "n_feasible": int(n_feas),

        # thresholds recorded
        "tau_maxabsz": float(TAU_MAXABSZ),
        "tau_znorm": float(TAU_ZNORM),
        "tau_relunc": float(TAU_RELUNC),
        "tau_conf_gamma": float(CONF_GAMMA),
    }


# =============================================================================
# Pareto HTML (2D): deterministic, unchanged (optional)
# =============================================================================
def run_pareto_html_2d(
    model,
    scaler,
    base: pd.Series,
    aerosol_list: list,
    dm_range: tuple,
    Cm_range: tuple,
    tp_level: float = 95.0,
):
    try:
        import plotly.graph_objects as go
        plotly_ok = True
    except Exception:
        plotly_ok = False

    if not plotly_ok:
        print("⚠️ Plotly is not available. Skip HTML generation.")
        return

    os.makedirs(FIG_DIR, exist_ok=True)

    dm_min, dm_max = float(dm_range[0]), float(dm_range[1])
    Cm_min, Cm_max = float(Cm_range[0]), float(Cm_range[1])

    base_Q = float(base["Spray Flow Rate (L/min)"])

    cfg = {
        "N_POINTS": 3600,
        "SEED0": 2026,

        "Q_SCALE_LOW": 0.5,
        "Q_SCALE_HIGH": 3.0,

        "Q_FRAC_NEAR": 0.80,
        "Q_SIGMA_LOG": 0.18,
        "Q_JITTER": 0.04,

        "CM_FRAC_NEAR": 0.65,
        "CM_SIGMA_LOG": 0.35,
        "CM_JITTER": 0.06,

        "DM_BETA_A": 1.6,
        "DM_BETA_B": 3.6,
        "DM_JITTER": 0.03,

        "PLOTLY_MAX_ALL": 60000,
        "PLOTLY_MAX_PARETO": 30000,
    }

    q_min = max(0.05, base_Q * cfg["Q_SCALE_LOW"])
    q_max = max(q_min + 1e-9, base_Q * cfg["Q_SCALE_HIGH"])

    def sample_mixture_around_baseline(low, high, baseline, n, rng,
                                      frac_near=0.7, sigma_log=0.25, jitter=0.03):
        n1 = int(round(n * frac_near))
        n2 = n - n1
        mu = np.log(max(baseline, 1e-12))
        x1 = np.exp(rng.normal(mu, sigma_log, size=n1))
        lo, hi = np.log10(max(low, 1e-12)), np.log10(max(high, 1e-12))
        x2 = 10 ** rng.uniform(lo, hi, size=n2)
        x = np.concatenate([x1, x2])
        if jitter > 0:
            x *= (1.0 + rng.normal(0.0, jitter, size=n))
        return np.clip(x, low, high)

    def sample_beta_range(low, high, n, rng, a=2.0, b=3.0, jitter=0.02):
        u = rng.beta(a, b, size=n)
        x = low + (high - low) * u
        if jitter > 0:
            x *= (1.0 + rng.normal(0.0, jitter, size=n))
        return np.clip(x, low, high)

    def _downsample(rng, *arrays, max_n: int):
        n = len(arrays[0])
        if n <= max_n:
            return arrays
        take = rng.choice(n, size=max_n, replace=False)
        return tuple(np.asarray(a)[take] for a in arrays)

    print("\n" + "=" * 90)
    print("Pareto HTML (2D): decision vars = (Q, dm, Cm)")
    print(f"Objectives: min[L1=Q*T{int(tp_level)}, T{int(tp_level)}]")
    print(f"  baseline Q = {base_Q} L/min")
    print(f"  Q range    = [{q_min:.4g}, {q_max:.4g}]")
    print(f"  dm range   = [{dm_min:.4g}, {dm_max:.4g}]")
    print(f"  Cm range   = [{Cm_min:.4g}, {Cm_max:.4g}]")
    print(f"  N_POINTS per aerosol = {cfg['N_POINTS']}")
    print(f"  Star constraint: T{int(tp_level)} <= {TP_CONSTRAINT_MIN} (None means disabled)")
    print("=" * 90)

    model.eval()

    for ai, aerosol_val in enumerate(aerosol_list):
        aerosol_val = float(aerosol_val)
        rng = np.random.default_rng(cfg["SEED0"] + 1000 * ai)
        N = int(cfg["N_POINTS"])

        Q_samp = sample_mixture_around_baseline(
            q_min, q_max, baseline=base_Q, n=N, rng=rng,
            frac_near=cfg["Q_FRAC_NEAR"], sigma_log=cfg["Q_SIGMA_LOG"], jitter=cfg["Q_JITTER"]
        )

        Cm_baseline = math.sqrt(max(Cm_min, 1e-12) * max(Cm_max, 1e-12))
        if (not np.isfinite(Cm_baseline)) or (Cm_baseline <= 0):
            Cm_baseline = (Cm_min + Cm_max) / 2.0

        Cm_samp = sample_mixture_around_baseline(
            Cm_min, Cm_max, baseline=Cm_baseline, n=N, rng=rng,
            frac_near=cfg["CM_FRAC_NEAR"], sigma_log=cfg["CM_SIGMA_LOG"], jitter=cfg["CM_JITTER"]
        )

        dm_samp = sample_beta_range(
            dm_min, dm_max, n=N, rng=rng,
            a=cfg["DM_BETA_A"], b=cfg["DM_BETA_B"], jitter=cfg["DM_JITTER"]
        )

        base_row = base[STATIC_FEATURE_COLS].to_numpy(dtype=np.float32).reshape(1, -1)
        static_mat_raw = np.repeat(base_row, repeats=N, axis=0)

        static_mat_raw[:, STATIC_FEATURE_COLS.index(X_NAME)] = dm_samp.astype(np.float32)
        static_mat_raw[:, STATIC_FEATURE_COLS.index(Y_NAME)] = Cm_samp.astype(np.float32)
        static_mat_raw[:, STATIC_FEATURE_COLS.index(AEROSOL_NAME)] = float(aerosol_val)
        static_mat_raw[:, IDX_Q] = Q_samp.astype(np.float32)

        Tp = tp_from_curve_direct_batch(
            model, scaler, static_mat_raw,
            tp_level=tp_level,
            t_init_list=TIME_POINTS_MIN,
            step_min=TP_SEARCH_STEP_MIN,
            t_max_min=TP_SEARCH_MAX_MIN
        ).astype(float)

        L1 = Q_samp * Tp

        valid = np.isfinite(Tp) & np.isfinite(L1)
        if not np.any(valid):
            print(f"❌ aerosol={aerosol_val}: no valid points. skip html.")
            continue

        Tp_v = Tp[valid]
        L1_v = L1[valid]
        dm_v = dm_samp[valid]
        Cm_v = Cm_samp[valid]
        Q_v  = Q_samp[valid]

        if (TP_CONSTRAINT_MIN is not None) and np.any(Tp_v <= float(TP_CONSTRAINT_MIN)):
            feas = (Tp_v <= float(TP_CONSTRAINT_MIN))
            Tp_s, L1_s = Tp_v[feas], L1_v[feas]
            dm_s, Cm_s, Q_s = dm_v[feas], Cm_v[feas], Q_v[feas]
        else:
            Tp_s, L1_s = Tp_v, L1_v
            dm_s, Cm_s, Q_s = dm_v, Cm_v, Q_v

        if len(Tp_s) == 0:
            print(f"⚠️ aerosol={aerosol_val}: star pool empty. skip html.")
            continue

        pmask = pareto_mask_minimize_2d(L1_s, Tp_s)
        L1_p, Tp_p = L1_s[pmask], Tp_s[pmask]
        dm_p, Cm_p, Q_p = dm_s[pmask], Cm_s[pmask], Q_s[pmask]

        if len(L1_p) == 0:
            print(f"⚠️ aerosol={aerosol_val}: pareto empty. skip html.")
            continue

        k_star = pick_compromise_point_2obj(L1_p, Tp_p)
        L1_star = float(L1_p[k_star])
        Tp_star = float(Tp_p[k_star])
        dm_star = float(dm_p[k_star])
        Cm_star = float(Cm_p[k_star])
        Q_star  = float(Q_p[k_star])

        (L1_h, Tp_h, dm_h, Cm_h, Q_h) = _downsample(
            rng, L1_v, Tp_v, dm_v, Cm_v, Q_v, max_n=cfg["PLOTLY_MAX_ALL"]
        )
        (L1p_h, Tpp_h) = _downsample(
            rng, L1_p, Tp_p, max_n=cfg["PLOTLY_MAX_PARETO"]
        )

        order = np.argsort(L1p_h)
        L1p_h2 = L1p_h[order]
        Tpp_h2 = Tpp_h[order]

        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=L1_h, y=Tp_h, mode="markers",
            name="all valid points",
            marker=dict(size=6),
            text=[f"Q={Q_h[i]:.4g}, dm={dm_h[i]:.4g}, Cm={Cm_h[i]:.4g}" for i in range(len(L1_h))],
            hoverinfo="x+y+text"
        ))

        fig.add_trace(go.Scatter(
            x=L1p_h2, y=Tpp_h2, mode="lines+markers",
            name="Pareto front (2D)",
            line=dict(width=2),
            marker=dict(size=7),
            hoverinfo="x+y"
        ))

        fig.add_trace(go.Scatter(
            x=[L1_star], y=[Tp_star],
            mode="markers",
            name="compromise point",
            marker=dict(size=14, symbol="diamond"),
            text=[f"Q={Q_star:.4g}, dm={dm_star:.4g}, Cm={Cm_star:.4g}"],
            hoverinfo="x+y+text"
        ))

        fig.update_layout(
            title=f"Pareto (2D): min[L1=Q*T{int(tp_level)}, T{int(tp_level)}] | aerosol={aerosol_val:.4g} μm",
            xaxis_title="L1 (L) = Q*T95",
            yaxis_title=f"T{int(tp_level)} (min)",
            legend=dict(x=1.02, y=0.98, xanchor="left", yanchor="top"),
            margin=dict(l=60, r=220, t=60, b=60),
        )

        html_path = os.path.join(
            FIG_DIR,
            f"{PARETO_HTML_PREFIX}_aerosol_{aerosol_val:.6g}.html"
        )
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        print(f"✅ Saved HTML: {html_path}")


# =============================================================================
# Load scaler & model
# =============================================================================
def load_scaler(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found: {path}")
    scaler = joblib.load(path)
    print(f"✅ Loaded scaler: {os.path.abspath(path)}")
    return scaler


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found: {path}")

    rate_net = MonotoneRateNet(
        num_static_features=NUM_STATIC_FEATURES,
        static_embed_dim=1024,
        time_embed_dim=128,
        transformer_d_model=2048,
        n_head=4,
        num_transformer_layers=7,
        transformer_dropout=0.02,
        idx_Q=IDX_Q,
        idx_Cm=IDX_CM,
        idx_dm=IDX_DM,
        coef_floor=COEF_FLOOR,
    ).to(DEVICE)

    model = MonotoneIntegralPredictor(rate_net, base_time_grid=BASE_TIME_GRID).to(DEVICE)

    state = torch.load(path, map_location=DEVICE)

    if isinstance(state, dict) and ("base_time_grid" in state):
        print("⚠️ Removing 'base_time_grid' from checkpoint to avoid size mismatch.")
        state.pop("base_time_grid", None)

    if isinstance(state, dict) and ("model_state_dict" in state) and isinstance(state["model_state_dict"], dict):
        sd = state["model_state_dict"]
        if "base_time_grid" in sd:
            print("⚠️ Removing 'base_time_grid' from checkpoint['model_state_dict'] to avoid size mismatch.")
            sd.pop("base_time_grid", None)
        state = sd

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"⚠️ Missing keys (ignored): {missing[:10]}{' ...' if len(missing)>10 else ''}")
    if len(unexpected) > 0:
        print(f"⚠️ Unexpected keys (ignored): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    model.eval()
    print(f"✅ Loaded model weights: {os.path.abspath(path)}")
    return model


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"Running on device: {DEVICE}")

    print("\n[PATH CHECK]")
    print(f"  BEST_MODEL_PATH abs = {os.path.abspath(BEST_MODEL_PATH)}")
    print(f"  SCALER_PATH abs     = {os.path.abspath(SCALER_PATH)}")
    print(f"  OUT_DIR             = {os.path.abspath(FIG_DIR)}")

    scaler = load_scaler(SCALER_PATH)
    model = load_model(BEST_MODEL_PATH)

    df_all = read_excel_df()
    case_df = get_case_df(df_all, CASE_VALUE)

    base = build_baseline_from_case(case_df)
    base = apply_baseline_override(base, BASELINE_OVERRIDE)

    dm_range, Cm_range, dm_raw, cm_raw = infer_dm_cm_range_from_data(
        df_all, X_NAME, Y_NAME, expand_ratio=float(EXPAND_RATIO)
    )

    print("\n[Ranges]")
    print(f"  Raw dm range = [{dm_raw[0]:.6g}, {dm_raw[1]:.6g}]")
    print(f"  Raw Cm range = [{cm_raw[0]:.6g}, {cm_raw[1]:.6g}]")
    print(f"  Expanded dm range (expand_ratio={EXPAND_RATIO}) = [{dm_range[0]:.6g}, {dm_range[1]:.6g}]")
    print(f"  Expanded Cm range (expand_ratio={EXPAND_RATIO}) = [{Cm_range[0]:.6g}, {Cm_range[1]:.6g}]")

    print("\n[Engineering constraints (A-scheme)]")
    print(f"  τ1: max|z| <= {TAU_MAXABSZ}")
    print(f"  τ2: z_norm <= {TAU_ZNORM}")
    print(f"  τ3: confidence >= gamma * confidence_baseline  (gamma={CONF_GAMMA})")
    print(f"  τ4: rel_unc (std/mean) <= {TAU_RELUNC}")
    print(f"  MC samples={MC_SAMPLES}, CI={(1.0-CI_ALPHA)*100:.0f}% (candidates only)")
    print(f"  Two-stage: N_det=200000, TOPK=2000 (edit inside run_fixedQ... if needed)")

    # 1) Pareto HTML (optional)
    if DO_PARETO_HTML and SAVE_HTML:
        run_pareto_html_2d(
            model=model,
            scaler=scaler,
            base=base,
            aerosol_list=[float(v) for v in AEROSOL_SIZES_TO_RUN],
            dm_range=dm_range,
            Cm_range=Cm_range,
            tp_level=float(TP_LEVEL),
        )

    # 2) Constrained optimization + saving vs user baseline
    res = run_fixedQ_dmCm_constrained_opt(
        model=model,
        scaler=scaler,
        base=base,
        dm_range=dm_range,
        Cm_range=Cm_range,
        tp_level=float(TP_LEVEL),
        aerosol_val=float(USER_AEROSOL_FOR_FIXED_OPT),
        dm_user=float(USER_SPEC_DM),
        Cm_user=float(USER_SPEC_CM),
        Q_user=float(USER_SPEC_Q),
        n_points=int(FIXEDQ_N_POINTS),
        mc_samples=int(MC_SAMPLES),
        ci_alpha=float(CI_ALPHA),
        N_DET=200000,
        TOPK=2000,
    )

    df_sheet2 = pd.DataFrame([{
        "aerosol (μm)": res.get("aerosol", USER_AEROSOL_FOR_FIXED_OPT),
        "expand_ratio": EXPAND_RATIO,
        "n_points": FIXEDQ_N_POINTS,
        "mc_samples (candidates)": MC_SAMPLES,
        "CI_level(%)": (1.0 - CI_ALPHA) * 100.0,

        "tau_max|z|": TAU_MAXABSZ,
        "tau_z_norm": TAU_ZNORM,
        "tau_rel_unc": TAU_RELUNC,
        "tau_conf_gamma": CONF_GAMMA,
        "tau_conf_abs = gamma*conf_base": res.get("confidence_threshold_abs", np.nan),

        "dm_user (μm)": res.get("dm_user", USER_SPEC_DM),
        "Cm_user (P/cm^3)": res.get("Cm_user", USER_SPEC_CM),
        "Q_user (L/min)": res.get("Q_user", USER_SPEC_Q),
        f"T{int(TP_LEVEL)}_user (min) [det]": res.get("Tp_user", np.nan),
        "L1_user = Q*T95 [det]": res.get("L1_user", np.nan),
        "OOD_user_max|z|": res.get("ood_user_maxz", np.nan),
        "OOD_user_z_norm": res.get("ood_user_znorm", np.nan),
        "confidence_user(0-1)": res.get("confidence_user", np.nan),
        "rel_unc_user": res.get("rel_unc_user", np.nan),

        "dm_opt (μm)": res.get("dm_opt", np.nan),
        "Cm_opt (P/cm^3)": res.get("Cm_opt", np.nan),
        f"T{int(TP_LEVEL)}_opt_mean (min)": res.get("Tp_opt_mean", np.nan),
        f"T{int(TP_LEVEL)}_opt_std (min)": res.get("Tp_opt_std", np.nan),
        f"T{int(TP_LEVEL)}_opt_CI_low (min)": res.get("Tp_opt_ci_low", np.nan),
        f"T{int(TP_LEVEL)}_opt_CI_high (min)": res.get("Tp_opt_ci_high", np.nan),
        "L1_opt_mean = Q*T95": res.get("L1_opt_mean", np.nan),
        "L1_opt_CI_low": res.get("L1_opt_ci_low", np.nan),
        "L1_opt_CI_high": res.get("L1_opt_ci_high", np.nan),
        "OOD_opt_max|z|": res.get("ood_opt_maxz", np.nan),
        "OOD_opt_z_norm": res.get("ood_opt_znorm", np.nan),
        "confidence_opt(0-1)": res.get("confidence_opt", np.nan),
        "rel_unc_opt": res.get("rel_unc_opt", np.nan),

        "saving_L (L)": res.get("saving_L", np.nan),
        "saving_%": res.get("saving_pct", np.nan),

        "n_feasible": res.get("n_feasible", np.nan),
        "note": res.get("note", "")
    }])

    if SAVE_SHEET2_EXCEL:
        os.makedirs(FIG_DIR, exist_ok=True)
        out_xlsx = os.path.join(FIG_DIR, SHEET2_EXCEL_NAME)
        write_sheet2_only(out_xlsx, df_sheet2, sheet_name="Sheet2")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()