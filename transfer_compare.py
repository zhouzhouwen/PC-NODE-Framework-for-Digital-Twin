# -*- coding: utf-8 -*-
"""
迁移学习基准测试
- 数据来源：Excel 宽表，时间列形如 "0min","1min",...（可能存在缺失单元格）
- 加载预训练的 单调( monotone ) PC-NODE 模型 + scaler
The following transfer-learning strategies were compared:
(a) method0_eval_only, where the pretrained model was evaluated without any parameter update;
(b) methodA_head_only, where only the prediction head was fine-tuned while all other modules were kept frozen;
(c) methodB_head_plus_last_encoderK, where the prediction head and the last K Transformer encoder layers were jointly fine-tuned;
(d) methodC_last_encoderK_only, where only the last K encoder layers were fine-tuned and the prediction head was kept frozen, so that adaptation was primarily performed in the representation space;
(e) methodD_add_3layer_mlp_calibrator, where a three-layer MLP calibrator was introduced to apply residual correction to the head outputs;
(f) methodE_static_embedder_plus_head, where both the static-feature embedder and the prediction head were fine-tuned;
(g) methodF_feature_affine_input_calibration, where a learnable feature-wise affine transformation was applied to the raw static inputs prior to scaling to achieve input-level calibration; and
(h) methodG_adapter, where lightweight adapter modules were injected into the last K encoder layers via forward hooks, enabling encoder adaptation while most pretrained parameters were kept fixed.


输出目录结构：
transfer_runs/run_xxx/
  ├─ run_config.json
  ├─ SiC/
  │    ├─ split_groups.csv
  │    ├─ summary_metrics.xlsx
  │    └─ method*/...
  ├─ CeO2/
  ├─ TiO2/
  └─ summary_metrics_ALL_conditions.xlsx
"""

import os
import re
import json
import time
import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import joblib


# =============================================================================
# 0) 用户参数（在这里修改）
# =============================================================================
# ----- 路径配置 -----
EXCEL_FILE_PATH = "/home/user/ZHOU-Wen/Aerosol_DT/Aerosol_data/Aerosol data 1__filtered.xlsx"
SHEET_NAME = "Sheet1"

BEST_MODEL_PATH = "best_aerosol_predictor_monotone_Q_Cm_dm_scheme_ZrO2.pth"
SCALER_PATH = "static_scaler.joblib"

# ----- 按行号段划分工况（数据行，表头之后按 0-based 计数）-----
# ✅请按你的截图/真实分段填写（下面是你截图推断的常见分段）
CONDITION_RANGES = {
    "SiC":  (1367, 1410),
    "CeO2": (1411, 1454),
    "TiO2": (1455, 1501),
}

# ----- 数据切分 -----
SPLIT_SEED = 2026
TRAIN_RATIO = 0.20
VAL_RATIO = 0.00
TEST_RATIO = 0.80
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# ----- 训练设置 -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True  # 只有在 cuda 时才会启用 AMP

BATCH_SIZE = 5
MAX_EPOCHS = 500
LR_HEAD = 1e-4
LR_ENCODER = 5e-5
LR_ADAPTER = 5e-4
LR_AFFINE = 5e-4
WEIGHT_DECAY = 1e-4

GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0  # 服务器上常用 0 更稳

# Encoder/Adapter 相关设置
LAST_ENCODER_K = 1  # 用于 methodB 和 methodG

# ----- 评估报告阈值（用于 clamp 以避免极小值导致相对误差爆炸）-----
EVAL_THRESHOLD = 1e-5

# ----- 输出目录 -----
RUNS_DIR = "transfer_runs"
RUN_NAME = None  # 若设置为字符串则固定名字，否则自动用时间戳生成


# =============================================================================
# 1) 静态特征列 / 时间设置（必须与预训练时一致）
# =============================================================================
STATIC_FEATURE_COLS = [
    "Spray Coverage",
    "Spray Flow Rate (L/min)",
    "Droplet mean size  (μm)",
    "Orfice diameter (mm)",
    "Mist mean size (μm)",
    "Mist mean concentration (P/cm^3)",
    "Aerosol density (kg/m3)",
    "Contact Angle",
    "Aerosol size (μm)",
]
NUM_STATIC_FEATURES = len(STATIC_FEATURE_COLS)

# 单调约束相关特征索引（必须与你训练 monotone 方案一致）
IDX_Q  = STATIC_FEATURE_COLS.index("Spray Flow Rate (L/min)")
IDX_DM = STATIC_FEATURE_COLS.index("Mist mean size (μm)")
IDX_CM = STATIC_FEATURE_COLS.index("Mist mean concentration (P/cm^3)")

# 时间归一化：模型使用 t_norm = t_min/60
TIME_NORMALIZATION_FACTOR = 60.0


# =============================================================================
# 2) 可复现性设置
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# 3) 你的 evaluate_metrics（保持原样，仅做“打包”）
# =============================================================================
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

    # t=0 => 为了误差统计，将 true 视为 1.0（用于相对误差等）
    t0_mask = np.isclose(inputs_time, 0.0)
    num_t0 = int(np.sum(t0_mask))
    if num_t0 > 0:
        y_true[t0_mask] = 1.0
        print(f"[评估] 对 {num_t0} 个 t=0 的样本，将 y_true 设为 1.0（用于误差计算）。")

    print(f"\n--- Evaluation (N={total_samples}) ---")
    print(f"阈值 threshold: {threshold}")
    print("正在生成 Excel 报告" + (" + plots..." if make_plots else "..."))

    # 对 true/pred 进行阈值截断，避免除以接近 0
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

    # 保存详细 Excel
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
    print(f"✅ 已保存 Excel: {save_path}")

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

    return {
        "r2": float(r2),
        "wmape_percent": float(wmape),
        "nrmse": float(nrmse),
        "mape_percent": float(mape),
        "pm_mean": float(pm_mean),
        "pm_std": float(pm_std),
        "rmspe": float(rmspe),
        "rmse": float(rmse),
        "mae": float(np.mean(abs_diff_raw)),
        "mse": float(np.mean(sq_err_raw)),
        "max_abs_error": float(np.max(abs_diff_raw)) if total_samples > 0 else float("nan"),
        "count": int(total_samples),
    }


# =============================================================================
# 4) 模型（与最新 monotone scheme 保持一致）
# =============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间位置编码（embed_dim 必须为偶数）"""
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
    """静态特征嵌入器：MLP (F -> 2D -> D)"""
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
    单调速率网络：
    k(s,t) = k0 + kQ*softplus(Q_scaled) + kC*softplus(Cm_scaled) + kD*softplus(dm_scaled)

    其中 k0,kQ,kC,kD 通过 Transformer encoder 的 time token 输出经 head 得到，
    并用 softplus 保证非负，从而保证整体速率 k >= 0。
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
        coef_floor: float = 1e-6,
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

        # 用于生成系数的静态输入：把 Q/Cm/dm 置 0（避免系数网络“偷看”单调特征值）
        static_for_coeff = static_features.clone()
        static_for_coeff[:, self.idx_Q] = 0.0
        static_for_coeff[:, self.idx_Cm] = 0.0
        static_for_coeff[:, self.idx_dm] = 0.0

        s_emb = self.static_embedder(static_for_coeff)
        t_emb = self.time_embedder(t_query.float())

        s_proj = self.proj_static(s_emb)
        t_proj = self.proj_time(t_emb)

        # 两个 token：static token + time token
        seq = torch.stack([s_proj, t_proj], dim=1)  # [B,2,D]
        out = self.encoder(seq)
        time_tok = out[:, 1, :]  # [B,D]

        raw = self.head(time_tok)  # [B,4]
        k0 = self.softplus(raw[:, 0:1])
        kQ = self.softplus(raw[:, 1:2])
        kC = self.softplus(raw[:, 2:3]) + self.coef_floor
        kD = self.softplus(raw[:, 3:4]) + self.coef_floor

        # 单调特征取出来（这些通常已是 scaler 后的值）
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
    积分形式预测器：
    y(t)=exp(-∫_0^t k(s,τ)dτ)
    使用 trapezoid（梯形积分）在“base_time_grid + t_query 合并网格”上计算，
    对不同 t_query 向量化。
    """
    def __init__(self, rate_net: MonotoneRateNet, base_time_grid: torch.Tensor):
        super().__init__()
        self.rate_net = rate_net
        self.register_buffer("base_time_grid", base_time_grid)

    def forward(self, static_features: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
        B = static_features.size(0)
        device = static_features.device
        t_flat = t_query.squeeze(-1)  # [B]

        # 合并 base grid 与 query 时间点
        times = torch.unique(torch.cat([self.base_time_grid.to(device), t_flat])).sort()[0]  # [T]
        T = times.numel()

        static_rep = static_features.repeat(T, 1)                # [T*B, F]
        t_rep = times.repeat_interleave(B)                       # [T*B]
        k_ts = self.rate_net(t_rep, static_rep).view(T, B, 1)    # [T,B,1]

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
        return y  # [B,1]


DEFAULT_BASE_TIME_GRID = torch.tensor(
    sorted(set([0.0, 1.0/60.0, 3.0/60.0, 5.0/60.0, 10.0/60.0, 15.0/60.0,
                20.0/60.0, 30.0/60.0, 40.0/60.0, 50.0/60.0, 60.0/60.0])),
    dtype=torch.float32
)


def load_scaler(path: str):
    """加载 sklearn scaler（通常是 StandardScaler）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)


def load_model(path: str) -> MonotoneIntegralPredictor:
    """加载预训练模型权重到 MonotoneIntegralPredictor"""
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
        coef_floor=1e-6,
    ).to(DEVICE)

    model = MonotoneIntegralPredictor(rate_net, base_time_grid=DEFAULT_BASE_TIME_GRID).to(DEVICE)

    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "base_time_grid" in state:
        state.pop("base_time_grid")

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "base_time_grid"]
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

    model.eval()
    return model


# =============================================================================
# 5) 可微分 scaler（让 methodF 能学习“先仿射，再缩放”）
# =============================================================================
@dataclass
class TorchScaler:
    kind: str  # "standard" 或 "unknown"
    mean: torch.Tensor
    scale: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind != "standard":
            raise RuntimeError("TorchScaler 仅支持 StandardScaler 类似的 transform。")
        return (x - self.mean) / self.scale


def build_torch_scaler(sklearn_scaler) -> TorchScaler:
    if hasattr(sklearn_scaler, "mean_") and hasattr(sklearn_scaler, "scale_"):
        mean = torch.tensor(np.array(sklearn_scaler.mean_, dtype=np.float32), device=DEVICE)
        scale = torch.tensor(np.array(sklearn_scaler.scale_, dtype=np.float32), device=DEVICE)
        scale = torch.clamp(scale, min=1e-12)
        return TorchScaler(kind="standard", mean=mean, scale=scale)
    return TorchScaler(kind="unknown", mean=torch.empty(0), scale=torch.empty(0))


# =============================================================================
# 6) 数据处理：Excel 宽表 -> 长表；按原始行分组切分；处理缺失时间点
# =============================================================================
def _ensure_cols(df: pd.DataFrame, cols: List[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Excel missing columns: {miss}")


def parse_time_min(col: str) -> Optional[float]:
    """解析时间列名：'0min','1min','60min'（允许空格/大小写）"""
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*min\s*$", str(col).strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


def load_excel_slice_to_long(
    excel_path: str,
    sheet_name: str,
    row_start: int,
    row_end_inclusive: int,
) -> pd.DataFrame:
    """从 Excel 读取指定行范围，宽表转长表（每行*每个时间点 => 一条样本）"""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    _ensure_cols(df, STATIC_FEATURE_COLS)

    # 行切片（对数据行 0-based）
    if row_start < 0:
        row_start = 0
    row_end_inclusive = min(row_end_inclusive, len(df) - 1)
    df_slice = df.iloc[row_start:row_end_inclusive + 1].copy().reset_index(drop=False)
    df_slice.rename(columns={"index": "_orig_row_index"}, inplace=True)

    # 找到时间列
    time_cols = []
    time_map = {}
    for c in df_slice.columns:
        t = parse_time_min(c)
        if t is not None:
            time_cols.append(c)
            time_map[c] = t

    if len(time_cols) == 0:
        raise ValueError("在 Excel 切片中未找到形如 '0min','1min',... 的时间列。")

    # melt 宽转长
    long_df = df_slice.melt(
        id_vars=["_orig_row_index"] + STATIC_FEATURE_COLS,
        value_vars=time_cols,
        var_name="_time_col",
        value_name="y",
    )
    long_df["time_min"] = long_df["_time_col"].map(time_map).astype(float)
    long_df.drop(columns=["_time_col"], inplace=True)

    # 转数值
    for c in STATIC_FEATURE_COLS:
        long_df[c] = pd.to_numeric(long_df[c], errors="coerce")
    long_df["y"] = pd.to_numeric(long_df["y"], errors="coerce")

    # 去除缺失 y 或缺失静态特征/时间的样本
    long_df = long_df.dropna(subset=["y"] + STATIC_FEATURE_COLS + ["time_min"]).reset_index(drop=True)

    # y 限制到 [0,1]（保险起见）
    long_df["y"] = long_df["y"].clip(lower=0.0, upper=1.0)

    return long_df


def split_by_group(
    long_df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按原始行索引 _orig_row_index 分组切分，
    同一原始行的所有时间点样本必须全部在同一 split 中，避免泄漏。
    """
    groups = long_df["_orig_row_index"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    test_g = set(groups[n_train + n_val:])

    train_df = long_df[long_df["_orig_row_index"].isin(train_g)].copy()
    val_df = long_df[long_df["_orig_row_index"].isin(val_g)].copy()
    test_df = long_df[long_df["_orig_row_index"].isin(test_g)].copy()

    split_groups = pd.DataFrame({
        "_orig_row_index": groups,
        "split": (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test),
    })

    return train_df, val_df, test_df, split_groups


# =============================================================================
# 7) Dataset / DataLoader
# =============================================================================
class LongDataset(Dataset):
    """长表数据集：返回 (x_static, t_min, y)"""
    def __init__(self, df_long: pd.DataFrame):
        self.df = df_long.reset_index(drop=True)
        self.x = self.df[STATIC_FEATURE_COLS].to_numpy(dtype=np.float32)
        self.t = self.df["time_min"].to_numpy(dtype=np.float32)
        self.y = self.df["y"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.y[idx]


def make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )


# =============================================================================
# 8) 方法模块：输入仿射 + 校准器 MLP + Adapter hooks
# =============================================================================
class FeatureAffine(nn.Module):
    """逐特征仿射：x' = gamma*x + beta（恒等初始化）"""
    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma + self.beta


class ResidualHeadCalibrator(nn.Module):
    """3层 MLP：对 head 输出做残差修正（raw += delta(time_tok)）"""
    def __init__(self, d_model: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def forward(self, time_tok: torch.Tensor) -> torch.Tensor:
        return self.net(time_tok)


class TimeTokenAdapter(nn.Module):
    """Adapter：time = time + W_up(GELU(W_down(time)))"""
    def __init__(self, d_model: int, bottleneck: int = 64, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.up(self.act(self.down(x))))


class EncoderAdapterHooks:
    """
    通过 forward hook 给 TransformerEncoder 的最后 K 层加 adapter。
    hook：out[:,1,:] += adapter(out[:,1,:])（只改 time token）
    """
    def __init__(self, encoder: nn.TransformerEncoder, last_k: int, d_model: int, bottleneck: int = 64):
        self.encoder = encoder
        layers = list(getattr(encoder, "layers", []))
        if len(layers) == 0:
            raise RuntimeError("TransformerEncoder 没有 .layers；无法挂 adapter hooks。")
        self.layer_indices = list(range(max(0, len(layers) - last_k), len(layers)))
        self.adapters = nn.ModuleDict({str(i): TimeTokenAdapter(d_model, bottleneck=bottleneck) for i in self.layer_indices})
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self):
        layers = list(self.encoder.layers)

        def make_hook(layer_idx: int):
            adapter = self.adapters[str(layer_idx)]
            def hook(module, inp, out):
                if not isinstance(out, torch.Tensor):
                    return out
                if out.dim() != 3 or out.size(1) < 2:
                    return out
                out2 = out.clone()
                time_tok = out2[:, 1, :]
                out2[:, 1, :] = time_tok + adapter(time_tok)
                return out2
            return hook

        for li in self.layer_indices:
            h = layers[li].register_forward_hook(make_hook(li))
            self._handles.append(h)

    def detach(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def parameters(self):
        return self.adapters.parameters()


# =============================================================================
# 9) 训练 / 预测 / 评估工具函数
# =============================================================================
def count_trainable_params(model: Optional[nn.Module]) -> int:
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


@torch.inference_mode()
def predict_batches(
    model: MonotoneIntegralPredictor,
    torch_scaler: TorchScaler,
    loader: DataLoader,
    feature_affine: Optional[FeatureAffine] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在 loader 上预测，返回 (y_true,y_pred,x_static_raw,t_min)"""
    model.eval()
    preds, trues, statics, times = [], [], [], []

    for xb, tb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        tb = tb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        x_raw = xb
        if feature_affine is not None:
            x_raw = feature_affine(x_raw)

        if torch_scaler.kind != "standard":
            raise RuntimeError("Scaler 不是 StandardScaler-like。请适配 TorchScaler。")
        x_scaled = torch_scaler.transform(x_raw)

        t_norm = (tb / TIME_NORMALIZATION_FACTOR).view(-1, 1)
        y_pred = model(x_scaled, t_norm).squeeze(-1).clamp(0.0, 1.0)

        preds.append(y_pred.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
        statics.append(xb.detach().cpu().numpy())  # 报告用 RAW（仿射前）静态特征
        times.append(tb.detach().cpu().numpy())

    return (
        np.concatenate(trues, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(statics, axis=0),
        np.concatenate(times, axis=0),
    )


def train_one_method(
    method_name: str,
    model: MonotoneIntegralPredictor,
    torch_scaler: TorchScaler,
    train_loader: DataLoader,
    test_loader: DataLoader,
    out_dir: str,
    max_epochs: int,
    optimizer: torch.optim.Optimizer,
    feature_affine: Optional[FeatureAffine] = None,
    adapter_hooks: Optional[EncoderAdapterHooks] = None,
) -> Dict:
    """训练某一种迁移方法（无验证集），并在 test 上评估输出各种文件"""
    os.makedirs(out_dir, exist_ok=True)

    if adapter_hooks is not None:
        adapter_hooks.attach()

    model.train()
    if feature_affine is not None:
        feature_affine.train()

    log_rows = []
    start = time.perf_counter()

    if torch_scaler.kind != "standard":
        raise RuntimeError("Scaler 在可微模式下不支持（期望 StandardScaler-like）。")

    amp_on = (USE_AMP and (DEVICE.type == "cuda"))
    scaler_amp = torch.cuda.amp.GradScaler(enabled=amp_on)
    loss_fn = nn.MSELoss()

    for epoch in range(1, max_epochs + 1):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        ep_start = time.perf_counter()

        total_loss = 0.0
        total_n = 0

        for xb, tb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            tb = tb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            x_raw = xb
            if feature_affine is not None:
                x_raw = feature_affine(x_raw)

            x_scaled = torch_scaler.transform(x_raw)
            t_norm = (tb / TIME_NORMALIZATION_FACTOR).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            if amp_on:
                with torch.cuda.amp.autocast():
                    yp = model(x_scaled, t_norm).squeeze(-1).clamp(0.0, 1.0)
                    loss = loss_fn(yp, yb)
                scaler_amp.scale(loss).backward()
                if GRAD_CLIP_NORM is not None:
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    if feature_affine is not None:
                        torch.nn.utils.clip_grad_norm_(feature_affine.parameters(), GRAD_CLIP_NORM)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                yp = model(x_scaled, t_norm).squeeze(-1).clamp(0.0, 1.0)
                loss = loss_fn(yp, yb)
                loss.backward()
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    if feature_affine is not None:
                        torch.nn.utils.clip_grad_norm_(feature_affine.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            bs = int(xb.size(0))
            total_loss += float(loss.detach().item()) * bs
            total_n += bs

        train_mse = total_loss / max(1, total_n)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - ep_start

        log_rows.append({"epoch": epoch, "train_mse": train_mse, "epoch_time_sec": ep_time})

        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(f"[{method_name}] epoch {epoch:4d}/{max_epochs} | train_mse={train_mse:.6e} | {ep_time:.2f}s")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    pd.DataFrame(log_rows).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)

    # test 评估
    y_true, y_pred, x_static_raw, t_min = predict_batches(model, torch_scaler, test_loader, feature_affine=feature_affine)

    eval_xlsx = os.path.join(out_dir, "evaluation_details.xlsx")
    metrics = evaluate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        inputs_static=x_static_raw,
        inputs_time=t_min,
        feature_names=STATIC_FEATURE_COLS,
        threshold=EVAL_THRESHOLD,
        save_path=eval_xlsx,
        make_plots=False,
    )

    save_obj = {
        "model_state_dict": model.state_dict(),
        "feature_affine_state_dict": feature_affine.state_dict() if feature_affine is not None else None,
        "adapter_state_dict": adapter_hooks.adapters.state_dict() if adapter_hooks is not None else None,
        "config": {"method": method_name, "max_epochs": max_epochs, "train_time_sec": train_time},
    }
    torch.save(save_obj, os.path.join(out_dir, "model_state.pt"))

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps({**metrics, "train_time_sec": train_time}, indent=2, ensure_ascii=False))

    if adapter_hooks is not None:
        adapter_hooks.detach()

    return {
        "method": method_name,
        "train_time_sec": float(train_time),
        "test_count": int(metrics["count"]),
        "test_mse": float(metrics["mse"]),
        "test_mae": float(metrics["mae"]),
        "test_rmse": float(metrics["rmse"]),
        "test_r2": float(metrics["r2"]),
        "test_max_abs_error": float(metrics["max_abs_error"]),
        "test_mape_percent": float(metrics["mape_percent"]),
        "test_wmape_percent": float(metrics["wmape_percent"]),
        "test_nrmse": float(metrics["nrmse"]),
        "test_pm_mean": float(metrics["pm_mean"]),
        "test_pm_std": float(metrics["pm_std"]),
        "test_rmspe": float(metrics["rmspe"]),
        "trainable_params": int(count_trainable_params(model) + count_trainable_params(feature_affine)),
        "epochs": int(max_epochs),
    }


# =============================================================================
# 10) 构建每种方法（冻结/解冻 + 加额外模块）
# =============================================================================
def configure_method(
    base_model: MonotoneIntegralPredictor,
    method: str,
) -> Tuple[MonotoneIntegralPredictor, Optional[torch.optim.Optimizer], Optional[FeatureAffine], Optional[EncoderAdapterHooks], int]:
    """
    返回：
      model, optimizer, feature_affine, adapter_hooks, max_epochs
    """
    model = copy.deepcopy(base_model).to(DEVICE)

    # 默认：冻结全部参数
    set_requires_grad(model, False)

    feature_affine = None
    adapter_hooks = None
    epochs = MAX_EPOCHS

    rate_net = model.rate_net
    encoder = rate_net.encoder

    if method == "method0_eval_only":
        return model, None, None, None, 0

    if method == "methodA_head_only":
        set_requires_grad(rate_net.head, True)
        opt = torch.optim.AdamW(rate_net.head.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
        return model, opt, None, None, epochs

    if method == "methodB_head_plus_last_encoderK":
        set_requires_grad(rate_net.head, True)

        layers = list(encoder.layers)
        k = min(LAST_ENCODER_K, len(layers))
        for li in range(len(layers) - k, len(layers)):
            set_requires_grad(layers[li], True)

        opt = torch.optim.AdamW(
            [
                {"params": rate_net.head.parameters(), "lr": LR_HEAD},
                {"params": [p for li in range(len(layers)-k, len(layers)) for p in layers[li].parameters()], "lr": LR_ENCODER},
            ],
            weight_decay=WEIGHT_DECAY
        )
        return model, opt, None, None, epochs
    if method == "methodC_last_encoderK_only":
        layers = list(encoder.layers)
        k = min(LAST_ENCODER_K, len(layers))
        for li in range(len(layers) - k, len(layers)):
            set_requires_grad(layers[li], True)

        # head 保持冻结（默认已经 freeze）
        opt = torch.optim.AdamW(
            [{"params": [p for li in range(len(layers)-k, len(layers)) for p in layers[li].parameters()],
            "lr": LR_ENCODER}],
            weight_decay=WEIGHT_DECAY
        )
        return model, opt, None, None, epochs

    if method == "methodD_add_3layer_mlp_calibrator":
        d_model = rate_net.head[0].in_features

        class HeadWithCalib(nn.Module):
            def __init__(self, orig_head: nn.Module, d_model_: int):
                super().__init__()
                self.orig = orig_head
                self.calib = ResidualHeadCalibrator(d_model=d_model_, hidden=512, dropout=0.0)
                set_requires_grad(self.orig, False)

            def forward(self, time_tok: torch.Tensor) -> torch.Tensor:
                raw = self.orig(time_tok)
                return raw + self.calib(time_tok)

        rate_net.head = HeadWithCalib(rate_net.head, d_model).to(DEVICE)
        set_requires_grad(rate_net.head.calib, True)

        opt = torch.optim.AdamW(rate_net.head.calib.parameters(), lr=5e-4, weight_decay=WEIGHT_DECAY)
        return model, opt, None, None, epochs

    if method == "methodE_static_embedder_plus_head":
        set_requires_grad(rate_net.static_embedder, True)
        set_requires_grad(rate_net.head, True)
        opt = torch.optim.AdamW(
            [
                {"params": rate_net.static_embedder.parameters(), "lr": 2e-4},
                {"params": rate_net.head.parameters(), "lr": LR_HEAD},
            ],
            weight_decay=WEIGHT_DECAY
        )
        return model, opt, None, None, epochs

    if method == "methodF_feature_affine_input_calibration":
        feature_affine = FeatureAffine(NUM_STATIC_FEATURES).to(DEVICE)
        opt = torch.optim.AdamW(feature_affine.parameters(), lr=LR_AFFINE, weight_decay=WEIGHT_DECAY)
        return model, opt, feature_affine, None, epochs

    if method == "methodG_adapter":
        set_requires_grad(rate_net.head, True)
        d_model = rate_net.head[0].in_features

        adapter_hooks = EncoderAdapterHooks(rate_net.encoder, last_k=LAST_ENCODER_K, d_model=d_model, bottleneck=64)

        # ✅关键：把 adapters 挂到 model 上，并搬到 DEVICE（避免出现两份 adapter 参数不一致）
        model._adapter_modules = adapter_hooks.adapters.to(DEVICE)
        adapter_hooks.adapters = model._adapter_modules

        opt = torch.optim.AdamW(
            [
                {"params": rate_net.head.parameters(), "lr": LR_HEAD},
                {"params": adapter_hooks.parameters(), "lr": LR_ADAPTER},
            ],
            weight_decay=WEIGHT_DECAY
        )
        return model, opt, None, adapter_hooks, epochs

    raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 11) 主流程：对每个工况（行段）分别跑
# =============================================================================
def run_one_condition(
    condition: str,
    row_start: int,
    row_end_inclusive: int,
    base_model: MonotoneIntegralPredictor,
    torch_scaler: TorchScaler,
    cond_dir: str,
) -> pd.DataFrame:
    os.makedirs(cond_dir, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"[工况] {condition} | rows {row_start}-{row_end_inclusive}")
    print("#" * 80)

    long_df = load_excel_slice_to_long(EXCEL_FILE_PATH, SHEET_NAME, row_start, row_end_inclusive)
    print(f"[{condition}] 长表样本数（非空 y）：{len(long_df)}")
    print(f"[{condition}] 唯一原始行（组）数量：{long_df['_orig_row_index'].nunique()}")

    train_df, val_df, test_df, split_groups = split_by_group(long_df, SPLIT_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    split_groups.to_csv(os.path.join(cond_dir, "split_groups.csv"), index=False)

    print(f"[{condition}] Train 样本：{len(train_df)} | groups：{train_df['_orig_row_index'].nunique()}")
    print(f"[{condition}] Val   样本：{len(val_df)} | groups：{val_df['_orig_row_index'].nunique()}")
    print(f"[{condition}] Test  样本：{len(test_df)} | groups：{test_df['_orig_row_index'].nunique()}")

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f"[{condition}] Train 或 Test 为空。请调整行范围或切分比例。")

    train_loader = make_loader(LongDataset(train_df), BATCH_SIZE, shuffle=True)
    test_loader = make_loader(LongDataset(test_df), BATCH_SIZE, shuffle=False)

    methods = [
    "method0_eval_only",
    "methodA_head_only",
    "methodB_head_plus_last_encoderK",
    "methodC_last_encoderK_only",
    "methodD_add_3layer_mlp_calibrator",
    "methodE_static_embedder_plus_head",
    "methodF_feature_affine_input_calibration",
    "methodG_adapter",
]

    summary_rows = []

    for method in methods:
        print("\n" + "=" * 80)
        print(f"[{condition}] [运行] {method}")

        method_dir = os.path.join(cond_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        model, opt, feature_affine, adapter_hooks, epochs = configure_method(base_model, method)

        if method == "method0_eval_only":
            y_true, y_pred, x_static_raw, t_min = predict_batches(model, torch_scaler, test_loader, feature_affine=None)
            eval_xlsx = os.path.join(method_dir, "evaluation_details.xlsx")
            metrics = evaluate_metrics(
                y_true=y_true,
                y_pred=y_pred,
                inputs_static=x_static_raw,
                inputs_time=t_min,
                feature_names=STATIC_FEATURE_COLS,
                threshold=EVAL_THRESHOLD,
                save_path=eval_xlsx,
                make_plots=False,
            )
            save_obj = {"model_state_dict": model.state_dict(), "config": {"method": method, "train_time_sec": 0.0, "max_epochs": 0}}
            torch.save(save_obj, os.path.join(method_dir, "model_state.pt"))
            with open(os.path.join(method_dir, "metrics.txt"), "w", encoding="utf-8") as f:
                f.write(json.dumps({**metrics, "train_time_sec": 0.0}, indent=2, ensure_ascii=False))

            summary_rows.append({
                "condition": condition,
                "method": method,
                "train_time_sec": 0.0,
                "test_count": int(metrics["count"]),
                "test_mse": float(metrics["mse"]),
                "test_mae": float(metrics["mae"]),
                "test_rmse": float(metrics["rmse"]),
                "test_r2": float(metrics["r2"]),
                "test_max_abs_error": float(metrics["max_abs_error"]),
                "test_mape_percent": float(metrics["mape_percent"]),
                "test_wmape_percent": float(metrics["wmape_percent"]),
                "test_nrmse": float(metrics["nrmse"]),
                "test_pm_mean": float(metrics["pm_mean"]),
                "test_pm_std": float(metrics["pm_std"]),
                "test_rmspe": float(metrics["rmspe"]),
                "trainable_params": int(count_trainable_params(model)),
                "epochs": 0,
            })
            continue

        if opt is None:
            raise RuntimeError(f"[{condition}] 训练方法的 optimizer 为 None：{method}")

        row = train_one_method(
            method_name=method,
            model=model,
            torch_scaler=torch_scaler,
            train_loader=train_loader,
            test_loader=test_loader,
            out_dir=method_dir,
            max_epochs=epochs,
            optimizer=opt,
            feature_affine=feature_affine,
            adapter_hooks=adapter_hooks,
        )
        row["condition"] = condition
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cond_dir, "summary_metrics.csv"), index=False)
    summary_df.to_excel(os.path.join(cond_dir, "summary_metrics.xlsx"), index=False)

    print("\n" + "=" * 80)
    print(f"[{condition}] 完成。汇总如下：")
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    set_seed(SPLIT_SEED)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = RUN_NAME or f"run_{ts}_seed{SPLIT_SEED}_train{int(TRAIN_RATIO*100)}_test{int(TEST_RATIO*100)}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg = {
        "EXCEL_FILE_PATH": EXCEL_FILE_PATH,
        "SHEET_NAME": SHEET_NAME,
        "CONDITION_RANGES": CONDITION_RANGES,
        "SPLIT_SEED": SPLIT_SEED,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "BEST_MODEL_PATH": BEST_MODEL_PATH,
        "SCALER_PATH": SCALER_PATH,
        "DEVICE": str(DEVICE),
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_EPOCHS": MAX_EPOCHS,
        "LAST_ENCODER_K": LAST_ENCODER_K,
        "EVAL_THRESHOLD": EVAL_THRESHOLD,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("[加载] scaler + 预训练模型")
    sklearn_scaler = load_scaler(SCALER_PATH)
    torch_scaler = build_torch_scaler(sklearn_scaler)
    if torch_scaler.kind != "standard":
        raise RuntimeError("你的 scaler 不是 StandardScaler-like。请适配 build_torch_scaler()。")

    base_model = load_model(BEST_MODEL_PATH).to(DEVICE)
    print(f"设备：{DEVICE}")
    print(f"Base 模型已加载。可训练参数量（base）：{count_trainable_params(base_model)}")

    all_summaries = []
    for cond, (rs, re_) in CONDITION_RANGES.items():
        cond_dir = os.path.join(run_dir, cond)
        df_sum = run_one_condition(
            condition=cond,
            row_start=rs,
            row_end_inclusive=re_,
            base_model=base_model,
            torch_scaler=torch_scaler,
            cond_dir=cond_dir,
        )
        all_summaries.append(df_sum)

    if all_summaries:
        grand = pd.concat(all_summaries, axis=0, ignore_index=True)
        grand.to_csv(os.path.join(run_dir, "summary_metrics_ALL_conditions.csv"), index=False)
        grand.to_excel(os.path.join(run_dir, "summary_metrics_ALL_conditions.xlsx"), index=False)

    print("\n" + "=" * 80)
    print("全部工况完成。")
    print(f"所有输出保存在：{os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
