import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
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
IDX_Q = FEATURES.index("Spray Flow Rate (L/min)")
IDX_CM = FEATURES.index("Mist mean concentration (P/cm^3)")
IDX_DM = FEATURES.index("Mist mean size (μm)")
TIMES = ["0min", "1min", "3min", "5min", "10min", "15min", "20min", "30min", "40min", "50min", "60min"]
TIME_MIN = [0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]
TIME_SCALE = 60.0
N_ZRO2_ROWS = 1365
TEST_RUNS = [4, 5, 15, 16, 29, 30]


def load_data(path):
    df = pd.read_excel(path, sheet_name="Sheet1", nrows=N_ZRO2_ROWS)
    run = df["Case"].astype(str).str.extract(r"^(\d+)")[0].astype(float).ffill()
    run.loc[410:446] = 10.0
    raw = df[FEATURES].values.astype(np.float32)
    y = df[TIMES].apply(lambda c: c.astype(float)).values
    y[:, 0] = 1.0
    return raw, y.astype(np.float32), run.values.astype(int)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, t):
        pe = torch.zeros(t.size(0), self.embed_dim, device=t.device, dtype=t.dtype)
        pe[:, 0::2] = torch.sin(t * self.div_term)
        pe[:, 1::2] = torch.cos(t * self.div_term)
        return pe


class StaticFeatureEmbedder(nn.Module):
    def __init__(self, num_features, embed_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(num_features, embed_dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, s):
        return self.fc2(self.dropout(self.act(self.fc1(s))))


class MonotoneRateNet(nn.Module):
    def __init__(self, num_features, static_embed, time_embed, d_model, n_head, n_layers, dropout, coef_floor):
        super().__init__()
        self.coef_floor = float(coef_floor)
        self.static_embedder = StaticFeatureEmbedder(num_features, static_embed, dropout)
        self.time_embedder = SinusoidalTimeEmbedding(time_embed)
        self.proj_static = nn.Linear(static_embed, d_model) if static_embed != d_model else nn.Identity()
        self.proj_time = nn.Linear(time_embed, d_model) if time_embed != d_model else nn.Identity()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True,
                                           dim_feedforward=d_model * 4)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 4))
        self.softplus = nn.Softplus()

    def forward(self, t, s):
        s_coef = s.clone()
        s_coef[:, [IDX_Q, IDX_CM, IDX_DM]] = 0.0
        h_s = self.proj_static(self.static_embedder(s_coef))
        h_t = self.proj_time(self.time_embedder(t.unsqueeze(-1).to(s.dtype)))
        raw = self.head(self.encoder(torch.stack([h_s, h_t], dim=1))[:, 1, :])
        k0 = self.softplus(raw[:, 0:1])
        kq = self.softplus(raw[:, 1:2])
        kc = self.softplus(raw[:, 2:3]) + self.coef_floor
        kd = self.softplus(raw[:, 3:4]) + self.coef_floor
        sp = self.softplus
        return (k0 + kq * sp(s[:, IDX_Q:IDX_Q + 1]) + kc * sp(s[:, IDX_CM:IDX_CM + 1])
                + kd * sp(s[:, IDX_DM:IDX_DM + 1]))


class PCNODE(nn.Module):
    def __init__(self, static_embed, time_embed, d_model, n_head, n_layers, dropout, coef_floor):
        super().__init__()
        self.rate_net = MonotoneRateNet(len(FEATURES), static_embed, time_embed, d_model, n_head, n_layers, dropout,
                                        coef_floor)

    def forward(self, z, t):
        b, n = z.shape[0], t.numel()
        zr = z.unsqueeze(1).expand(b, n, z.shape[1]).reshape(b * n, -1)
        k = self.rate_net(t.repeat(b), zr).view(b, n)
        inc = 0.5 * (k[..., 1:] + k[..., :-1]) * (t[1:] - t[:-1])
        return torch.exp(-torch.cat([torch.zeros_like(k[..., :1]), torch.cumsum(inc, dim=-1)], dim=-1))


def load_model(model_dir, device):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    scaler = json.loads((model_dir / "scaler.json").read_text(encoding="utf-8"))
    model = PCNODE(**cfg["model"]).to(device)
    model.load_state_dict(torch.load(model_dir / "pcnode.pt", map_location=device, weights_only=True))
    model.eval()
    return model, np.array(scaler["mean"]), np.array(scaler["scale"])


def standardize(raw, mean, scale):
    z = raw.astype(np.float32)
    z -= mean
    z /= scale
    return z


@torch.no_grad()
def predict(model, z, device, batch=256):
    t = torch.tensor(np.array(TIME_MIN) / TIME_SCALE, dtype=torch.float32, device=device)
    z = torch.as_tensor(z, dtype=torch.float32, device=device)
    return torch.cat([model(z[s:s + batch], t) for s in range(0, len(z), batch)]).cpu().numpy()


def metrics(y, p, t):
    y, p, t = np.asarray(y, float), np.asarray(p, float), np.asarray(t, float)
    pos = t > 0
    log_err = np.log10(np.maximum(y[pos], 1e-3)) - np.log10(np.maximum(p[pos], 1e-3))
    return dict(N=len(y), R2=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
                RMSE=np.sqrt(np.mean((y - p) ** 2)), logRMSE=np.sqrt(np.mean(log_err ** 2)),
                WMAPE=np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "aerosol_data.xlsx"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    model, mean, scale = load_model(args.models, device)

    if args.input:
        df = pd.read_csv(args.input)
        pred = predict(model, standardize(df[FEATURES].values, mean, scale), device)
        out = pd.concat([df[FEATURES], pd.DataFrame(pred, columns=TIMES)], axis=1)
        out.to_csv(args.output, index=False)
        print(out.round(4).to_string(index=False))
        print(f"saved to {args.output}")
        return

    raw, y, run = load_data(args.data)
    pred = predict(model, standardize(raw, mean, scale), device)
    valid = np.isfinite(y)
    t = np.array(TIME_MIN)
    rows = []
    for name, runs in [(f"run {r}", [r]) for r in TEST_RUNS] + [("all test runs", TEST_RUNS)]:
        i, j = np.nonzero(valid & np.isin(run, runs)[:, None])
        rows.append(dict(set=name, **metrics(y[i, j], pred[i, j], t[j])))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
