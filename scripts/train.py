import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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
MODEL = dict(static_embed=128, time_embed=32, d_model=128, n_head=8, n_layers=3, dropout=0.02, coef_floor=1e-6)
TRAIN = dict(epochs=1760, batch_size=16, lr=7.575009105024426e-4, weight_decay=1e-5, clip=1.0, alpha=0.3, y_min=1e-5,
             r_max=10.0, eps=1e-8, lr_factor=0.5, lr_patience=10, min_lr=1e-6)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def hybrid_loss(pred, target, alpha, y_min, r_max, eps):
    mse = torch.mean((pred - target) ** 2)
    tc = torch.clamp(target, min=y_min)
    rel = torch.clamp((pred - tc) / (tc + eps), -r_max, r_max)
    rmspe = torch.sqrt(torch.mean(rel ** 2) + eps)
    scale = mse.detach() / (rmspe.detach() + eps)
    return mse + alpha * scale * rmspe


def train(model, z, y, mask, t, cfg, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=cfg["lr_patience"], factor=cfg["lr_factor"],
                                                       min_lr=cfg["min_lr"])
    rows = torch.nonzero(mask.any(1)).squeeze(1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = rows[torch.randperm(len(rows), generator=g).to(z.device)]
        total, nb = 0.0, 0
        for s in range(0, len(perm), cfg["batch_size"]):
            r = perm[s:s + cfg["batch_size"]]
            m = mask[r]
            loss = hybrid_loss(model(z[r], t)[m], y[r][m], cfg["alpha"], cfg["y_min"], cfg["r_max"], cfg["eps"])
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            total += float(loss.detach())
            nb += 1
        sched.step(total / max(nb, 1))
        print(f"epoch {epoch + 1}/{cfg['epochs']}  loss {total / max(nb, 1):.4e}  lr {opt.param_groups[0]['lr']:.2e}  "
              f"{time.time() - t0:.0f} s", flush=True)
    model.eval()
    return model


def save(out, model, scaler, n_params, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "pcnode.pt")
    (out / "scaler.json").write_text(json.dumps(dict(features=FEATURES, mean=scaler.mean_.tolist(),
                                                     scale=scaler.scale_.tolist()), indent=1, ensure_ascii=False),
                                     encoding="utf-8")
    (out / "config.json").write_text(json.dumps(dict(model=MODEL, train=dict(TRAIN, seed=seed), n_params=n_params,
                                                     time_min=TIME_MIN, time_scale=TIME_SCALE, test_runs=TEST_RUNS),
                                                indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "aerosol_data.xlsx"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)

    raw, y, run = load_data(args.data)
    dev = ~np.isin(run, TEST_RUNS)
    scaler = StandardScaler().fit(raw[dev])
    z = torch.tensor(scaler.transform(raw).astype(np.float32), device=device)
    valid = np.isfinite(y)
    y = torch.tensor(np.nan_to_num(y), device=device)
    mask = torch.tensor(valid & dev[:, None], device=device)
    t = torch.tensor(np.array(TIME_MIN) / TIME_SCALE, dtype=torch.float32, device=device)

    set_seed(args.seed)
    model = PCNODE(**MODEL).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"development runs {len(np.unique(run[dev]))}  trajectories {int(dev.sum())}  parameters {n_params:,}")
    model = train(model, z, y, mask, t, TRAIN, args.seed)
    save(args.out, model, scaler, n_params, args.seed)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
