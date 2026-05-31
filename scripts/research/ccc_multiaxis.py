"""Multi-axis CCC (fix #1) under the CORRECT objective (placement dan-points).

A single scalar c* collapses defense/offense/tempo.  Here we extract K orthogonal
consequence axes via PLS deflation of g against the placement-return residual,
and INTERPRET each axis by its correlation with:
  - Y_place~ : placement dan-point residual (the true objective)  -> what we align to
  - Y_round~ : round score-delta residual (chips/tempo)           -> "win points" axis
Axes that load on Y_round but not extra Y_place = pure tempo; axes loading on
Y_place beyond Y_round = rank-management (e.g. fold to protect placement).

Reports per-axis alignment + the combined K-axis alignment (regression), showing
the multi-axis representation needed for genuine multi-target conditioning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

ROUND_VALUE = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])


def ridge_fit(X, Y, lam=10.0):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/ccc_features_all.npz")
    ap.add_argument("--targets", default="outputs/research/ccc_placement_target.npz")
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.features); tg = np.load(args.targets)
    S = torch.tensor(z["state"].astype(np.float32)).to(device)
    P = torch.tensor(z["phi"].astype(np.float32)).to(device)
    game = z["game"]; N, d = S.shape
    Yp = torch.tensor(tg["y_term"].astype(np.float32)).to(device)
    Yr = ROUND_VALUE.to(device)[torch.tensor(z["bucket"]).to(device)]

    games = np.unique(game); rng = np.random.default_rng(0); rng.shuffle(games)
    teset = set(games[: len(games) // 4].tolist())
    te = torch.tensor(np.array([g in teset for g in game])).to(device); tr = ~te
    tri = torch.nonzero(tr).squeeze(-1); tei = torch.nonzero(te).squeeze(-1)

    smu, ssd = S[tri].mean(0), S[tri].std(0) + 1e-6
    pmu, psd = P[tri].mean(0), P[tri].std(0) + 1e-6
    Ss = (S - smu) / ssd; Ps = (P - pmu) / psd
    S1 = torch.cat([Ss, torch.ones(N, 1, device=device)], 1)
    Wm = ridge_fit(S1[tri], Ps[tri]); G = Ps - (S1 @ Wm); G = G - G[tri].mean(0)
    Vp = (S1 @ ridge_fit(S1[tri], Yp[tri].unsqueeze(1))).squeeze(1); Ytp = Yp - Vp
    Vr = (S1 @ ridge_fit(S1[tri], Yr[tri].unsqueeze(1))).squeeze(1); Ytr = Yr - Vr

    # PLS deflation of g against the PLACEMENT residual
    Yres = Ytp.clone(); Gw = G.clone(); comps = []
    for k in range(args.K):
        b = (Gw[tri].T @ Yres[tri]) / len(tri)
        u = b / (b.norm() + 1e-12)
        c = Gw @ u
        comps.append({
            "axis": k + 1,
            "align_placement": round(corr(c[tei], Ytp[tei]), 4),
            "corr_round_delta": round(corr(c[tei], Ytr[tei]), 4),
            "influence_Var": round(float(c[tei].var()), 4),
        })
        cc = c - c[tri].mean(); den = float(cc[tri] @ cc[tri]) + 1e-9
        Yres = Yres - cc * float((cc[tri] @ Yres[tri]) / den)
        coef = (Gw[tri].T @ cc[tri]) / den
        Gw = Gw - torch.outer(cc, coef)

    # combined K-axis alignment with placement (held-out regression): stack component scores
    Yres = Ytp.clone(); Gw = G.clone(); scores = []
    for k in range(args.K):
        b = (Gw[tri].T @ Yres[tri]) / len(tri); u = b / (b.norm() + 1e-12)
        c = Gw @ u; scores.append(c)
        cc = c - c[tri].mean(); den = float(cc[tri] @ cc[tri]) + 1e-9
        Yres = Yres - cc * float((cc[tri] @ Yres[tri]) / den)
        Gw = Gw - torch.outer(cc, (Gw[tri].T @ cc[tri]) / den)
    Cmat = torch.stack(scores, 1)
    Cmat1 = torch.cat([Cmat, torch.ones(N, 1, device=device)], 1)
    wC = ridge_fit(Cmat1[tri], Ytp[tri].unsqueeze(1), lam=1.0)
    combined = corr((Cmat1 @ wC).squeeze(1)[tei], Ytp[tei])
    single = corr((G[tei] @ ridge_fit(G[tri], Ytp[tri].unsqueeze(1), lam=100.).squeeze(1)), Ytp[tei])

    print(json.dumps({
        "objective": "placement dan-points (neural equity baseline V(s))",
        "single_axis_cstar_alignment": round(single, 4),
        "multi_axis": comps,
        "combined_Kaxis_alignment": round(combined, 4),
        "interpretation": "axis with high corr_round_delta but low extra align_placement = tempo/chips; "
                          "axis with align_placement >> corr_round_delta = rank-management (placement "
                          "value beyond raw points). Multi-axis is the conditioning target for fix #3.",
    }, indent=2))


if __name__ == "__main__":
    main()
