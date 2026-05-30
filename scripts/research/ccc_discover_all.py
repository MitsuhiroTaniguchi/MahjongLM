"""CCC discovery over ALL viewer decision points (discard/self/react/chi_pos/
red/kan_tile).  Same math as ccc_discover_v2 (c* = max-aligned controllable
consequence = ridge(Y~ ~ g), g = phi - E[phi|s]) but fit over every decision
type at once, with a per-type alignment breakdown.

Reports held-out alignment Corr(c*, Y~):
  * overall (one universal proxy across all decision points)
  * per decision type
and the binarize/sign (good side) per type.
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

from gpt2.viewer_decisions import DTYPE_NAMES

BUCKET_VALUE = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])


def ridge_fit(X, Y, lam=10.0):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def corr(x, y):
    if len(x) < 3:
        return float("nan")
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/ccc_features_all.npz")
    ap.add_argument("--lam-sweep", type=float, nargs="+", default=[10., 100., 1000., 5000.])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.features)
    S = torch.tensor(z["state"].astype(np.float32)).to(device)
    P = torch.tensor(z["phi"].astype(np.float32)).to(device)
    DTc = torch.tensor(z["dtype"]).to(device)
    Y = BUCKET_VALUE.to(device)[torch.tensor(z["bucket"]).to(device)]
    game = z["game"]; N, d = S.shape

    games = np.unique(game); rng = np.random.default_rng(0); rng.shuffle(games)
    teset = set(games[: len(games) // 4].tolist())
    te = torch.tensor(np.array([g in teset for g in game])).to(device)
    tr = ~te
    tri = torch.nonzero(tr).squeeze(-1); tei = torch.nonzero(te).squeeze(-1)

    smu, ssd = S[tri].mean(0), S[tri].std(0) + 1e-6
    pmu, psd = P[tri].mean(0), P[tri].std(0) + 1e-6
    Ss = (S - smu) / ssd; Ps = (P - pmu) / psd
    S1 = torch.cat([Ss, torch.ones(N, 1, device=device)], 1)
    V = (S1 @ ridge_fit(S1[tri], Y[tri].unsqueeze(1))).squeeze(1); Yt = Y - V
    Wm = ridge_fit(S1[tri], Ps[tri]); G = Ps - (S1 @ Wm); G = G - G[tri].mean(0)

    # ONE universal c* over all decision points (lambda chosen on held-out overall)
    best = None
    for lam in args.lam_sweep:
        w = ridge_fit(G[tri], Yt[tri].unsqueeze(1), lam=lam).squeeze(1)
        a = abs(corr((G[tei] @ w), Yt[tei]))
        if best is None or a > best[0]:
            best = (a, lam, w)
    align_overall, lam_star, w = best
    cstar = G @ w
    # GOOD sign
    if float(Yt[tri][cstar[tri] >= cstar[tri].median()].mean()) < \
       float(Yt[tri][cstar[tri] < cstar[tri].median()].mean()):
        w = -w; cstar = -cstar
    align_overall = corr(cstar[tei], Yt[tei])

    # per-type breakdown (shared universal c*), plus per-type SELF-FIT c* for ceiling
    per_type = []
    for i, name in enumerate(DTYPE_NAMES):
        mt = (DTc == i)
        tei_t = tei[mt[tei]]; tri_t = tri[mt[tri]]
        n_te = int(len(tei_t))
        align_shared = corr(cstar[tei_t], Yt[tei_t]) if n_te >= 3 else float("nan")
        # self-fit ceiling (fit a c* using only this type's train rows)
        align_selffit = float("nan")
        if int(len(tri_t)) > 50 and n_te >= 3:
            wt = ridge_fit(G[tri_t], Yt[tri_t].unsqueeze(1), lam=lam_star).squeeze(1)
            align_selffit = abs(corr((G[tei_t] @ wt), Yt[tei_t]))
        # binarize good-side return gap (shared c*)
        gap = float("nan")
        if n_te >= 10:
            thr = float(cstar[tri_t].median()) if len(tri_t) else float(cstar[tei_t].median())
            hi = cstar[tei_t] >= thr
            if hi.any() and (~hi).any():
                gap = abs(float(Yt[tei_t][hi].mean()) - float(Yt[tei_t][~hi].mean()))
        per_type.append({
            "type": name, "n_train": int(len(tri_t)), "n_test": n_te,
            "align_shared_cstar": round(align_shared, 4) if align_shared == align_shared else None,
            "align_selffit_cstar": round(align_selffit, 4) if align_selffit == align_selffit else None,
            "binarized_return_gap": round(gap, 4) if gap == gap else None,
        })

    print(json.dumps({
        "N": N, "train": int(len(tri)), "test": int(len(tei)),
        "universal_cstar": {"alignment_overall": round(align_overall, 4), "lambda": lam_star},
        "per_decision_type": per_type,
        "note": "align_shared_cstar = ONE universal proxy applied to each type; "
                "align_selffit_cstar = proxy fit on that type alone (per-type ceiling). "
                "All decision points now covered, not just discard.",
    }, indent=2))


if __name__ == "__main__":
    main()
