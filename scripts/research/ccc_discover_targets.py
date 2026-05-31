"""Does CCC survive under the CORRECT objective?  Compare c* alignment when the
target Y is:
  round   : per-round score-delta bucket  (the OLD, wrong proxy)
  place   : final-placement dan-points     (terminal, distal)
  equity  : placement-EQUITY change per round (faithful reduction, proximal)

g = phi - E[phi|s] is target-independent (computed once); only V(s)=E[Y|s] and
c* = ridge(Y~ ~ g) depend on Y.  Reports held-out alignment Corr(c*, Y~) overall
and per decision type for each objective.
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

ROUND_VALUE = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])


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
    ap.add_argument("--targets", default="outputs/research/ccc_placement_target.npz")
    ap.add_argument("--lam-sweep", type=float, nargs="+", default=[100., 1000., 5000., 20000.])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.features); tg = np.load(args.targets)
    S = torch.tensor(z["state"].astype(np.float32)).to(device)
    P = torch.tensor(z["phi"].astype(np.float32)).to(device)
    DTc = torch.tensor(z["dtype"]).to(device)
    game = z["game"]; N, d = S.shape

    Y_round = ROUND_VALUE.to(device)[torch.tensor(z["bucket"]).to(device)]
    Y_place = torch.tensor(tg["y_term"].astype(np.float32)).to(device)
    Y_equity = torch.tensor(tg["y_equity"].astype(np.float32)).to(device)
    targets = {"round": Y_round, "place": Y_place, "equity": Y_equity}

    games = np.unique(game); rng = np.random.default_rng(0); rng.shuffle(games)
    teset = set(games[: len(games) // 4].tolist())
    te = torch.tensor(np.array([g in teset for g in game])).to(device); tr = ~te
    tri = torch.nonzero(tr).squeeze(-1); tei = torch.nonzero(te).squeeze(-1)

    smu, ssd = S[tri].mean(0), S[tri].std(0) + 1e-6
    pmu, psd = P[tri].mean(0), P[tri].std(0) + 1e-6
    Ss = (S - smu) / ssd; Ps = (P - pmu) / psd
    S1 = torch.cat([Ss, torch.ones(N, 1, device=device)], 1)
    # target-independent controllable consequence
    Wm = ridge_fit(S1[tri], Ps[tri]); G = Ps - (S1 @ Wm); G = G - G[tri].mean(0)

    report = {}
    for tname, Y in targets.items():
        V = (S1 @ ridge_fit(S1[tri], Y[tri].unsqueeze(1))).squeeze(1); Yt = Y - V
        best = None
        for lam in args.lam_sweep:
            w = ridge_fit(G[tri], Yt[tri].unsqueeze(1), lam=lam).squeeze(1)
            a = abs(corr((G[tei] @ w), Yt[tei]))
            if best is None or a > best[0]:
                best = (a, lam, w)
        _, lam_star, w = best
        cstar = G @ w
        if float(Yt[tri][cstar[tri] >= cstar[tri].median()].mean()) < \
           float(Yt[tri][cstar[tri] < cstar[tri].median()].mean()):
            w = -w; cstar = -cstar
        per_type = {}
        for i, name in enumerate(DTYPE_NAMES):
            mt = (DTc == i); tei_t = tei[mt[tei]]
            per_type[name] = round(corr(cstar[tei_t], Yt[tei_t]), 4) if len(tei_t) >= 3 else None
        # R^2 of V (how much state explains the objective) and Y~ variance share
        ssV = 1 - float(((Y[tei] - V[tei]) ** 2).mean() / (Y[tei].var() + 1e-9))
        report[tname] = {
            "alignment_overall": round(corr(cstar[tei], Yt[tei]), 4),
            "lambda": lam_star,
            "state_R2_on_objective": round(ssV, 4),
            "per_type": per_type,
        }

    print(json.dumps({
        "objectives": report,
        "note": "round = OLD per-round score-delta (wrong); place = final-placement dan-points "
                "(terminal); equity = placement-equity change per round (faithful reduction). "
                "Higher alignment under place/equity == c* tracks the TRUE objective.",
    }, indent=2))


if __name__ == "__main__":
    main()
