"""CCC fix #3 (lightweight): GENUINE conditioning via a continuous multi-axis
c*-target tilt head on a FROZEN base LM, under the CORRECT objective (placement
dan-points).  Replaces the forced argmax reranker with a calibrated, steerable
conditional policy

    pi(a | s, t) = softmax( log pi0(a|s) + tilt(state, t) ),   t in R^K

where t is the DESIRED consequence vector along K auto-discovered axes
(oriented so higher = higher placement return).  Trained by conditioning on the
realised consequence of the taken action  t = c_axes(s, a+)  (decision-transformer
style; K is small so t is a coarse semantic target, not an action id).

Held-out tests of GENUINE conditioning (no forced argmax):
  - dNLL(true - shuffled target): conditioning on a+'s true consequence beats a
    shuffled target => the policy actually uses the target (and it generalises).
  - swing TV(pi(.|good) || pi(.|bad)) + in-support mass: calibrated steerable shift
    (contrast: argmax rerank TV 0.61 is one-hot, not calibrated).
  - return steering: corr( logpi(a+|good) - logpi(a+|bad), Y~(a+) ) > 0 means
    conditioning 'good' upweights actually-higher-placement-return actions.
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
import torch.nn as nn
import torch.nn.functional as F


def ridge_fit(X, Y, lam=10.0):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/ccc_features.npz")
    ap.add_argument("--features-all", default="outputs/research/ccc_features_all.npz")
    ap.add_argument("--targets", default="outputs/research/ccc_placement_target.npz")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--target-mag", type=float, default=1.5)   # +/- sigma for good/bad
    ap.add_argument("--support-thresh", type=float, default=0.02)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    f = np.load(args.features); fa = np.load(args.features_all); tg = np.load(args.targets)
    mask = fa["dtype"] == 0
    assert np.array_equal(f["game"], fa["game"][mask]), "discard alignment broken"
    S = torch.tensor(f["state"].astype(np.float32)).to(device)
    P = torch.tensor(f["phi"].astype(np.float32)).to(device)
    LP = torch.tensor(f["base_logp"].astype(np.float32)).to(device)
    AP = torch.tensor(f["aplus"]).to(device).long()
    Y = torch.tensor(tg["y_term"][mask].astype(np.float32)).to(device)   # placement dan-points
    game = f["game"]; N, d = S.shape; n_act = LP.shape[1]

    games = np.unique(game); rng = np.random.default_rng(0); rng.shuffle(games)
    teset = set(games[: len(games) // 4].tolist())
    te = torch.tensor(np.array([g in teset for g in game])).to(device); tr = ~te
    tri = torch.nonzero(tr).squeeze(-1); tei = torch.nonzero(te).squeeze(-1)

    smu, ssd = S[tri].mean(0), S[tri].std(0) + 1e-6
    pmu, psd = P[tri].mean(0), P[tri].std(0) + 1e-6
    Ss = (S - smu) / ssd; Ps = (P - pmu) / psd
    S1 = torch.cat([Ss, torch.ones(N, 1, device=device)], 1)
    Wm = ridge_fit(S1[tri], Ps[tri]); G = Ps - (S1 @ Wm); G = G - G[tri].mean(0)
    V = (S1 @ ridge_fit(S1[tri], Y[tri].unsqueeze(1))).squeeze(1); Yt = Y - V

    # K-axis consequence (PLS deflation vs placement residual), oriented higher=better return
    Yres = Yt.clone(); Gw = G.clone(); axes = []
    for k in range(args.K):
        b = (Gw[tri].T @ Yres[tri]) / len(tri); u = b / (b.norm() + 1e-12)
        c = Gw @ u
        if corr(c[tri], Yt[tri]) < 0:
            u = -u; c = -c
        axes.append(u)
        cc = c - c[tri].mean(); den = float(cc[tri] @ cc[tri]) + 1e-9
        Yres = Yres - cc * float((cc[tri] @ Yres[tri]) / den)
        Gw = Gw - torch.outer(cc, (Gw[tri].T @ cc[tri]) / den)
    W = torch.stack(axes, 1)                      # [d, K]
    Caxes = G @ W                                 # [N, K] consequence of taken action
    cmu, csd = Caxes[tri].mean(0), Caxes[tri].std(0) + 1e-6
    T = (Caxes - cmu) / csd                       # standardized target of the realised action

    class Tilt(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d + args.K, 256), nn.GELU(), nn.Linear(256, n_act))
        def forward(self, s, t):
            return self.net(torch.cat([s, t], -1))
    tilt = Tilt().to(device)
    opt = torch.optim.AdamW(tilt.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        tilt.train(); perm = tri[torch.randperm(len(tri))]
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]; opt.zero_grad()
            logits = LP[b] + tilt(Ss[b], T[b])
            F.cross_entropy(logits, AP[b]).backward(); opt.step()

    tilt.eval()
    with torch.no_grad():
        # 1) true vs shuffled target NLL (held-out)
        lt = LP[tei] + tilt(Ss[tei], T[tei])
        nll_true = F.cross_entropy(lt, AP[tei]).item()
        perm = tei[torch.randperm(len(tei))]
        ls = LP[tei] + tilt(Ss[tei], T[perm])
        nll_shuf = F.cross_entropy(ls, AP[tei]).item()
        nll_pi0 = F.cross_entropy(LP[tei], AP[tei]).item()
        # 2) good vs bad steering (t = +/- mag on all axes)
        good = torch.full((len(tei), args.K), args.target_mag, device=device)
        bad = torch.full((len(tei), args.K), -args.target_mag, device=device)
        pg = torch.softmax(LP[tei] + tilt(Ss[tei], good), -1)
        pb = torch.softmax(LP[tei] + tilt(Ss[tei], bad), -1)
        p0 = torch.softmax(LP[tei], -1)
        tv_gb = (0.5 * (pg - pb).abs().sum(-1)).mean().item()
        tv_g0 = (0.5 * (pg - p0).abs().sum(-1)).mean().item()
        sup = (p0 >= args.support_thresh).float()
        ins_good = (pg * sup).sum(-1).mean().item()
        # 3) return steering: does conditioning good upweight higher-Y~ taken actions?
        lpg_a = torch.log(pg.gather(1, AP[tei].unsqueeze(1)).squeeze(1) + 1e-12)
        lpb_a = torch.log(pb.gather(1, AP[tei].unsqueeze(1)).squeeze(1) + 1e-12)
        steer = corr(lpg_a - lpb_a, Yt[tei])
        # leakage diagnostic: does good-vs-bad upweight a+ by its OWN c* goodness?
        steer_cstar = corr(lpg_a - lpb_a, Caxes[tei].sum(-1))

    print(json.dumps({
        "objective": "placement dan-points", "K_axes": args.K,
        "genuine_conditioning_uses_target": {
            "NLL_pi0": round(nll_pi0, 5), "NLL_true_target": round(nll_true, 5),
            "NLL_shuffled_target": round(nll_shuf, 5),
            "dNLL_true_minus_shuffled": round(nll_true - nll_shuf, 5),
            "note": "true < shuffled (and < pi0) => the policy genuinely conditions on the target.",
        },
        "calibrated_steering": {
            "swing_TV_good_vs_bad": round(tv_gb, 4),
            "TV_good_vs_pi0": round(tv_g0, 4),
            "good_in_support_mass": round(ins_good, 4),
            "vs_forced_argmax_rerank_TV": 0.61,
        },
        "return_steering_corr": round(steer, 4),
        "cstar_steering_corr": round(steer_cstar, 4),
        "verdict": "if dNLL very negative but return_steering_corr~0 => LEAKAGE (target encodes the "
                   "action with few candidates), NOT genuine return-improving conditioning. "
                   "cstar_steering_corr shows whether good-vs-bad even upweights high-c* actions.",
    }, indent=2))


if __name__ == "__main__":
    main()
