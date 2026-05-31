"""Root-cause dissection of CCC's match failure.

Hypothesis: c* aligns with return BETWEEN states (state-confounded) but has ~no
WITHIN-state (across-candidate) action signal, so argmax_a c* is noise that just
deviates from the well-calibrated pi0.  Tests, from the bos-correct stored
features + the DEPLOYED head:

  1. Corr(c*, Yt)           : the headline alignment (taken actions)
  2. Corr(c*, V(s))         : state-value confound
  3. R2( c* ~ state )       : how much of c* is a pure function of STATE (not action)
                              -> if high, within a state c* barely varies => argmax noise
  4. within-vs-between var  : Var of c* explained by state vs residual
  5. decile monotonicity    : mean Yt by c* decile (does the TOP decile turn over?)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch


def ridge_fit(X, Y, lam=10.0):
    A = X.T @ X + lam * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/ccc_features_all.npz")
    ap.add_argument("--targets", default="outputs/research/ccc_placement_target.npz")
    ap.add_argument("--head", default="outputs/ccc_release/ccc_head.npz")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.features); tg = np.load(args.targets); H = np.load(args.head)
    S = torch.tensor(z["state"].astype(np.float32)).to(dev)
    P = torch.tensor(z["phi"].astype(np.float32)).to(dev)
    Y = torch.tensor(tg["y_term"].astype(np.float32)).to(dev)
    DT = torch.tensor(z["dtype"]).to(dev)
    t = lambda k: torch.tensor(H[k].astype(np.float32)).to(dev)
    SMU, SSD, PMU, PSD, WM, GMEAN, WVEC = t("smu"), t("ssd"), t("pmu"), t("psd"), t("Wm"), t("gmean"), t("w")
    N, d = S.shape

    # c*(taken) for ALL decisions, exactly as the head computes
    Ss = (S - SMU) / SSD
    S1 = torch.cat([Ss, torch.ones(N, 1, device=dev)], 1)
    Es = S1 @ WM
    G = (P - PMU) / PSD - Es - GMEAN
    cstar = G @ WVEC

    # V(s) on placement
    V = (S1 @ ridge_fit(S1, Y.unsqueeze(1))).squeeze(1)
    Yt = Y - V

    print(f"N={N} d={d}")
    print(f"[1] Corr(c*, Yt)               = {corr(cstar, Yt):+.4f}   (headline alignment)")
    print(f"[2] Corr(c*, V(s))             = {corr(cstar, V):+.4f}   (state-value confound)")
    # [3] R2 of c* explained purely by STATE (standardized state -> c*)
    wsc = ridge_fit(S1, cstar.unsqueeze(1)); cstar_hat = (S1 @ wsc).squeeze(1)
    ss_res = float(((cstar - cstar_hat) ** 2).sum()); ss_tot = float(((cstar - cstar.mean()) ** 2).sum())
    r2_state = 1 - ss_res / ss_tot
    print(f"[3] R2( c* ~ state )           = {r2_state:+.4f}   (how much of c* is a pure STATE function)")
    print(f"    => within-state c* std (residual) / total c* std = {np.sqrt(ss_res/ss_tot):.4f}")
    # how much of c*'s correlation with Yt survives removing the state-predictable part of c*
    cstar_resid = cstar - cstar_hat
    print(f"[4] Corr(state-removed c*, Yt) = {corr(cstar_resid, Yt):+.4f}   (genuine within-state action signal)")
    print(f"    Corr(state-part  c*, Yt)   = {corr(cstar_hat, Yt):+.4f}   (confounded between-state part)")
    # [5] decile monotonicity (does argmax region have the best return?)
    order = torch.argsort(cstar)
    print("[5] mean Yt by c* decile (low->high c*):")
    dec = torch.chunk(order, 10)
    print("    " + "  ".join(f"{float(Yt[idx].mean()):+.3f}" for idx in dec))
    print("    mean y_term(placement pts) by decile:")
    print("    " + "  ".join(f"{float(Y[idx].mean()):+.1f}" for idx in dec))
    # discard-only view (the dominant decision)
    md = DT == 0
    print(f"[discard-only] Corr(c*,Yt)={corr(cstar[md], Yt[md]):+.4f}  R2(c*~state)={r2_state:.3f} (shared)")


if __name__ == "__main__":
    main()
