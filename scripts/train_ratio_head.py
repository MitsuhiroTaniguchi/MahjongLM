"""Train the Outcome Ratio Head: r_y(h,a) = log pi_y(a|h) - log pi0(a|h).

pi0 (the base discard policy) is FROZEN and supplied as base_logp; the head only
learns the residual tilt, so all capacity/gradient focuses on the
state-cancelled outcome signal. Objective = CE of the actual discard under
pi_y = softmax(base_logp + r_y), which is the K->inf limit of the user's
conditional-NCE density-ratio estimator; its optimum is r_y = log(pi_y/pi0).

Decisive diagnostics on a held-out (by-game) split:
  * dCE = CE(pi0) - CE(pi_y)   > 0  => the outcome carries real conditional
    information about the discard that the plain conditional LM buried
    (the failure was OPTIMIZATION, not a true A _|_ Y | H).
  * chi2_y(h) = E_{a~pi0}[(exp r - 1)^2]  = the per-state action-influence the
    head extracted (compare to the ~0 movement of v2/v3).
  * per-k policy movement TV(pi_bigwin || pi_bigloss).
  * SHUFFLED-label control: same training with permuted outcome labels should
    give dCE ~ 0 (rules out head overfitting).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RatioHead(nn.Module):
    def __init__(self, d_model: int, n_act: int, n_cls: int, d_hidden: int = 256):
        super().__init__()
        self.k_embed = nn.Embedding(n_cls, d_model)
        self.trunk = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU())
        self.out = nn.Linear(d_hidden, n_act)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # r=0 at init -> pi_y=pi0

    def forward(self, hidden, k):
        z = self.trunk(hidden + self.k_embed(k))
        return self.out(z)  # [B, n_act] residual logits r_y(h, .)


def ce(base_logp, r, aplus):
    """CE of aplus under pi_y = softmax(base_logp + r)."""
    logits = base_logp + r
    logp = torch.log_softmax(logits, -1)
    return F.nll_loss(logp, aplus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/ratio_features.npz")
    ap.add_argument("--outcome", choices=("bucket", "rank"), default="bucket")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--d-hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.features)
    hidden = torch.tensor(z["hidden"].astype(np.float32))
    base_logp = torch.tensor(z["base_logp"].astype(np.float32))
    aplus = torch.tensor(z["aplus"])
    game = z["game"]
    y_raw = z["bucket"] if args.outcome == "bucket" else (z["rank"] - 1)
    y = torch.tensor(y_raw)
    n_cls = int(y.max().item()) + 1
    n_act = base_logp.shape[1]; d_model = hidden.shape[1]
    print(f"N={len(aplus)} d={d_model} n_act={n_act} n_cls={n_cls} outcome={args.outcome}")

    # split by game
    games = np.unique(game)
    rng = np.random.default_rng(0); rng.shuffle(games)
    n_test = int(len(games) * args.test_frac)
    test_games = set(games[:n_test].tolist())
    is_test = np.array([g in test_games for g in game])
    tr = torch.tensor(np.where(~is_test)[0]); te = torch.tensor(np.where(is_test)[0])

    def run(labels, tag):
        torch.manual_seed(0)
        head = RatioHead(d_model, n_act, n_cls, args.d_hidden).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        H = hidden.to(device); BL = base_logp.to(device); A = aplus.to(device); Y = labels.to(device)
        best = None
        for ep in range(args.epochs):
            head.train(); perm = tr[torch.randperm(len(tr))]
            for i in range(0, len(perm), args.batch):
                b = perm[i:i + args.batch]
                opt.zero_grad()
                r = head(H[b], Y[b])
                loss = ce(BL[b], r, A[b])
                loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                r_te = head(H[te], Y[te])
                ce_y = ce(BL[te], r_te, A[te]).item()
            best = ce_y if best is None else min(best, ce_y)
        with torch.no_grad():
            ce_pi0 = F.nll_loss(torch.log_softmax(BL[te], -1), A[te]).item()
            r_te = head(H[te], Y[te])
            ce_y = ce(BL[te], r_te, A[te]).item()
            # chi^2_y(h) = sum_a pi0(a) (exp r - 1)^2
            pi0 = torch.softmax(BL[te], -1)
            chi2 = (pi0 * (torch.exp(r_te) - 1) ** 2).sum(-1).mean().item()
            # per-k movement: TV(pi_kmax || pi_kmin) on test states
            kmax = torch.full_like(Y[te], n_cls - 1) * 0 + (0 if args.outcome == "bucket" else 0)  # bigwin/1st = idx 0
            kmin = torch.full_like(Y[te], n_cls - 1)  # bigloss/last = idx n_cls-1
            r_hi = head(H[te], kmax); r_lo = head(H[te], kmin)
            p_hi = torch.softmax(BL[te] + r_hi, -1); p_lo = torch.softmax(BL[te] + r_lo, -1)
            tv = (0.5 * (p_hi - p_lo).abs().sum(-1)).mean().item()
            tv_max = (0.5 * (p_hi - p_lo).abs().sum(-1)).max().item()
        return {
            "tag": tag, "CE_pi0": round(ce_pi0, 5), "CE_pi_y": round(ce_y, 5),
            "dCE_reduction": round(ce_pi0 - ce_y, 5),
            "best_test_CE": round(best, 5),
            "chi2_action_influence_mean": round(chi2, 5),
            "TV_topclass_vs_bottomclass_mean": round(tv, 5),
            "TV_max": round(tv_max, 5),
        }

    real = run(y, "real_labels")
    # shuffled-label control (permute labels across decisions)
    perm = torch.tensor(np.random.default_rng(1).permutation(len(y)))
    shuf = run(y[perm], "shuffled_labels")
    print(json.dumps({
        "outcome": args.outcome, "n_train": int(len(tr)), "n_test": int(len(te)),
        "real": real, "shuffled_control": shuf,
        "verdict": "dCE(real) >> dCE(shuffled)~0  => outcome carries real conditional "
                   "discard signal that plain conditioning buried (optimization failure). "
                   "dCE(real) ~ 0 => genuine A _|_ Y | H.",
    }, indent=2))


if __name__ == "__main__":
    main()

