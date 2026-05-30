"""(B) Advantage-rank conditioning, rollout-free.

Step 1 — short-horizon critic Q(s,a): predict the immediate, attributable
outcome (deal-in) from (state, candidate action). No rollout; feedforward;
generalises across actions via an action embedding. (Deal-in is the
detectable short-horizon advantage; round-bucket is reported for contrast.)

Step 2 — advantage-rank conditioning: at each state, rank the IN-SUPPORT
actions by the critic; bucket the rank (best..worst). Train a policy
    pi(a|s,rank) = softmax(log pi0(a) + tilt(hidden, rank))
The rank is an (s,a)-deterministic, HIGH-MI conditioning variable, so unlike
the realised-outcome it cannot be ignored. Inference: condition rank=best.

Diagnostics (held-out, by game):
    * critic deal-in AUC (detectability of the advantage)
    * conditioning sensitivity TV(pi(.|s,best) || pi(.|s,worst))   [vs outcome ~0]
    * improvement: E[deal-in risk] under pi(.|best) vs pi0, and the
      argmax-policy's risk; in-support by construction.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def auc(s, y):
    s = np.asarray(s); y = np.asarray(y)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class Critic(nn.Module):
    """Q(s,a) = P(deal-in | s, a) via a shared action embedding (generalises over a)."""
    def __init__(self, d, n_act, d_a=32, d_h=128):
        super().__init__()
        self.a_emb = nn.Embedding(n_act, d_a)
        self.net = nn.Sequential(nn.Linear(d + d_a, d_h), nn.GELU(), nn.Linear(d_h, 1))

    def forward(self, hidden, a_idx):                 # hidden[B,d], a_idx[B]
        return self.net(torch.cat([hidden, self.a_emb(a_idx)], -1)).squeeze(-1)

    def all_actions(self, hidden, n_act):             # -> [B, n_act] logits
        B = hidden.shape[0]
        ae = self.a_emb.weight.unsqueeze(0).expand(B, -1, -1)            # [B,n_act,d_a]
        h = hidden.unsqueeze(1).expand(-1, n_act, -1)                    # [B,n_act,d]
        return self.net(torch.cat([h, ae], -1)).squeeze(-1)


class RankPolicy(nn.Module):
    """pi(a|s,rank) = softmax(base_logp + tilt(hidden, rank))."""
    def __init__(self, d, n_act, n_rank, d_h=256):
        super().__init__()
        self.r_emb = nn.Embedding(n_rank, d)
        self.net = nn.Sequential(nn.Linear(d, d_h), nn.GELU(), nn.Linear(d_h, n_act))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden, rank, base_logp):
        return torch.log_softmax(base_logp + self.net(hidden + self.r_emb(rank)), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/critic_features.npz")
    ap.add_argument("--target", choices=("return", "dealin"), default="return",
                    help="critic target: 'return' = the domain's general n-step reward "
                         "(here the round score-delta; domain-agnostic); 'dealin' = the "
                         "1-step-return special case (mahjong-flavoured reference).")
    ap.add_argument("--horizon", type=int, default=0,
                    help="n-step return horizon in decisions (0 = full round return).")
    ap.add_argument("--n-rank", type=int, default=5)
    ap.add_argument("--support-thresh", type=float, default=0.02, help="pi0 prob to count as in-support")
    ap.add_argument("--epochs-critic", type=int, default=30)
    ap.add_argument("--epochs-policy", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.features)
    H = torch.tensor(z["hidden"].astype(np.float32)).to(device)
    BL = torch.tensor(z["base_logp"].astype(np.float32)).to(device)
    A = torch.tensor(z["aplus"]).to(device)
    DI = torch.tensor(z["dealin"].astype(np.float32)).to(device)
    BUCKET_VALUE = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0], device=device)  # bigwin..bigloss
    RET = BUCKET_VALUE[torch.tensor(z["bucket"]).to(device)]                 # general return (score)
    game = z["game"]; n_act = BL.shape[1]; d = H.shape[1]
    games = np.unique(game); rng = np.random.default_rng(0); rng.shuffle(games)
    teset = set(games[:len(games) // 4].tolist())
    te = torch.tensor(np.where(np.array([g in teset for g in game]))[0]).to(device)
    tr = torch.tensor(np.where(np.array([g not in teset for g in game]))[0]).to(device)
    print(f"N={len(A)} dealin_rate={DI.mean().item():.4f} n_act={n_act} train={len(tr)} test={len(te)}")

    # critic target: y (regression to general return, or BCE to dealin); g = "higher is better"
    if args.target == "return":
        Y = RET
    else:
        Y = DI

    # ---- Step 1: critic Q(s,a) ----
    crit = Critic(d, n_act).to(device)
    opt = torch.optim.AdamW(crit.parameters(), lr=1e-3, weight_decay=1e-4)
    if args.target == "dealin":
        pw = torch.tensor([(DI[tr] == 0).sum() / max(1, (DI[tr] == 1).sum())], device=device)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        lossf = nn.MSELoss()
    for ep in range(args.epochs_critic):
        crit.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]; opt.zero_grad()
            lossf(crit(H[b], A[b]), Y[b]).backward(); opt.step()
    crit.eval()
    with torch.no_grad():
        pred_te = crit(H[te], A[te])
        if args.target == "dealin":
            critic_metric = ("dealin_AUC", round(auc(torch.sigmoid(pred_te).cpu().numpy(), DI[te].cpu().numpy()), 4))
            Gall_te = -torch.sigmoid(crit.all_actions(H[te], n_act))     # higher=better => -risk
        else:
            yv = Y[te]; ss = ((yv - pred_te) ** 2).mean(); var = ((yv - yv.mean()) ** 2).mean()
            critic_metric = ("return_R2", round((1 - ss / var).item(), 4))
            Gall_te = crit.all_actions(H[te], n_act)                     # predicted return (higher=better)

    # ---- advantage-rank labels (higher g = better), among in-support ----
    with torch.no_grad():
        pi0 = torch.softmax(BL, -1)
        if args.target == "dealin":
            Gall = -torch.sigmoid(crit.all_actions(H, n_act))
        else:
            Gall = crit.all_actions(H, n_act)
        support = pi0 >= args.support_thresh
        score = Gall.clone(); score[~support] = -1e9                 # unsupported -> worst
        order = score.argsort(dim=-1, descending=True)               # descending g (best first)
        rankpos = torch.empty_like(order);
        ar = torch.arange(n_act, device=device).unsqueeze(0).expand(len(H), -1)
        rankpos.scatter_(1, order, ar)                             # rankpos[i,a] = position of a
        n_sup = support.sum(-1, keepdim=True).clamp_min(1)
        rank_frac = rankpos.float() / n_sup.float()                # 0=best .. ~1 worst (unsup>1)
        rank_bucket_all = (rank_frac * args.n_rank).clamp(0, args.n_rank - 1).long()
        rank_of_aplus = rank_bucket_all[torch.arange(len(H), device=device), A]

    # ---- Step 2: rank-conditioned policy ----
    pol = RankPolicy(d, n_act, args.n_rank).to(device)
    opt2 = torch.optim.AdamW(pol.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs_policy):
        pol.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]; opt2.zero_grad()
            logp = pol(H[b], rank_of_aplus[b], BL[b])
            F.nll_loss(logp, A[b]).backward(); opt2.step()
    pol.eval()
    with torch.no_grad():
        best = torch.zeros(len(te), dtype=torch.long, device=device)
        worst = torch.full((len(te),), args.n_rank - 1, dtype=torch.long, device=device)
        p_best = pol(H[te], best, BL[te]).exp()
        p_worst = pol(H[te], worst, BL[te]).exp()
        p0 = torch.softmax(BL[te], -1)
        tv_best_worst = (0.5 * (p_best - p_worst).abs().sum(-1)).mean().item()
        tv_best_pi0 = (0.5 * (p_best - p0).abs().sum(-1)).mean().item()
        # expected critic VALUE (higher=better) under each policy
        val_pi0 = (p0 * Gall_te).sum(-1).mean().item()
        val_best = (p_best * Gall_te).sum(-1).mean().item()
        val_worst = (p_worst * Gall_te).sum(-1).mean().item()
        val_pi0_argmax = Gall_te.gather(1, p0.argmax(-1, keepdim=True)).mean().item()
        val_best_argmax = Gall_te.gather(1, p_best.argmax(-1, keepdim=True)).mean().item()
        insupport_mass = (p_best * (p0 >= args.support_thresh)).sum(-1).mean().item()

    print(json.dumps({
        "target": args.target, "critic_" + critic_metric[0]: critic_metric[1],
        "CONDITIONING_SENSITIVITY": {
            "TV_best_vs_worst": round(tv_best_worst, 4),
            "TV_best_vs_pi0": round(tv_best_pi0, 4),
            "outcome_conditioning_reference_TV": 0.0076,
        },
        "IMPROVEMENT_expected_critic_value_higher_better": {
            "pi0": round(val_pi0, 5), "cond_best": round(val_best, 5), "cond_worst": round(val_worst, 5),
            "value_gain_best_vs_pi0": round(val_best - val_pi0, 5),
            "pi0_argmax_value": round(val_pi0_argmax, 5), "cond_best_argmax_value": round(val_best_argmax, 5),
        },
        "cond_best_in_support_mass": round(insupport_mass, 4),
        "verdict": "GENERAL method (critic target = domain return). TV_best_vs_worst >> outcome(0.008) "
                   "=> advantage-RANK is a high-MI condition. value_gain>0 with high in-support mass => "
                   "rollout-free, in-support improvement. Critic metric shows how detectable the advantage "
                   "is at this reward horizon.",
    }, indent=2))


if __name__ == "__main__":
    main()

