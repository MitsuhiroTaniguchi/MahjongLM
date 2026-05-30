"""Entropy-Gated / Entropy-Splitting Outcome Conditioning (ESOC) prototype.

Decompose the frozen base discard policy pi0(a|s) into K latent strategy modes:

    pi_z(a|s) = softmax(log pi0(a|s) + u_z(s, .))     (tilt on the frozen base)
    pi_mix(a|s) = sum_z q(z|s) pi_z(a|s)

Loss (per the proposed design):
    L =  -log pi_mix(a+|s)                       # marginal BC -> pulls pi_mix -> pi0
       +  lam_kl * KL(pi_mix || pi0)             # explicit marginal preservation
       -  lam_mi * [H(pi_mix) - sum_z q_z H(pi_z)]   # = -lam_mi * I(Z;A|S)  (force conditioning to matter)
       +  lam_out * sum_z r_z (Y - V_z(s))^2     # outcome alignment (r_z = responsibilities)

The three things that must hold SIMULTANEOUSLY on a held-out (by-game) split:
    (1) I(Z;A|S) ↑          (conditioning changes the action)
    (2) KL(pi_mix||pi0) ≈ 0 (the averaged policy stays at base -> chi^2-safe)
    (3) outcome alignment GENERALISES: knowing the mode (via the action) predicts
        the held-out outcome better than the state-only value.

(1)+(2) are optimisable by construction; (3) is the binding test = whether pi0's
entropy hides outcome-relevant strategic modes (behaviour heterogeneity) or just
noise. (3) is the make-or-break for beating the action-influence ceiling.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUCKET_VALUE = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])  # bigwin..bigloss -> utility


class ESOC(nn.Module):
    def __init__(self, d, n_act, K):
        super().__init__()
        self.K = K
        self.gate = nn.Linear(d, K)
        self.tilt = nn.Linear(d, K * n_act)
        self.value = nn.Linear(d, K)
        self.n_act = n_act
        nn.init.zeros_(self.tilt.weight); nn.init.zeros_(self.tilt.bias)  # start at pi_z = pi0

    def forward(self, hidden, base_logp):
        B = hidden.shape[0]
        q = torch.log_softmax(self.gate(hidden), -1)          # log q(z|s)  [B,K]
        u = self.tilt(hidden).view(B, self.K, self.n_act)     # tilt        [B,K,A]
        logpi_z = torch.log_softmax(base_logp.unsqueeze(1) + u, -1)  # [B,K,A]
        V = self.value(hidden)                                # [B,K]
        return q, logpi_z, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/ratio_features.npz")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-mi", type=float, default=0.3)
    ap.add_argument("--lam-out", type=float, default=0.3)
    ap.add_argument("--lam-kl", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.features)
    H = torch.tensor(z["hidden"].astype(np.float32))
    BL = torch.tensor(z["base_logp"].astype(np.float32))
    A = torch.tensor(z["aplus"])
    Y = BUCKET_VALUE[torch.tensor(z["bucket"])]           # outcome utility
    game = z["game"]
    n_act = BL.shape[1]; d = H.shape[1]
    games = np.unique(game); rng = np.random.default_rng(0); rng.shuffle(games)
    te_games = set(games[:len(games) // 4].tolist())
    te_mask = np.array([g in te_games for g in game])
    tr = torch.tensor(np.where(~te_mask)[0]); te = torch.tensor(np.where(te_mask)[0])

    H, BL, A, Y = H.to(device), BL.to(device), A.to(device), Y.to(device)
    pi0 = torch.softmax(BL, -1)
    net = ESOC(d, n_act, args.K).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def losses(idx):
        q, logpi_z, V = net(H[idx], BL[idx])           # q:[b,K] log, logpi_z:[b,K,A], V:[b,K]
        qd = q.exp()
        # mixture over modes
        logpi_mix = torch.logsumexp(q.unsqueeze(-1) + logpi_z, dim=1)   # [b,A]
        # marginal BC
        bc = F.nll_loss(logpi_mix, A[idx])
        # KL(pi_mix || pi0)
        pim = logpi_mix.exp()
        kl_mix = (pim * (logpi_mix - torch.log_softmax(BL[idx], -1))).sum(-1).mean()
        # I(Z;A|S) = H(pi_mix) - sum_z q_z H(pi_z)
        H_mix = -(pim * logpi_mix).sum(-1)
        H_z = -(logpi_z.exp() * logpi_z).sum(-1)          # [b,K]
        mi = (H_mix - (qd * H_z).sum(-1)).mean()
        # responsibilities r(z|s,a+) ∝ q_z pi_z(a+)
        logpz_a = logpi_z.gather(-1, A[idx].view(-1, 1, 1).expand(-1, args.K, 1)).squeeze(-1)  # [b,K]
        r = torch.softmax(q + logpz_a, -1)                # [b,K]
        out = (r.detach() * (Y[idx].unsqueeze(-1) - V) ** 2).sum(-1).mean()
        return bc, kl_mix, mi, out, (q, qd, logpi_z, V, r)

    for ep in range(args.epochs):
        net.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            bc, kl_mix, mi, out, _ = losses(b)
            loss = bc + args.lam_kl * kl_mix - args.lam_mi * mi + args.lam_out * out
            loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        q, logpi_z, V = net(H[te], BL[te])
        qd = q.exp()
        logpi_mix = torch.logsumexp(q.unsqueeze(-1) + logpi_z, dim=1)
        pim = logpi_mix.exp()
        I_za = (-(pim * logpi_mix).sum(-1) - (qd * (-(logpi_z.exp() * logpi_z).sum(-1))).sum(-1)).mean().item()
        kl_mix = (pim * (logpi_mix - torch.log_softmax(BL[te], -1))).sum(-1).mean().item()
        ce_mix = F.nll_loss(logpi_mix, A[te]).item()
        ce_pi0 = F.nll_loss(torch.log_softmax(BL[te], -1), A[te]).item()
        # (3) outcome alignment generalisation
        logpz_a = logpi_z.gather(-1, A[te].view(-1, 1, 1).expand(-1, args.K, 1)).squeeze(-1)
        r = torch.softmax(q + logpz_a, -1)                # responsibilities using the action
        val_state = (qd * V).sum(-1)                      # E[Y|s]   (no action info)
        val_action = (r * V).sum(-1)                      # E[Y|s,a] (action shifts the mode)
        yte = Y[te]
        mse_state = ((yte - val_state) ** 2).mean().item()
        mse_action = ((yte - val_action) ** 2).mean().item()
        var_y = ((yte - yte.mean()) ** 2).mean().item()
        # best-mode policy movement (objective = maximise value)
        zbest = V.argmax(-1)
        logpi_best = logpi_z[torch.arange(len(te)), zbest]
        tv_best = (0.5 * (logpi_best.exp() - pi0[te]).abs().sum(-1)).mean().item()
        # per-mode mean predicted value spread (are modes outcome-differentiated?)
        vmean = V.mean(0).tolist()

    print(json.dumps({
        "K": args.K, "n_test": int(len(te)),
        "ONE_I_Z_A_given_S": round(I_za, 5),
        "TWO_KL_pimix_pi0": round(kl_mix, 6),
        "CE_pi_mix": round(ce_mix, 5), "CE_pi0": round(ce_pi0, 5),
        "THREE_outcome_alignment": {
            "var_Y": round(var_y, 4),
            "MSE_state_only_value": round(mse_state, 4),
            "MSE_action_informed_value": round(mse_action, 4),
            "R2_state": round(1 - mse_state / var_y, 4),
            "R2_action_informed": round(1 - mse_action / var_y, 4),
            "alignment_gain_R2": round((mse_state - mse_action) / var_y, 5),
        },
        "best_mode_policy_TV_vs_pi0": round(tv_best, 4),
        "per_mode_mean_value": [round(v, 3) for v in vmean],
        "verdict": "If I(Z;A|S)>0 AND KL(pimix||pi0)~0 AND alignment_gain_R2>0 (action-informed "
                   "value predicts held-out outcome better than state-only) -> ESOC extracts a real, "
                   "in-support improvement. If alignment_gain_R2~0 -> modes are not outcome-real "
                   "(the action-influence ceiling holds even with forced MI).",
    }, indent=2))


if __name__ == "__main__":
    main()

