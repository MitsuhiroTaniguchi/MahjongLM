"""Controllable Consequence Conditioning (CCC) — auto-discover a proxy that is
BOTH high action-influence (like deal-in) AND aligned with the true return
(unlike deal-in), without hand-crafting "deal-in/safety/shanten".

Construction (Russo Prop 6.1: proxy improvement = influence x alignment):
  z_t      = Enc(s_t)                          (base hidden at the decision)
  z_{t+1}  = Enc(s_{t+1})                       (base hidden at the NEXT decision)
  m(s,a)   = E[z_{t+1} | s,a]                   (action-conditioned transition model)
  mz(s)    = E[z_{t+1} | s] = E_{a'~pi0} m(s,a')(state-only next-state predictor)
  g(s,a)   = m(s,a) - mz(s)                     (CONTROLLABLE consequence; luck removed)
  c_u(s,a) = u . g(s,a)                         (proxy along direction u)
  Y, V(s)  = round return and its state value;  Yt~ = Y - V(s)
  u        = argmax  influence(c_u) * alignment(c_u, Y~)   -> here u = ridge( Y~ ~ g(s,a+) )

Reuses critic_features.npz (z=hidden, a+, base_logp, dealin, bucket=Y, game);
z_{t+1} is reconstructed from consecutive same-game decisions (no LM re-run).
Reports held-out alignment Corr(c_u,Y~) and influence Var_a(c_u), vs the deal-in
proxy (high influence, limited alignment). Sign of u (step 7) = sign that makes
c_u predict higher return.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUCKET_VALUE = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])


def ridge_fit(X, y, lam=10.0):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=X.device)
    return torch.linalg.solve(A, X.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/critic_features.npz")
    ap.add_argument("--epochs-m", type=int, default=15)
    ap.add_argument("--support-thresh", type=float, default=0.02)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.features)
    Z = torch.tensor(z["hidden"].astype(np.float32)).to(device)
    A = torch.tensor(z["aplus"]).to(device)
    BL = torch.tensor(z["base_logp"].astype(np.float32)).to(device)
    DI = torch.tensor(z["dealin"].astype(np.float32)).to(device)
    Y = BUCKET_VALUE.to(device)[torch.tensor(z["bucket"]).to(device)]
    game = z["game"]
    n_act = BL.shape[1]; d = Z.shape[1]
    # standardize z
    mu, sd = Z.mean(0), Z.std(0) + 1e-6; Zs = (Z - mu) / sd

    # z_next from consecutive same-game decisions
    g_arr = game
    znext = torch.zeros_like(Zs); valid = np.zeros(len(g_arr), dtype=bool)
    valid[:-1] = (g_arr[1:] == g_arr[:-1])
    znext[:-1] = Zs[1:]
    valid_t = torch.tensor(valid).to(device)

    games = np.unique(g_arr); rng = np.random.default_rng(0); rng.shuffle(games)
    teset = set(games[:len(games) // 4].tolist())
    te = torch.tensor(np.array([x in teset for x in g_arr]) & valid).to(device)
    tr = torch.tensor(np.array([x not in teset for x in g_arr]) & valid).to(device)
    tri = torch.nonzero(tr).squeeze(-1); tei = torch.nonzero(te).squeeze(-1)
    print(f"N={len(A)} valid_pairs={int(valid.sum())} train={len(tri)} test={len(tei)} n_act={n_act}")

    # V(s): ridge Z->Y ; residual return
    Z1 = torch.cat([Zs, torch.ones(len(Zs), 1, device=device)], 1)
    wV = ridge_fit(Z1[tri], Y[tri].unsqueeze(1))
    V = (Z1 @ wV).squeeze(1)
    Yt = Y - V                                   # return residual

    # mz(s): state-only next-state predictor (= E_{a'~pi0} m)
    Wmz = ridge_fit(Z1[tri], znext[tri])         # [d+1, d]
    mz_all = Z1 @ Wmz                            # [N, d]

    # m(s,a): action-conditioned transition (small MLP with action embedding)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_act, 32)
            self.net = nn.Sequential(nn.Linear(d + 32, 256), nn.GELU(), nn.Linear(256, d))
        def forward(self, zz, aa):
            return self.net(torch.cat([zz, self.emb(aa)], -1))
    m = M().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs_m):
        perm = tri[torch.randperm(len(tri))]
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]; opt.zero_grad()
            F.mse_loss(m(Zs[b], A[b]), znext[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        # controllable consequence at the taken action: g(s,a+) = m(s,a+) - mz(s)
        g_aplus = torch.empty_like(Zs)
        for i in range(0, len(Zs), 16384):
            g_aplus[i:i + 16384] = m(Zs[i:i + 16384], A[i:i + 16384]) - mz_all[i:i + 16384]
        # u: ridge  Y~  ~  g(s,a+)
        u = ridge_fit(g_aplus[tri], Yt[tri].unsqueeze(1)).squeeze(1)   # [d]
        # orient so c_u predicts higher return (step 7): ridge already signs it
        c_aplus = g_aplus @ u
        # ALIGNMENT (held-out): corr(c_u(a+), Y~)
        def corr(x, y):
            x = x - x.mean(); y = y - y.mean()
            return float((x @ y) / (x.norm() * y.norm() + 1e-9))
        align_ccc = corr(c_aplus[tei], Yt[tei])
        align_dealin = corr(DI[tei], Yt[tei])             # deal-in alignment (expect negative)
        # also: does the action add over state for predicting Y~ THROUGH g? (the point)
        # baseline: raw action-advantage alignment is ~0 (established); we show c_u>0.

        # INFLUENCE (held-out sample): Var_{a~pi0}(c_u(s,a)) averaged over states
        smp = tei[torch.randperm(len(tei))[:3000]]
        infl_ccc = []; infl_di_proxy = []
        pi0_all = torch.softmax(BL, -1)
        for j in smp.tolist():
            zz = Zs[j:j + 1]
            sup = (pi0_all[j] >= args.support_thresh)
            cand = torch.nonzero(sup).squeeze(-1)
            if len(cand) < 2:
                continue
            gz = m(zz.expand(len(cand), -1), cand) - mz_all[j:j + 1]     # [k, d]
            cu = gz @ u                                                  # [k]
            p = torch.softmax(BL[j][cand], -1)
            mean = (p * cu).sum()
            infl_ccc.append(float((p * (cu - mean) ** 2).sum()))
        influence = float(np.mean(infl_ccc)) if infl_ccc else float("nan")

    print(json.dumps({
        "alignment_corr_with_return_residual": {
            "CCC_proxy_c_u": round(align_ccc, 4),
            "dealin_proxy": round(align_dealin, 4),
            "note": "CCC auto-discovered; deal-in is one (influential, oppositely-signed) component.",
        },
        "CCC_action_influence_Var_a(c_u)_mean": round(influence, 5),
        "verdict": "|align_CCC| > |align_dealin| AND influence>0 => CCC found a proxy that is both "
                   "action-influential (like deal-in) and BETTER aligned with the return than deal-in "
                   "alone — a return-aligned controllable-consequence proxy, no hand-crafting.",
    }, indent=2))


if __name__ == "__main__":
    main()

