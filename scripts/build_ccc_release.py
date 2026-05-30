"""Assemble the deployable mahjonglm-10m-ccc release.

CCC is a consequence-scoring head on top of the frozen base LM.  This packages a
SELF-CONTAINED HF repo:  base LM weights + tokenizer + the fitted universal c*
head (ccc_head.npz) + the inference wrapper (ccc_policy.py) + README.

The head is fit on ALL decisions in ccc_features_all.npz (all viewer decision
points).  Held-out generalisation metrics are reported by ccc_discover_all.py /
ccc_rank_conditioning_all.py and quoted in the README.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from tenhou_tokenizer.huggingface import MahjongTokenizerFast

BUCKET_VALUE = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])


def ridge_fit(X, Y, lam=10.0):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/research/ccc_features_all.npz")
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--lam", type=float, default=5000.0)
    ap.add_argument("--out", default="outputs/ccc_release")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- fit the universal c* head on ALL decisions ----
    z = np.load(args.features)
    S = torch.tensor(z["state"].astype(np.float32)).to(device)
    P = torch.tensor(z["phi"].astype(np.float32)).to(device)
    Y = BUCKET_VALUE.to(device)[torch.tensor(z["bucket"]).to(device)]
    N, d = S.shape
    smu, ssd = S.mean(0), S.std(0) + 1e-6
    pmu, psd = P.mean(0), P.std(0) + 1e-6
    Ss = (S - smu) / ssd; Ps = (P - pmu) / psd
    S1 = torch.cat([Ss, torch.ones(N, 1, device=device)], 1)
    V = (S1 @ ridge_fit(S1, Y.unsqueeze(1))).squeeze(1); Yt = Y - V
    Wm = ridge_fit(S1, Ps); G = Ps - (S1 @ Wm); gmean = G.mean(0); G = G - gmean
    w = ridge_fit(G, Yt.unsqueeze(1), lam=args.lam).squeeze(1)
    cstr = G @ w
    sign = 1.0
    if float(Yt[cstr >= cstr.median()].mean()) < float(Yt[cstr < cstr.median()].mean()):
        w = -w; sign = -1.0

    np.savez(out / "ccc_head.npz",
             smu=smu.cpu().numpy(), ssd=ssd.cpu().numpy(),
             pmu=pmu.cpu().numpy(), psd=psd.cpu().numpy(),
             Wm=Wm.cpu().numpy(), gmean=gmean.cpu().numpy(), w=w.cpu().numpy(),
             lam=np.float32(args.lam), sign=np.float32(sign), n_fit=np.int64(N))
    print(f"fit c* head on {N} decisions; saved ccc_head.npz (d={d}, lam={args.lam}, sign={sign})")

    # ---- bundle base LM + tokenizer (self-contained) ----
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    base.save_pretrained(str(out))
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    tok.save_pretrained(str(out))
    # bundle the inference wrapper + decision helpers
    shutil.copy(SRC / "gpt2" / "ccc_policy.py", out / "ccc_policy.py")
    shutil.copy(SRC / "gpt2" / "viewer_decisions.py", out / "viewer_decisions.py")
    print(f"bundled base LM + tokenizer + ccc_policy.py + viewer_decisions.py into {out}")


if __name__ == "__main__":
    main()
