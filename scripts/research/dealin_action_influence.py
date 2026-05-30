"""Does the DISCARD CHOICE carry outcome information beyond the state?

Critique accepted: round-rank/delta is a distal, luck-dominated outcome where a
single discard's effect drowns in variance (R^2_state ~0.07) -> a bad alignment
target. The discard's DIRECT, attributable, low-noise outcome is DEAL-IN (放銃).

We measure I(A; deal-in | S) on the imperfect view by comparing deal-in
predictability from:
  * STATE only   : hidden at pos-1 (before the discard is chosen)  -> P(deal-in | s)
  * STATE+ACTION : hidden at pos   (after emitting the discard)     -> P(deal-in | s, a)

If AUC(state+action) >> AUC(state), the discard choice strongly changes the
(observable, immediate) outcome -> outcome influence is real and large; the
earlier ~0 alignment was a measurement artifact of using a distal outcome.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_")
RON_RE = re.compile(r"^take_react_(\d+)_ron$")


def viewer_dealin_discards(toks):
    out = []
    seat = None
    for i, t in enumerate(toks):
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        m = DISCARD_RE.match(t)
        if m and seat is not None and int(m.group(1)) == seat:
            dealin = False
            for j in range(i + 1, min(i + 40, len(toks))):
                tj = toks[j]
                if tj.startswith("draw_") or DISCARD_RE.match(tj) or tj == "round_start":
                    break
                rm = RON_RE.match(tj)
                if rm and int(rm.group(1)) != seat:
                    dealin = True; break
            out.append((i, dealin))
    return out


def auc(scores, labels):
    labels = np.asarray(labels); scores = np.asarray(scores)
    npos = int((labels == 1).sum()); nneg = int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def probe(X, Y, G, device, epochs=80):
    X = torch.tensor(X.astype(np.float32)); y = torch.tensor(Y.astype(np.float32))
    games = np.unique(G); rng = np.random.default_rng(0); rng.shuffle(games)
    te_games = set(games[:len(games) // 4].tolist())
    te = np.array([g in te_games for g in G]); tr = ~te
    Xtr, ytr, Xte, yte = X[tr].to(device), y[tr].to(device), X[te].to(device), y[te].to(device)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).to(device)
    pw = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        net.train(); opt.zero_grad(); lf(net(Xtr).squeeze(-1), ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        s = torch.sigmoid(net(Xte).squeeze(-1)).cpu().numpy()
    return auc(s, yte.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shard", default="2024/data-00000-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=2500)
    ap.add_argument("--max-len", type=int, default=2600)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32,
                                                output_hidden_states=True).to(device).eval()
    ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset"))
    Hs, Hsa, Y, G = [], [], [], []
    seen = 0
    with torch.no_grad():
        for row in ds:
            if row["view_type"] != "imperfect" or row["length"] > args.max_len:
                continue
            ids = [int(t) for t in row["input_ids"]]
            toks = tok.convert_ids_to_tokens(ids)
            dec = viewer_dealin_discards(toks)
            if not dec:
                continue
            hs = base(torch.tensor([ids], device=device)).hidden_states[-1][0]
            g = hash(row["game_id"]) % (10 ** 8)
            for pos, di in dec:
                Hs.append(hs[pos - 1].float().cpu().numpy().astype(np.float16))   # state (pre-discard)
                Hsa.append(hs[pos].float().cpu().numpy().astype(np.float16))      # state+action (post-discard)
                Y.append(1 if di else 0); G.append(g)
            seen += 1
            if seen >= args.num_games:
                break
    Hs = np.asarray(Hs); Hsa = np.asarray(Hsa); Y = np.asarray(Y); G = np.asarray(G)
    auc_s = probe(Hs, Y, G, device)
    auc_sa = probe(Hsa, Y, G, device)
    print(json.dumps({
        "n_discards": int(len(Y)), "dealin_rate": round(float(Y.mean()), 4), "n_games": seen,
        "AUC_state_only_P(dealin|s)": round(auc_s, 4),
        "AUC_state+action_P(dealin|s,a)": round(auc_sa, 4),
        "action_influence_AUC_gain": round(auc_sa - auc_s, 4),
        "interpretation": "gain>>0 => the discard CHOICE strongly changes the immediate outcome "
                          "(deal-in); outcome influence is real & large. The earlier ~0 alignment used "
                          "a distal (rank/round-delta) outcome and was the wrong measurement.",
    }, indent=2))


if __name__ == "__main__":
    main()

