"""Deal-in (放銃) predictability: omniscient vs imperfect — the forward-direction
test of whether the discard's outcome influence is visible given hidden info.

For each player discard we label whether it was immediately RON'd (deal-in), and
probe how well the FROZEN base model's hidden representation at the discard
predicts deal-in, under two views:
  * omniscient : h includes ALL hands + wall  -> opponents' waits are visible
  * imperfect  : h is the player's own view   -> waits are hidden

If AUC(omniscient) >> AUC(imperfect), the discard's deal-in outcome is
controllable/knowable only WITH hidden info. That gap is the value of hidden
information for the discard decision = the headroom an omniscient->imperfect
value/deal-in teacher could (partially) transfer beyond pure imitation, and
explains why outcome-conditioning on imperfect data cannot move the policy.
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


def dealin_discards(toks, viewer_only):
    """Return [(pos, seat, dealin)] for discards.

    viewer_only=True: only the non-hidden (viewer) seat's discards (imperfect).
    viewer_only=False: all discards (omniscient).
    A discard deals in if a take_react_{w}_ron (w != discarder) appears in the
    reaction window before the next draw/discard/round_start.
    """
    out = []
    seat = None
    for i, t in enumerate(toks):
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        m = DISCARD_RE.match(t)
        if m:
            s = int(m.group(1))
            if viewer_only and s != seat:
                continue
            dealin = False
            for j in range(i + 1, min(i + 40, len(toks))):
                tj = toks[j]
                if tj.startswith("draw_") or DISCARD_RE.match(tj) or tj == "round_start":
                    break
                rm = RON_RE.match(tj)
                if rm and int(rm.group(1)) != s:
                    dealin = True
                    break
            out.append((i, s, dealin))
    return out


def auc(scores, labels):
    labels = np.asarray(labels); scores = np.asarray(scores)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def collect(base, tok, ds, view_is_omni, device, num_games, max_len, want_games=None):
    H, Y, G = [], [], []
    seen = 0
    target_type = "omniscient" if view_is_omni else "imperfect"
    with torch.no_grad():
        for row in ds:
            if row["view_type"] != target_type or row["length"] > max_len:
                continue
            gid = row["game_id"]
            if want_games is not None and gid not in want_games:
                continue
            ids = [int(t) for t in row["input_ids"]]
            toks = tok.convert_ids_to_tokens(ids)
            dec = dealin_discards(toks, viewer_only=not view_is_omni)
            if not dec:
                continue
            hs = base(torch.tensor([ids], device=device)).hidden_states[-1][0]
            for pos, s, di in dec:
                H.append(hs[pos].float().cpu().numpy().astype(np.float16))
                Y.append(1 if di else 0)
                G.append(hash(gid) % (10 ** 8))
            seen += 1
            if seen >= num_games:
                break
    return np.asarray(H, dtype=np.float16), np.asarray(Y), np.asarray(G)


def train_probe(H, Y, G, device, epochs=60):
    X = torch.tensor(H.astype(np.float32)); y = torch.tensor(Y.astype(np.float32))
    g = G
    games = np.unique(g); rng = np.random.default_rng(0); rng.shuffle(games)
    te_games = set(games[:len(games) // 4].tolist())
    te = np.array([x in te_games for x in g]); tr = ~te
    Xtr, ytr = X[tr].to(device), y[tr].to(device)
    Xte, yte = X[te].to(device), y[te].to(device)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).to(device)
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        out = net(Xtr).squeeze(-1)
        lossf(out, ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        s_te = torch.sigmoid(net(Xte).squeeze(-1)).cpu().numpy()
    return auc(s_te, yte.cpu().numpy()), float(yte.mean().item()), int(te.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shard", default="2024/data-00000-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=1200)
    ap.add_argument("--max-len", type=int, default=3600)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32,
                                                output_hidden_states=True).to(device).eval()
    ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset"))

    # restrict to a common set of games (those with an omniscient view)
    omni_games = set()
    for row in ds:
        if row["view_type"] == "omniscient":
            omni_games.add(row["game_id"])
        if len(omni_games) >= args.num_games:
            break

    Ho, Yo, Go = collect(base, tok, ds, True, device, args.num_games, args.max_len, want_games=omni_games)
    Hi, Yi, Gi = collect(base, tok, ds, False, device, args.num_games, args.max_len, want_games=omni_games)
    auc_o, rate_o, n_o = train_probe(Ho, Yo, Go, device)
    auc_i, rate_i, n_i = train_probe(Hi, Yi, Gi, device)
    print(json.dumps({
        "omniscient": {"dealin_AUC": round(auc_o, 4), "dealin_rate": round(rate_o, 4),
                       "n_test": n_o, "n_total": int(len(Yo))},
        "imperfect": {"dealin_AUC": round(auc_i, 4), "dealin_rate": round(rate_i, 4),
                      "n_test": n_i, "n_total": int(len(Yi))},
        "AUC_gap_omni_minus_imp": round(auc_o - auc_i, 4),
        "interpretation": "gap>0 => deal-in (the discard's catastrophic outcome) is far more "
                          "predictable WITH hidden info; that is the headroom an omniscient->imperfect "
                          "value teacher can transfer, and why imperfect outcome-conditioning is flat.",
    }, indent=2))


if __name__ == "__main__":
    main()

