"""Features for Controllable Consequence Conditioning (CCC) discovery.

Per viewer DISCARD decision, from the FROZEN imperfect base model, in ONE forward:
  * state : hs[pos-1]  -- pre-action state s  (to estimate E_{a'~pi0}[phi | s])
  * phi   : hs[pos]    -- ACTION-PROCESSED hidden of the TAKEN action a+
                         (the model has attended to the action token; a linear
                          readout of this predicts deal-in at AUC 0.936 ->
                          the immediate consequence is linearly present here)
  * base_logp : log pi0 over the 74 canonical discards
  * aplus     : canonical index of the chosen discard
  * dealin    : 1 if this discard was immediately RON'd (for COMPARISON only)
  * bucket    : round score-delta bucket 0..4 (the TRUE-return target, diluted)
  * game      : game index (train/test split)

The controllable consequence is g(s,a+) = phi - E[phi|s]; E[phi|s] is the
pi0-conditional mean (a+ ~ pi0), estimated by ridge(state -> phi). deal-in's
centered linear readout lives inside g, so CCC's search space contains it.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from gpt2.round_outcome import viewer_round_deltas, ROUND_OC_TOKENS
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_(.+)$")
RON_RE = re.compile(r"^take_react_(\d+)_ron$")
FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")
BUCKET_INDEX = {t: i for i, t in enumerate(ROUND_OC_TOKENS)}


def canonical_discards(tok):
    suffixes = set()
    for t in tok.get_vocab():
        m = DISCARD_RE.match(t)
        if m:
            suffixes.add(m.group(2))
    canon = {suf: i for i, suf in enumerate(sorted(suffixes))}
    id_to_canon, seat_canon = {}, {}
    for t, tid in tok.get_vocab().items():
        m = DISCARD_RE.match(t)
        if m:
            s = int(m.group(1)); ci = canon[m.group(2)]
            id_to_canon[tid] = ci
            seat_canon.setdefault(s, {})[ci] = tid
    n = len(canon)
    seat_index = {s: [d.get(ci, -1) for ci in range(n)] for s, d in seat_canon.items()}
    return id_to_canon, seat_index, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shards", nargs="+", default=["2024/data-00000-of-00016.arrow",
                                                    "2024/data-00001-of-00016.arrow"])
    ap.add_argument("--num-games", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--out", default="outputs/research/ccc_features.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32,
                                                output_hidden_states=True).to(device).eval()
    id_to_canon, seat_index, n_act = canonical_discards(tok)
    seat_idx_t = {s: torch.tensor(v, device=device) for s, v in seat_index.items()}

    ST, PH, LP, AP, DI, BK, GM = [], [], [], [], [], [], []
    games = 0
    with torch.no_grad():
        for shard in args.shards:
            ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", shard, repo_type="dataset"))
            for row in ds:
                if row["view_type"] != "imperfect" or row["length"] > args.max_len:
                    continue
                sc = int(row["seat_count"]); viewer = int(row["viewer_seat"])
                ids = [int(t) for t in row["input_ids"]]
                toks = tok.convert_ids_to_tokens(ids)
                if not any(FINAL_RANK_RE.match(t) for t in toks):
                    continue
                rinfo = viewer_round_deltas(toks, viewer_seat=viewer, seat_count=sc)
                pos_bucket = {}
                for d in rinfo:
                    for j in range(d["start"], d["end"]):
                        pos_bucket[j] = BUCKET_INDEX[d["bucket"]]
                seat = None
                dec = []
                for i, t in enumerate(toks):
                    if t == "round_start":
                        seat = None
                    elif t.startswith("draw_") and not t.endswith("_hidden"):
                        seat = int(t.split("_")[1])
                    elif seat is not None and DISCARD_RE.match(t) and int(t.split("_")[1]) == seat:
                        di = False
                        for j in range(i + 1, min(i + 40, len(toks))):
                            tj = toks[j]
                            if tj.startswith("draw_") or DISCARD_RE.match(tj) or tj == "round_start":
                                break
                            rm = RON_RE.match(tj)
                            if rm and int(rm.group(1)) != seat:
                                di = True; break
                        dec.append((i, seat, di))
                if not dec:
                    continue
                out = base(torch.tensor([ids], device=device))
                hs = out.hidden_states[-1][0]; logits = out.logits[0]
                for pos, seat, di in dec:
                    if seat not in seat_idx_t:
                        continue
                    cids = seat_idx_t[seat]; valid = cids >= 0
                    lpf = torch.log_softmax(logits[pos - 1].float(), -1)
                    lp = torch.full((n_act,), -1e9, device=device)
                    lp[valid] = lpf[cids[valid]]
                    lp = torch.log_softmax(lp, -1)
                    aid = ids[pos]
                    if aid not in id_to_canon:
                        continue
                    ST.append(hs[pos - 1].float().cpu().numpy().astype(np.float16))
                    PH.append(hs[pos].float().cpu().numpy().astype(np.float16))     # action-processed
                    LP.append(lp.cpu().numpy().astype(np.float16))
                    AP.append(id_to_canon[aid]); DI.append(1 if di else 0)
                    BK.append(pos_bucket.get(pos, BUCKET_INDEX["round_oc_zero"])); GM.append(games)
                games += 1
                if games >= args.num_games:
                    break
            if games >= args.num_games:
                break
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
                        state=np.asarray(ST, dtype=np.float16), phi=np.asarray(PH, dtype=np.float16),
                        base_logp=np.asarray(LP, dtype=np.float16),
                        aplus=np.asarray(AP, dtype=np.int64), dealin=np.asarray(DI, dtype=np.int64),
                        bucket=np.asarray(BK, dtype=np.int64), game=np.asarray(GM, dtype=np.int64), n_act=n_act)
    print(f"saved {len(AP)} decisions / {games} games; dealin_rate={np.mean(DI):.4f}; n_act={n_act}")


if __name__ == "__main__":
    main()
