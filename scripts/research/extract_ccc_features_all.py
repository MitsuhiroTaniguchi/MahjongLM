"""CCC features over ALL viewer decision points (discard / self / react /
chi_pos / red / kan_tile), not just discards.

Per decision, from the FROZEN imperfect base model (one forward per game):
  state : hs[pos-1]   -- pre-action state s
  phi   : hs[pos]     -- ACTION-PROCESSED hidden of the taken action a+
  dtype : decision-type code (gpt2.viewer_decisions.DTYPE)
  bucket: round score-delta bucket 0..4 (the TRUE-return target)
  game  : game index (train/test split)

The controllable consequence g = phi - E[phi|s] and the proxy c* are
decision-type-agnostic; CCC auto-discovers the immediate consequence, so no
per-type hand-crafted reward (deal-in was discard-only) is needed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import re
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from gpt2.round_outcome import viewer_round_deltas, ROUND_OC_TOKENS
from gpt2.viewer_decisions import iter_viewer_decisions
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")
BUCKET_INDEX = {t: i for i, t in enumerate(ROUND_OC_TOKENS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shards", nargs="+", default=["2024/data-00000-of-00016.arrow",
                                                    "2024/data-00001-of-00016.arrow"])
    ap.add_argument("--num-games", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--out", default="outputs/research/ccc_features_all.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32,
                                                output_hidden_states=True).to(device).eval()

    ST, PH, DT, BK, GM = [], [], [], [], []
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
                dec = iter_viewer_decisions(toks, viewer)
                if not dec:
                    continue
                rinfo = viewer_round_deltas(toks, viewer_seat=viewer, seat_count=sc)
                pos_bucket = {}
                for d in rinfo:
                    for j in range(d["start"], d["end"]):
                        pos_bucket[j] = BUCKET_INDEX[d["bucket"]]
                hs = base(torch.tensor([ids], device=device)).hidden_states[-1][0]
                for (pos, dt, seat) in dec:
                    if pos == 0:
                        continue
                    ST.append(hs[pos - 1].float().cpu().numpy().astype(np.float16))
                    PH.append(hs[pos].float().cpu().numpy().astype(np.float16))
                    DT.append(dt)
                    BK.append(pos_bucket.get(pos, BUCKET_INDEX["round_oc_zero"]))
                    GM.append(games)
                games += 1
                if games >= args.num_games:
                    break
            if games >= args.num_games:
                break
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
                        state=np.asarray(ST, dtype=np.float16), phi=np.asarray(PH, dtype=np.float16),
                        dtype=np.asarray(DT, dtype=np.int64), bucket=np.asarray(BK, dtype=np.int64),
                        game=np.asarray(GM, dtype=np.int64))
    DT = np.asarray(DT)
    from gpt2.viewer_decisions import DTYPE_NAMES
    counts = {DTYPE_NAMES[i]: int((DT == i).sum()) for i in range(len(DTYPE_NAMES))}
    print(f"saved {len(DT)} decisions / {games} games; by_type={counts}")


if __name__ == "__main__":
    main()
