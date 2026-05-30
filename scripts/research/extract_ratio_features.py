"""Extract features for Outcome Ratio Head training.

From the FROZEN base policy pi0, at every viewer DISCARD decision, record:
  * hidden    : base last-layer hidden at the predicting position (pos-1)  [d]
  * base_logp : log pi0 over the canonical discard action set (37 tiles x
                {tedashi,tsumogiri} = 74)                                  [74]
  * aplus     : canonical index of the actually-chosen discard
  * bucket    : the round's viewer score-delta bucket (0..4) [proximal y]
  * rank      : the viewer's terminal placement (1..seat_count) [distal y]
  * game      : game index (for train/test split by game)

These let us train r_y(h,a) = log pi_y/pi0 with pi0 frozen, concentrating all
learning on the residual (state-cancelled outcome signal).
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from gpt2.round_outcome import viewer_round_deltas, ROUND_OC_TOKENS
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_(.+)$")  # group2 = {tile}_{marker}
FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")
BUCKET_INDEX = {t: i for i, t in enumerate(ROUND_OC_TOKENS)}


def canonical_discards(tok):
    """Map discard token id -> canonical action index (tile_marker, seat-agnostic)."""
    suffixes = set()
    for t in tok.get_vocab():
        m = DISCARD_RE.match(t)
        if m:
            suffixes.add(m.group(2))
    canon = {suf: i for i, suf in enumerate(sorted(suffixes))}  # 74 classes
    # per-seat: token_id -> canon index ; and canon index -> per-seat token id
    id_to_canon = {}
    seat_canon_ids = {}
    for t, tid in tok.get_vocab().items():
        m = DISCARD_RE.match(t)
        if m:
            s = int(m.group(1)); ci = canon[m.group(2)]
            id_to_canon[tid] = ci
            seat_canon_ids.setdefault(s, {})[ci] = tid
    n = len(canon)
    seat_index = {}  # seat -> LongTensor[n] of token ids (canon order)
    for s, d in seat_canon_ids.items():
        seat_index[s] = [d.get(ci, -1) for ci in range(n)]
    return canon, id_to_canon, seat_index, n


def discards_in(toks):
    out = []
    seat = None
    for i, t in enumerate(toks):
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        elif seat is not None and DISCARD_RE.match(t) and int(t.split("_")[1]) == seat:
            out.append((i, seat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shards", nargs="+", default=["2024/data-00000-of-00016.arrow",
                                                    "2024/data-00001-of-00016.arrow"])
    ap.add_argument("--num-games", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--out", default="outputs/research/ratio_features.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32,
                                                output_hidden_states=True).to(device).eval()
    canon, id_to_canon, seat_index, n_act = canonical_discards(tok)
    seat_idx_t = {s: torch.tensor(v, device=device) for s, v in seat_index.items()}

    H, LP, AP, BK, RK, GM = [], [], [], [], [], []
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
                places = {int(m.group(1)): int(m.group(2)) for m in (FINAL_RANK_RE.match(t) for t in toks) if m}
                if viewer not in places:
                    continue
                vrank = places[viewer]
                rinfo = viewer_round_deltas(toks, viewer_seat=viewer, seat_count=sc)
                # map token index -> round bucket
                pos_bucket = {}
                for d in rinfo:
                    bi = BUCKET_INDEX[d["bucket"]]
                    for j in range(d["start"], d["end"]):
                        pos_bucket[j] = bi
                dec = discards_in(toks)
                if not dec:
                    continue
                out = base(torch.tensor([ids], device=device))
                hs = out.hidden_states[-1][0]      # [seq, d]
                logits = out.logits[0]             # [seq, vocab]
                for pos, seat in dec:
                    if seat not in seat_idx_t:
                        continue
                    canon_ids = seat_idx_t[seat]            # [n_act] token ids (-1 if missing)
                    valid = canon_ids >= 0
                    lp_full = torch.log_softmax(logits[pos - 1].float(), -1)
                    lp = torch.full((n_act,), -1e9, device=device)
                    lp[valid] = lp_full[canon_ids[valid]]
                    lp = torch.log_softmax(lp, -1)          # renormalise over the discard set
                    aid = ids[pos]
                    if aid not in id_to_canon:
                        continue
                    H.append(hs[pos - 1].float().cpu().numpy().astype(np.float16))
                    LP.append(lp.cpu().numpy().astype(np.float16))
                    AP.append(id_to_canon[aid])
                    BK.append(pos_bucket.get(pos, BUCKET_INDEX["round_oc_zero"]))
                    RK.append(vrank)
                    GM.append(games)
                games += 1
                if games >= args.num_games:
                    break
            if games >= args.num_games:
                break

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        hidden=np.asarray(H, dtype=np.float16),
        base_logp=np.asarray(LP, dtype=np.float16),
        aplus=np.asarray(AP, dtype=np.int64),
        bucket=np.asarray(BK, dtype=np.int64),
        rank=np.asarray(RK, dtype=np.int64),
        game=np.asarray(GM, dtype=np.int64),
        n_act=n_act,
    )
    print(f"saved {len(AP)} decisions from {games} games to {args.out}; n_act={n_act}, d={len(H[0])}")


if __name__ == "__main__":
    main()

