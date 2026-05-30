"""Decisive test: does the conditioned outcome carry information about the
viewer's actions? Compares mean action-token NLL under TRUE-rank conditioning
vs WRONG-rank conditioning (viewer's place permuted), averaged over held-out
games. Uses the exact training builder + action loss mask.

If true_nll << wrong_nll -> the model uses the condition (sensitivity metric
under-reported it). If true_nll ~= wrong_nll -> the condition is genuinely
~uninformative about actions (confirms low action-influence, not a bug).
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from gpt2.outcome_conditioning import build_outcome_conditioned_v2
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")


def set_viewer_place(ids, toks, tok, viewer, vplace, sc):
    """Replace final_rank tokens so the viewer occupies vplace (others fill the
    remaining places in seat order)."""
    others = [p for p in range(1, sc + 1) if p != vplace]
    place = {viewer: vplace}
    oi = 0
    for s in range(sc):
        if s == viewer:
            continue
        place[s] = others[oi]; oi += 1
    new = list(ids)
    for j, t in enumerate(toks):
        m = FINAL_RANK_RE.match(t)
        if m:
            new[j] = tok.convert_tokens_to_ids(f"final_rank_{int(m.group(1))}_{place[int(m.group(1))]}")
    return new


@torch.no_grad()
def action_nll(model, out_ids, mask, device):
    """Mean NLL over the action (label-kept) positions of a conditioned seq."""
    x = torch.tensor([out_ids], device=device)
    logits = model(x).logits[0].float()
    logp = torch.log_softmax(logits, -1)
    tot = 0.0; cnt = 0
    for i in range(1, len(out_ids)):
        if mask[i]:  # token i is an action -> predicted by logits[i-1]
            tot += -float(logp[i - 1, out_ids[i]])
            cnt += 1
    return tot, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reinject", default="turn")
    ap.add_argument("--shard", default="2021/data-00000-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=2600)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    shard = hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset")
    ds = Dataset.from_file(shard)

    true_tot = true_cnt = 0.0
    wrong_tot = wrong_cnt = 0.0
    games = 0
    per_game = []
    for row in ds:
        if row["view_type"] != "imperfect" or row["length"] > args.max_len:
            continue
        sc = int(row["seat_count"]); viewer = int(row["viewer_seat"])
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        places = {int(m.group(1)): int(m.group(2)) for m in (FINAL_RANK_RE.match(t) for t in toks) if m}
        if viewer not in places:
            continue
        vtrue = places[viewer]

        # TRUE conditioning = ids as-is
        t_ids, t_mask = build_outcome_conditioned_v2(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                     reinject=args.reinject, loss_scope="actions")
        tt, tc = action_nll(model, t_ids, t_mask, device)

        # WRONG conditioning = average over all other viewer placements
        wrong_places = [p for p in range(1, sc + 1) if p != vtrue]
        gw_tot = gw_cnt = 0.0
        for wp in wrong_places:
            w_in = set_viewer_place(ids, toks, tok, viewer, wp, sc)
            w_ids, w_mask = build_outcome_conditioned_v2(w_in, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                         reinject=args.reinject, loss_scope="actions")
            wt, wc = action_nll(model, w_ids, w_mask, device)
            gw_tot += wt; gw_cnt += wc
        true_tot += tt; true_cnt += tc
        wrong_tot += gw_tot; wrong_cnt += gw_cnt
        per_game.append((tt / max(1, tc), gw_tot / max(1, gw_cnt)))
        games += 1
        if games >= args.num_games:
            break

    true_nll = true_tot / max(1, true_cnt)
    wrong_nll = wrong_tot / max(1, wrong_cnt)
    # paired per-game difference (wrong - true); positive => true conditioning helps
    diffs = [w - t for t, w in per_game]
    import statistics as st
    print(json.dumps({
        "model": args.model, "n_games": games,
        "true_rank_action_nll": round(true_nll, 5),
        "wrong_rank_action_nll": round(wrong_nll, 5),
        "nll_reduction_from_true": round(wrong_nll - true_nll, 5),
        "per_game_mean_diff(wrong-true)": round(st.mean(diffs), 5) if diffs else None,
        "per_game_frac_true_better": round(sum(d > 0 for d in diffs) / len(diffs), 3) if diffs else None,
        "note": "reduction>0 and frac>0.5 => model uses the condition; ~0 => uninformative",
    }, indent=2))


if __name__ == "__main__":
    main()

