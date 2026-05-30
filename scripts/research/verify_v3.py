"""Verify v3 (proximal per-round outcome) conditioning.

Two decisive measurements on a v3 checkpoint (held-out 2021):

1. INFORMATION: mean action-token NLL under TRUE per-round buckets vs
   ROUND-SHUFFLED buckets (rotate by 1). If true << shuffled, the per-round
   outcome carries real information about the viewer's actions. Compare the
   gap to v2 terminal-rank's ~0.0034 nats.

2. SENSITIVITY: discard-policy movement when every round is forced to
   round_oc_bigwin vs round_oc_bigloss (chi2/TV). Compare to v2's ~0.0076.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from gpt2.outcome_conditioning import build_outcome_conditioned_v3, extend_tokenizer_for_v3
from gpt2.round_outcome import viewer_round_deltas
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_")


def discard_sets(tok):
    by = defaultdict(list)
    for t, i in tok.get_vocab().items():
        m = DISCARD_RE.match(t)
        if m:
            by[int(m.group(1))].append(i)
    return {s: sorted(v) for s, v in by.items()}


def discards_in(cond_toks):
    out = []
    seat = None
    for i, t in enumerate(cond_toks):
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        elif seat is not None and DISCARD_RE.match(t) and int(t.split("_")[1]) == seat:
            out.append((i, seat))
    return out


@torch.no_grad()
def forward_logp(model, ids, device):
    x = torch.tensor([ids], device=device)
    return torch.log_softmax(model(x).logits[0].float(), -1)


def action_nll(logp, ids, mask):
    tot = 0.0; cnt = 0
    for i in range(1, len(ids)):
        if mask[i]:
            tot += -float(logp[i - 1, ids[i]]); cnt += 1
    return tot, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reinject", default="turn")
    ap.add_argument("--shard", default="2021/data-00000-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--max-decisions", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    extend_tokenizer_for_v3(tok)
    dsets = discard_sets(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset"))

    true_tot = true_cnt = shuf_tot = shuf_cnt = 0.0
    per_game = []
    chi2s, tvs = [], []
    games = 0
    for row in ds:
        if row["view_type"] != "imperfect" or row["length"] > args.max_len:
            continue
        sc = int(row["seat_count"]); viewer = int(row["viewer_seat"])
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        rinfo = viewer_round_deltas(toks, viewer_seat=viewer, seat_count=sc)
        if len(rinfo) < 2:
            continue
        true_b = [d["bucket"] for d in rinfo]
        shuf_b = true_b[1:] + true_b[:1]  # rotate by 1 (deterministic derangement)

        t_ids, t_mask = build_outcome_conditioned_v3(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                     reinject=args.reinject, loss_scope="actions", round_buckets=true_b)
        s_ids, s_mask = build_outcome_conditioned_v3(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                     reinject=args.reinject, loss_scope="actions", round_buckets=shuf_b)
        lt = forward_logp(model, t_ids, device)
        ls = forward_logp(model, s_ids, device)
        tt, tc = action_nll(lt, t_ids, t_mask)
        st, scnt = action_nll(ls, s_ids, s_mask)
        true_tot += tt; true_cnt += tc; shuf_tot += st; shuf_cnt += scnt
        per_game.append((tt / max(1, tc), st / max(1, scnt)))

        # sensitivity: bigwin vs bigloss (all rounds)
        w_ids, _ = build_outcome_conditioned_v3(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                reinject=args.reinject, loss_scope="actions", override_bucket="round_oc_bigwin")
        l_ids, _ = build_outcome_conditioned_v3(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                reinject=args.reinject, loss_scope="actions", override_bucket="round_oc_bigloss")
        wtoks = tok.convert_ids_to_tokens(w_ids)
        dec = discards_in(wtoks)
        if dec:
            if len(dec) > args.max_decisions:
                step = len(dec) / args.max_decisions
                dec = [dec[int(k * step)] for k in range(args.max_decisions)]
            lw = forward_logp(model, w_ids, device)
            ll = forward_logp(model, l_ids, device)
            for pos, seat in dec:
                aset = dsets[seat]
                pw = torch.softmax(lw[pos - 1][aset], -1)
                pl = torch.softmax(ll[pos - 1][aset], -1)
                chi2s.append(float(((pw - pl) ** 2 / pl.clamp_min(1e-12)).sum()))
                tvs.append(float(0.5 * (pw - pl).abs().sum()))
        games += 1
        if games >= args.num_games:
            break

    import statistics as st
    diffs = [s - t for t, s in per_game]
    print(json.dumps({
        "model": args.model, "n_games": games,
        "INFORMATION": {
            "true_bucket_action_nll": round(true_tot / max(1, true_cnt), 5),
            "shuffled_bucket_action_nll": round(shuf_tot / max(1, shuf_cnt), 5),
            "nll_reduction_from_true": round(shuf_tot / max(1, shuf_cnt) - true_tot / max(1, true_cnt), 5),
            "per_game_frac_true_better": round(sum(d > 0 for d in diffs) / len(diffs), 3) if diffs else None,
            "v2_terminal_rank_reference_nll_reduction": 0.00344,
        },
        "SENSITIVITY_bigwin_vs_bigloss": {
            "n_decisions": len(tvs),
            "tv_mean": round(st.mean(tvs), 5) if tvs else None,
            "chi2_mean": round(st.mean(chi2s), 5) if chi2s else None,
            "tv_max": round(max(tvs), 5) if tvs else None,
            "v2_top_vs_bot_reference_tv": 0.0076,
        },
    }, indent=2))


if __name__ == "__main__":
    main()

