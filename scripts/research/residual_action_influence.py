"""Operationalise "remove the orthogonal (state) component, keep the improvement
(action) component".

The raw round outcome O is confounded by state S (a good starting hand drives
both the action and the good outcome). We:
  1. Estimate a STATE baseline V_hat(S) by ridge-regressing the round delta on
     the model's hidden state at the round's first viewer decision.
  2. Form the residual (advantage proxy) A = delta - V_hat(S), which is
     orthogonal to the state by construction.
  3. Test whether the residual still carries ACTION-correlated signal:
     E[A | riichi] - E[A | no-riichi] (and for calls). If this gap is large,
     the action has genuine influence beyond state (the "improvement component"
     survives -> residual-bucket conditioning v3.5 is worth building). If ~0,
     the action's apparent value was all state confound (ceiling confirmed).

Also reports R^2 of V_hat (how much of delta the state explains) and compares
the action-vs-RAW-delta gap to the action-vs-RESIDUAL gap.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from gpt2.round_outcome import viewer_round_deltas
from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shard", default="2021/data-00000-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--ridge", type=float, default=10.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 output_hidden_states=True).to(device).eval()
    ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset"))

    States, deltas, riichi_f, call_f, won_f = [], [], [], [], []
    games = 0
    with torch.no_grad():
        for row in ds:
            if row["view_type"] != "imperfect" or row["length"] > args.max_len:
                continue
            sc = int(row["seat_count"]); viewer = int(row["viewer_seat"])
            ids = [int(t) for t in row["input_ids"]]
            toks = tok.convert_ids_to_tokens(ids)
            rinfo = viewer_round_deltas(toks, viewer_seat=viewer, seat_count=sc)
            if not rinfo:
                continue
            hs = model(torch.tensor([ids], device=device)).hidden_states[-1][0]  # (seq, hidden)
            for d in rinfo:
                lo, hi, seat = d["start"], d["end"], d["seat"]
                if seat is None:
                    continue
                # state = hidden at the round's first viewer draw (encodes starting hand+board)
                draw_pos = None
                for i in range(lo, hi):
                    t = toks[i]
                    if t.startswith("draw_") and not t.endswith("_hidden") and int(t.split("_")[1]) == seat:
                        draw_pos = i; break
                if draw_pos is None:
                    continue
                seg = toks[lo:hi]
                States.append(hs[draw_pos].float().cpu().numpy())
                deltas.append(float(d["delta"]))
                riichi_f.append(1.0 if f"take_self_{seat}_riichi" in seg else 0.0)
                call_f.append(1.0 if any(f"take_react_{seat}_{c}" in seg for c in ("chi", "pon", "minkan")) else 0.0)
                won_f.append(1.0 if (f"take_self_{seat}_tsumo" in seg or f"take_react_{seat}_ron" in seg) else 0.0)
            games += 1
            if games >= args.num_games:
                break

    X = np.asarray(States); y = np.asarray(deltas)
    riichi = np.asarray(riichi_f); call = np.asarray(call_f); won = np.asarray(won_f)
    n = len(y)
    # Ridge with intercept (split train/test for honest R^2 / residual)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    Xc = X - X[tr].mean(0); yc = y - y[tr].mean()
    d = Xc.shape[1]
    W = np.linalg.solve(Xc[tr].T @ Xc[tr] + args.ridge * np.eye(d), Xc[tr].T @ yc[tr])
    yhat = Xc @ W + y[tr].mean()
    resid = y - yhat
    ss_res = float(((y[te] - yhat[te]) ** 2).sum()); ss_tot = float(((y[te] - y[tr].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    def gap(flag, vals):
        a = vals[(flag == 1)]; b = vals[(flag == 0)]
        return float(a.mean() - b.mean()), int((flag == 1).sum()), int((flag == 0).sum())

    out = {
        "model": args.model, "n_rounds": n, "n_games": games,
        "Vhat_state_R2_on_delta_test": round(r2, 4),
        "delta_std": round(float(y.std()), 1),
        "residual_std": round(float(resid.std()), 1),
        "RAW_delta_gap": {
            "riichi_minus_noriichi": round(gap(riichi, y)[0], 1),
            "call_minus_nocall": round(gap(call, y)[0], 1),
            "won_minus_notwon": round(gap(won, y)[0], 1),
        },
        "RESIDUAL_gap_state_removed": {
            "riichi_minus_noriichi": round(gap(riichi, resid)[0], 1),
            "call_minus_nocall": round(gap(call, resid)[0], 1),
            "won_minus_notwon": round(gap(won, resid)[0], 1),
        },
        "n_riichi": int(riichi.sum()), "n_call": int(call.sum()), "n_won": int(won.sum()),
        "interpretation": "RESIDUAL gap >> 0 => action carries influence beyond state "
                          "(improvement component survives). RESIDUAL gap ~ 0 => raw gap was state confound.",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

