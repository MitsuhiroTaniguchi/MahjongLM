"""Dissect why CCC fails online: does the live encoding (leading <bos>, etc.)
change c*'s within-state ranking vs the offline dataset encoding it was fit on?

For yonma imperfect dataset games, at each viewer discard decision, compute c*
for every in-support candidate TWO ways:
  (A) offline format: prefix = ids[:pos]              (no <bos>, as the head was fit)
  (B) live format   : prefix = [<bos>] + ids[:pos]    (as the JS player sends)
and compare the argmax-c* pick and the c* vectors.  Also report how often the
c*-best equals the EXPERT's actual discard (a move-quality sanity check) under
each encoding.
"""
from __future__ import annotations

import argparse
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
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_(.+)$")
FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="outputs/ccc_release/ccc_head.npz")
    ap.add_argument("--shards", nargs="+", default=["2024/data-00002-of-00016.arrow"])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--support-thresh", type=float, default=0.02)
    args = ap.parse_args()
    device = "cuda"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    vocab = tok.get_vocab()
    bos = vocab["<bos>"]
    base = AutoModelForCausalLM.from_pretrained("mitsutani/mahjonglm-10m", dtype=torch.float32,
                                                output_hidden_states=True).to(device).eval()
    H = np.load(args.head); t = lambda k: torch.tensor(H[k].astype(np.float32)).to(device)
    SMU, SSD, PMU, PSD, WM, GMEAN, WVEC = (t("smu"), t("ssd"), t("pmu"), t("psd"), t("Wm"), t("gmean"), t("w"))

    def cstar(state_h, phi):
        ss = (state_h - SMU) / SSD
        Es = torch.cat([ss, torch.ones(1, device=device)]) @ WM
        g = (phi - PMU) / PSD - Es.unsqueeze(0) - GMEAN.unsqueeze(0)
        return g @ WVEC

    # discard candidate token-ids per seat
    seat_disc = {}
    for tk, ti in vocab.items():
        m = DISCARD_RE.match(tk)
        if m:
            seat_disc.setdefault(int(m.group(1)), []).append(ti)

    @torch.no_grad()
    def cstar_over(prefix_ids, cand_ids, off):
        # off = number of leading tokens that are framing (to locate state pos)
        out = base(torch.tensor([prefix_ids], device=device))
        logits = out.logits[0, -1].float()
        state_h = out.hidden_states[-1][0, -1]
        cand = torch.tensor(cand_ids, device=device)
        lp = torch.log_softmax(logits[cand], -1); pi0 = lp.exp()
        batch = torch.tensor([prefix_ids + [c] for c in cand_ids], device=device)
        phi = base(batch).hidden_states[-1][:, -1, :]
        return pi0, cstar(state_h, phi)

    ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", args.shards[0], repo_type="dataset"))
    n_dec = 0; agree_pick = 0; corr_sum = 0.0; n_corr = 0
    expert_match = {"A": 0, "B": 0}
    g = 0
    for row in ds:
        if row["view_type"] != "imperfect" or int(row["seat_count"]) != 4 or row["length"] > 2600:
            continue
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        if not any(FINAL_RANK_RE.match(t) for t in toks):
            continue
        seat = None
        for i, tkn in enumerate(toks):
            if tkn == "round_start":
                seat = None
            elif tkn.startswith("draw_") and not tkn.endswith("_hidden"):
                seat = int(tkn.split("_")[1])
            elif seat is not None and DISCARD_RE.match(tkn) and int(tkn.split("_")[1]) == seat and i > 0:
                cand = seat_disc.get(seat, [])
                if len(cand) < 2:
                    continue
                # in-support from offline forward
                pi0A, cA = cstar_over(ids[:i], cand, 0)
                supA = pi0A >= args.support_thresh
                if int(supA.sum()) < 2:
                    continue
                pi0B, cB = cstar_over([bos] + ids[:i], cand, 1)
                idxA = torch.nonzero(supA).squeeze(-1)
                bestA = idxA[cA[idxA].argmax()].item()
                bestB = idxA[cB[idxA].argmax()].item()
                agree_pick += int(bestA == bestB)
                # corr of c* across in-support candidates (A vs B)
                ca = cA[idxA] - cA[idxA].mean(); cb = cB[idxA] - cB[idxA].mean()
                d = float(ca.norm() * cb.norm())
                if d > 1e-9:
                    corr_sum += float((ca @ cb) / d); n_corr += 1
                # expert move
                aid = ids[i]
                if aid in cand:
                    ei = cand.index(aid)
                    if supA[ei]:
                        expert_match["A"] += int(bestA == ei)
                        expert_match["B"] += int(bestB == ei)
                n_dec += 1
        g += 1
        if g >= args.games:
            break
    import json
    print(json.dumps({
        "n_decisions": n_dec,
        "argmax_c*_agree_offline_vs_live(bos)": round(agree_pick / max(n_dec, 1), 4),
        "mean_within_state_c*_corr_offline_vs_live": round(corr_sum / max(n_corr, 1), 4),
        "c*_best_equals_expert_discard": {"offline_A": round(expert_match["A"] / max(n_dec, 1), 4),
                                          "live_bos_B": round(expert_match["B"] / max(n_dec, 1), 4)},
        "note": "if corr/agree are LOW, the leading <bos> (live) materially changes c* vs the "
                "no-bos encoding the head was fit on => online distribution mismatch (bug/認識相違).",
    }, indent=2))


if __name__ == "__main__":
    main()
