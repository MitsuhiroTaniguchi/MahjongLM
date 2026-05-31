"""Root cause part B: does argmax_a c* systematically OVERRIDE the expert toward
rare (low-pi0) actions?  c* monotonically ranks TAKEN actions by return
(observational), but argmax picks the highest-c* IN-SUPPORT candidate -- which
may be an action the expert rarely takes.  If so, conditioning forces anti-expert
moves => loses in play (interventional != observational).

On dataset yonma discards (bos-correct), per decision compute c* for every
in-support candidate and report:
  - P(argmax c* == expert's actual discard)
  - mean pi0 of the argmax-c* candidate vs the expert's action
  - mean c* margin (argmax - expert)  in c* std units
  - expert action's rank in the c* ordering (1 = expert is the c* best)
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
    ap.add_argument("--shards", nargs="+", default=["2024/data-00003-of-00016.arrow"])
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--support-thresh", type=float, default=0.02)
    args = ap.parse_args()
    dev = "cuda"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer"); V = tok.get_vocab()
    bos = V["<bos>"]
    base = AutoModelForCausalLM.from_pretrained("mitsutani/mahjonglm-10m", dtype=torch.float32,
                                                output_hidden_states=True).to(dev).eval()
    torch.set_grad_enabled(False)
    H = np.load(args.head); t = lambda k: torch.tensor(H[k].astype(np.float32)).to(dev)
    SMU, SSD, PMU, PSD, WM, GMEAN, WVEC = t("smu"), t("ssd"), t("pmu"), t("psd"), t("Wm"), t("gmean"), t("w")
    cstd = float(np.std(np.load("outputs/research/ccc_features_all.npz")["phi"][:5000]))  # rough scale ref

    def cstar(state_h, phi):
        ss = (state_h - SMU) / SSD
        Es = torch.cat([ss, torch.ones(1, device=dev)]) @ WM
        g = (phi - PMU) / PSD - Es.unsqueeze(0) - GMEAN.unsqueeze(0)
        return g @ WVEC

    seat_disc = {}
    for tk, ti in V.items():
        m = DISCARD_RE.match(tk)
        if m:
            seat_disc.setdefault(int(m.group(1)), []).append(ti)

    ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", args.shards[0], repo_type="dataset"))
    agree = 0; n = 0; pi0_pick = []; pi0_exp = []; margins = []; exp_rank = []; all_c = []
    g = 0
    for row in ds:
        if row["view_type"] != "imperfect" or int(row["seat_count"]) != 4 or row["length"] > 2600:
            continue
        ids = [int(x) for x in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        if not any(FINAL_RANK_RE.match(x) for x in toks):
            continue
        seat = None
        for i, tkn in enumerate(toks):
            if tkn == "round_start":
                seat = None
            elif tkn.startswith("draw_") and not tkn.endswith("_hidden"):
                seat = int(tkn.split("_")[1])
            elif seat is not None and DISCARD_RE.match(tkn) and int(tkn.split("_")[1]) == seat and i > 0:
                expert = ids[i]
                cand = seat_disc.get(seat, [])
                if expert not in cand or len(cand) < 2:
                    continue
                pref = [bos] + ids[:i]
                out = base(torch.tensor([pref], device=dev))
                logits = out.logits[0, -1].float(); state_h = out.hidden_states[-1][0, -1]
                ct = torch.tensor(cand, device=dev)
                lp = torch.log_softmax(logits[ct], -1); pi0 = lp.exp()
                sup = pi0 >= args.support_thresh
                if int(sup.sum()) < 2:
                    continue
                batch = torch.tensor([pref + [c] for c in cand], device=dev)
                phi = base(batch).hidden_states[-1][:, -1, :]
                c = cstar(state_h, phi).masked_fill(~sup, float("-inf"))
                bi = int(c.argmax())
                ei = cand.index(expert)
                agree += int(cand[bi] == expert)
                pi0_pick.append(float(pi0[bi])); pi0_exp.append(float(pi0[ei]))
                cc = c[sup]
                margins.append((float(c[bi]) - float(c[ei])) / (float(cc.std()) + 1e-9))
                # expert's rank by c* among in-support (1=best)
                order = torch.argsort(c, descending=True)
                rank = int((order == ei).nonzero()[0]) + 1
                exp_rank.append(rank)
                n += 1
        g += 1
        if g >= args.games:
            break
    import json
    print(json.dumps({
        "n_decisions": n,
        "P(argmax_c*==expert)": round(agree / max(n, 1), 4),
        "mean_pi0_of_argmax_c*_pick": round(float(np.mean(pi0_pick)), 4),
        "mean_pi0_of_expert_action": round(float(np.mean(pi0_exp)), 4),
        "mean_c*_margin_(argmax-expert)_in_std": round(float(np.mean(margins)), 3),
        "mean_expert_rank_in_c*_order": round(float(np.mean(exp_rank)), 2),
        "interpretation": "if P(argmax==expert) is LOW and argmax pi0 << expert pi0, then argmax c* "
                          "systematically picks RARE (anti-expert) actions -> conditioning overrides "
                          "the expert -> loses. c* is observational (expert TAKES high-c* in good spots), "
                          "not interventional.",
    }, indent=2))


if __name__ == "__main__":
    main()
