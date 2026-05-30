"""Controllable Consequence Conditioning (CCC) inference wrapper.

CCC is a *consequence-scoring head* on top of a frozen base mahjong LM. For any
viewer decision it scores each in-support candidate action by the auto-discovered
return-aligned controllable consequence

    c*(s, a) = w . ( std(phi(s,a)) - E[phi|s] )

where phi(s,a) = the base model's ACTION-PROCESSED hidden (hidden at the action
token, after the model attends to it) and E[phi|s] is the policy-averaged
consequence (ridge state->phi).  Higher c* == better expected return.  We then
return the c*-conditioned policy: either argmax c* within the chi^2 trust region
(in-support actions) or a tempered tilt  softmax(log pi0 + beta * c*).

This is decision-type-agnostic: discard / self (riichi/tsumo/kan) / react
(ron/pon/chi/pass) / chi_pos / red / kan_tile all use the SAME c*.

Artifacts (ccc_head.npz):  smu, ssd, pmu, psd, Wm, gmean, w  (+ meta).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

try:  # decision/candidate enumeration helpers (optional, for full-game use)
    from gpt2.viewer_decisions import (iter_viewer_decisions, build_candidate_groups,
                                       candidates_for)
except Exception:  # pragma: no cover
    iter_viewer_decisions = build_candidate_groups = candidates_for = None


class CCCPolicy:
    def __init__(self, model_dir, head_path=None, device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, output_hidden_states=True).to(self.device).eval()
        if head_path is None:
            local = Path(model_dir) / "ccc_head.npz"
            if local.exists():
                head_path = str(local)
            else:  # model_dir is a Hub repo id -> fetch the head
                from huggingface_hub import hf_hub_download
                head_path = hf_hub_download(repo_id=str(model_dir), filename="ccc_head.npz")
        h = np.load(str(head_path))
        t = lambda k: torch.tensor(h[k].astype(np.float32)).to(self.device)
        self.smu, self.ssd = t("smu"), t("ssd")
        self.pmu, self.psd = t("pmu"), t("psd")
        self.Wm, self.gmean, self.w = t("Wm"), t("gmean"), t("w")
        self.meta = {k: h[k].item() for k in ("lam", "sign", "n_fit") if k in h}

    def _cstar(self, state_h, phi_h):
        ss = (state_h - self.smu) / self.ssd
        s1 = torch.cat([ss, torch.ones(len(ss), 1, device=self.device)], 1)
        ps = (phi_h - self.pmu) / self.psd
        g = ps - (s1 @ self.Wm) - self.gmean
        return g @ self.w

    @torch.no_grad()
    def score_candidates(self, prefix_ids, candidate_ids):
        """c*(s,a) for each candidate token id, at the decision after ``prefix_ids``.

        prefix_ids: list[int] up to (exclusive) the decision position.
        candidate_ids: list[int] candidate action token ids.
        Returns a 1-D tensor of c* aligned with candidate_ids.
        """
        out = self.model(torch.tensor([list(prefix_ids)], device=self.device),
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        state_h = out.hidden_states[-1][0, -1:]
        scores = torch.empty(len(candidate_ids), device=self.device)
        for i, tid in enumerate(candidate_ids):
            step = self.model(torch.tensor([[int(tid)]], device=self.device),
                              past_key_values=past, use_cache=True, output_hidden_states=True)
            phi_h = step.hidden_states[-1][0, -1:]
            scores[i] = self._cstar(state_h, phi_h)[0]
        return scores

    @torch.no_grad()
    def conditioned_policy(self, prefix_ids, candidate_ids, support_thresh=0.02, beta=None):
        """Return dict with base policy pi0, c* scores, and the CCC policy over
        ``candidate_ids``.

        - pi0: base log-softmax over the candidate group at the decision.
        - in_support: pi0 >= support_thresh (the chi^2 trust region).
        - best: argmax c* among in-support candidates (recommended action).
        - ccc: if beta is None -> one-hot on ``best``; else softmax(log pi0 + beta*c*)
               restricted to in-support (renormalised).  Both stay in-support.
        """
        prefix_ids = list(prefix_ids); cand = list(candidate_ids)
        logits = self.model(torch.tensor([prefix_ids], device=self.device)).logits[0, -1]
        cid = torch.tensor(cand, device=self.device)
        lp = torch.log_softmax(logits[cid].float(), -1)
        pi0 = lp.exp()
        sup = pi0 >= support_thresh
        if int(sup.sum()) < 1:
            sup = pi0 >= pi0.max()        # fall back to the mode
        c = self.score_candidates(prefix_ids, cand)
        c_sup = c.masked_fill(~sup, float("-inf"))
        best = int(c_sup.argmax())
        if beta is None:
            ccc = torch.zeros(len(cand), device=self.device); ccc[best] = 1.0
        else:
            logits_c = (lp + beta * c).masked_fill(~sup, float("-inf"))
            ccc = torch.softmax(logits_c, -1)
        return {
            "candidate_ids": cand,
            "pi0": pi0.cpu(),
            "cstar": c.cpu(),
            "in_support": sup.cpu(),
            "best_index": best,
            "best_token_id": int(cand[best]),
            "ccc_policy": ccc.cpu(),
        }

    @torch.no_grad()
    def rank_full_game(self, input_ids, viewer_seat, tokenizer, support_thresh=0.02):
        """Convenience: walk an entire game and, at every viewer decision, return
        the recommended (c*-best, in-support) action vs the realised action.

        Requires gpt2.viewer_decisions.  Yields dicts with pos, dtype, the
        realised token, the recommended token, and whether they agree.
        """
        if iter_viewer_decisions is None:
            raise RuntimeError("gpt2.viewer_decisions is required for rank_full_game")
        from gpt2.viewer_decisions import DTYPE_NAMES
        ids = [int(t) for t in input_ids]
        toks = tokenizer.convert_ids_to_tokens(ids)
        groups = build_candidate_groups(tokenizer)
        results = []
        for (pos, dt, seat) in iter_viewer_decisions(toks, viewer_seat):
            if pos == 0:
                continue
            cand = candidates_for(groups, dt, seat)
            if len(cand) < 2:
                continue
            res = self.conditioned_policy(ids[:pos], cand, support_thresh=support_thresh)
            results.append({
                "pos": pos, "dtype": DTYPE_NAMES[dt],
                "realised_token": tokenizer.convert_ids_to_tokens(ids[pos]),
                "recommended_token": tokenizer.convert_ids_to_tokens(res["best_token_id"]),
                "agree": ids[pos] == res["best_token_id"],
            })
        return results
