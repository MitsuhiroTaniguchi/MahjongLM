"""Parse two run-match.js stdout logs (base vs CCC) and compute a PAIRED
(same-seed) comparison of the MahjongLM seat's placement and dan-points.

Each log has per-game lines:
  ===== 第N半荘終了 (yonma, seed=K, 観戦対象席:0) =====
    [MahjongLM(PORT)] MahjongLM(PORT): 31800点 (1位, +41.8)
We extract seed -> (rank, point) for the MahjongLM player and pair by seed.
"""
from __future__ import annotations

import argparse
import re
import math

HDR = re.compile(r"seed=(\d+)")
ROW = re.compile(r"\[MahjongLM\([^)]*\)\]\s*MahjongLM\([^)]*\):\s*(-?\d+)点\s*\((\d)位,\s*([+-]?\d+\.?\d*)\)")


def parse(path):
    out = {}
    seed = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        h = HDR.search(line)
        if "半荘終了" in line and h:
            seed = int(h.group(1))
        m = ROW.search(line)
        if m and seed is not None:
            out[seed] = {"score": int(m.group(1)), "rank": int(m.group(2)), "point": float(m.group(3))}
            seed = None
    return out


def stats(xs):
    n = len(xs); mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("ccc")
    args = ap.parse_args()
    B = parse(args.base); C = parse(args.ccc)
    seeds = sorted(set(B) & set(C))
    print(f"base games={len(B)} ccc games={len(C)} paired={len(seeds)}")
    if not seeds:
        return
    dr = [C[s]["rank"] - B[s]["rank"] for s in seeds]      # negative = CCC better rank
    dp = [C[s]["point"] - B[s]["point"] for s in seeds]    # positive = CCC more points
    br = [B[s]["rank"] for s in seeds]; cr = [C[s]["rank"] for s in seeds]
    bp = [B[s]["point"] for s in seeds]; cp = [C[s]["point"] for s in seeds]
    for label, b, c in [("avg_rank", br, cr), ("avg_point", bp, cp)]:
        mb, sb, _ = stats(b); mc, sc, _ = stats(c)
        print(f"{label}: base={mb:.3f}(sd {sb:.2f})  ccc={mc:.3f}(sd {sc:.2f})")
    mdr, sdr, n = stats(dr); mdp, sdp, _ = stats(dp)
    se_r = sdr / math.sqrt(n); se_p = sdp / math.sqrt(n)
    print(f"PAIRED d_rank (ccc-base): mean={mdr:+.3f} se={se_r:.3f}  t={mdr/se_r if se_r else 0:+.2f}  (neg=CCC better)")
    print(f"PAIRED d_point(ccc-base): mean={mdp:+.2f} se={se_p:.2f}  t={mdp/se_p if se_p else 0:+.2f}  (pos=CCC better)")
    wins = sum(1 for d in dp if d > 0); losses = sum(1 for d in dp if d < 0); ties = sum(1 for d in dp if d == 0)
    print(f"per-seed point sign: CCC>base {wins}, CCC<base {losses}, tie {ties}")
    rank_better = sum(1 for d in dr if d < 0); rank_worse = sum(1 for d in dr if d > 0)
    print(f"per-seed rank: CCC better {rank_better}, worse {rank_worse}, same {len(seeds)-rank_better-rank_worse}")


if __name__ == "__main__":
    main()
