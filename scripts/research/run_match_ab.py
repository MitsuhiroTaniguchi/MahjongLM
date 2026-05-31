"""Parallel A/B match driver: base (argmax pi0) vs CCC (argmax c*) on identical
seeds, against 3 kobalab AIs at the other seats.  Runs LANES server processes +
LANES node workers per mode (seed-sharded) for throughput, base then CCC, then
prints the PAIRED (same-seed) placement / dan-point comparison.
"""
from __future__ import annotations

import argparse
import os
import re
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MAJIANG = Path(r"C:/Users/taniguchi/Documents/Majiang")
REPO = Path(r"C:/Users/taniguchi/Documents/MahjongLM_OutcomeConditioned_Policy")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
HEAD = str(REPO / "outputs" / "ccc_release" / "ccc_head.npz")
LOGDIR = REPO / "outputs" / "match_ab"
HDR = re.compile(r"seed=(\d+)")
ROW = re.compile(r"\[MahjongLM\([^)]*\)\]\s*MahjongLM\([^)]*\):\s*(-?\d+)点\s*\((\d)位,\s*([+-]?\d+\.?\d*)\)")


def wait_health(port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def run_mode(server_mode, n, lanes, base_port, tag, beta=3.0):
    chunk = math.ceil(n / lanes)
    workers, logs = [], []
    mode = tag
    # ONE shared GPU server (LRU prefix cache serves all lanes); workers parallel on CPU
    senv = dict(os.environ, MAHJONGLM_MODEL="mitsutani/mahjonglm-10m", CCC_HEAD=HEAD,
                CCC_MODE=server_mode, CCC_BETA=str(beta), CCC_MAX_GAMES=str(lanes + 4))
    server = subprocess.Popen([PY, str(MAJIANG / "mahjonglm-ccc-server.py"), str(base_port)],
                              cwd=str(MAJIANG), env=senv,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_health(base_port):
        print(f"server {base_port} failed health")
    print(f"[{mode}] server up on {base_port}, launching {lanes} workers")
    for i in range(lanes):
        start = i * chunk
        cnt = min(chunk, n - start)
        if cnt <= 0:
            break
        wenv = dict(os.environ, SEAT0=f"mahjonglm:{base_port}", SEAT1="ai", SEAT2="ai", SEAT3="ai",
                    YONMA_GAMES=str(cnt), SANMA_GAMES="0", SEED_START=str(start), NO_PAIPU="1")
        lf = open(LOGDIR / f"{mode}_{i}.log", "w", encoding="utf-8")
        logs.append(LOGDIR / f"{mode}_{i}.log")
        workers.append(subprocess.Popen(["node", "run-match.js"], cwd=str(MAJIANG), env=wenv,
                                        stdout=lf, stderr=subprocess.STDOUT))
    for w in workers:
        w.wait()
    server.terminate()
    try:
        server.wait(timeout=10)
    except Exception:
        server.kill()
    # parse
    res = {}
    for lf in logs:
        seed = None
        for line in open(lf, encoding="utf-8", errors="ignore"):
            h = HDR.search(line)
            if "半荘終了" in line and h:
                seed = int(h.group(1))
            m = ROW.search(line)
            if m and seed is not None:
                res[seed] = {"rank": int(m.group(2)), "point": float(m.group(3))}
                seed = None
    return res


def stats(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd, n


def compare(B, V, tag):
    seeds = sorted(set(B) & set(V))
    if not seeds:
        print(f"[{tag}] no paired games"); return
    br = [B[s]["rank"] for s in seeds]; vr = [V[s]["rank"] for s in seeds]
    bp = [B[s]["point"] for s in seeds]; vp = [V[s]["point"] for s in seeds]
    mbr, _, _ = stats(br); mvr, _, _ = stats(vr); mbp, _, _ = stats(bp); mvp, _, _ = stats(vp)
    dr = [V[s]["rank"] - B[s]["rank"] for s in seeds]
    dp = [V[s]["point"] - B[s]["point"] for s in seeds]
    mdr, sdr, n = stats(dr); mdp, sdp, _ = stats(dp)
    ser = sdr / math.sqrt(n) if n else 0; sep = sdp / math.sqrt(n) if n else 0
    print(f"\n[{tag} vs base]  paired={n}")
    print(f"  avg_rank : base {mbr:.3f}  {tag} {mvr:.3f}   d={mdr:+.3f}+/-{ser:.3f} t={mdr/ser if ser else 0:+.2f} (neg better)")
    print(f"  avg_point: base {mbp:+.2f}  {tag} {mvp:+.2f}   d={mdp:+.2f}+/-{sep:.2f} t={mdp/sep if sep else 0:+.2f} (pos better)")
    print(f"  rank {tag} better/worse/same: {sum(d<0 for d in dr)}/{sum(d>0 for d in dr)}/{sum(d==0 for d in dr)}"
          f"   point >/</=: {sum(d>0 for d in dp)}/{sum(d<0 for d in dp)}/{sum(d==0 for d in dp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=120)
    ap.add_argument("--lanes", type=int, default=12)
    ap.add_argument("--variants", nargs="+", default=["ccc", "worst", "random"])
    args = ap.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    B = run_mode("base", args.n, args.lanes, 8901, "base")
    res = {}
    for v in args.variants:
        res[v] = run_mode(v, args.n, args.lanes, 8901, v)
    print(f"\n===== match A/B ({time.time()-t0:.0f}s, n={args.n}) =====")
    print(f"base avg_rank {stats([B[s]['rank'] for s in B])[0]:.3f}  avg_point {stats([B[s]['point'] for s in B])[0]:+.2f}")
    for v in args.variants:
        compare(B, res[v], v)


if __name__ == "__main__":
    main()
