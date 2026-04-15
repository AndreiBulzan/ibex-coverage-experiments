"""1-episode smoke test: RVC-only program on real Ibex.

Goal: confirm every RVC op decodes without raising illegal-instruction and
that the compressed_decoder module actually activates (toggle coverage on
its signals should be nonzero after this run).
"""

import os, sys, time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from codec_rvc import Op, N_OPS, IMM_BUCKETS, RVC_ALL
from env_rvc import IbexRVCEnv, run_program

RVC_OP_IDS = [int(op) for op in Op if op in RVC_ALL]


def main():
    print(f"RVC ops = {len(RVC_OP_IDS)}: " + ", ".join(Op(i).name for i in RVC_OP_IDS))
    print()

    # ---- Phase 1: RVC-only program, all 16 ops × several imm buckets ----
    actions = []
    for bucket in range(IMM_BUCKETS):
        for op_id in RVC_OP_IDS:
            # rd/rs1/rs2 chosen to land in safe ranges for every op
            actions.append((op_id, 5, 6, 7, bucket))
    # Repeat pattern to fill ~256 instructions (enough for meaningful coverage)
    actions = (actions * (256 // len(actions) + 1))[:256]

    print(f"Phase 1: {len(actions)} actions, RVC-only")
    t0 = time.time()
    summary = run_program(actions)
    dt = time.time() - t0
    assert summary is not None, "Vtop failed on RVC-only program"
    cov, tot = summary.by_kind["toggle"]
    print(f"  toggle: {cov}/{tot} = {100*cov/tot:.2f}%   ({dt:.1f}s)")
    print(f"  branch: {summary.by_kind['branch'][0]}/{summary.by_kind['branch'][1]} "
          f"= {summary.kind_pct('branch'):.2f}%")
    print(f"  line:   {summary.by_kind['line'][0]}/{summary.by_kind['line'][1]} "
          f"= {summary.kind_pct('line'):.2f}%")

    # Check compressed_decoder got exercised
    cd_pages = [p for p in summary.by_page if "compressed_decoder" in p]
    print(f"\n  compressed_decoder coverage:")
    for p in sorted(cd_pages):
        c, t = summary.by_page[p]
        print(f"    {p:55s}  {c:>3}/{t:<3} = {100*c/t:.1f}%")

    # ---- Phase 2: same via env (sanity) ----
    print("\nPhase 2: 1 random episode through IbexRVCEnv")
    import numpy as np
    rng = np.random.default_rng(0)
    env = IbexRVCEnv(episode_steps=1024, seed=0, kind="toggle")
    env.reset()
    info = {}
    t0 = time.time()
    for _ in range(1024):
        a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
             rng.integers(32), rng.integers(IMM_BUCKETS)]
        obs, r, term, trunc, info = env.step(a)
        if trunc: break
    dt = time.time() - t0
    print(f"  reward: {r:.0f}")
    print(f"  ep_pct: {info.get('ep_pct', 0):.2f}%")
    print(f"  branch_pct: {info.get('branch_pct', 0):.2f}%")
    print(f"  line_pct:   {info.get('line_pct', 0):.2f}%")
    print(f"  wall: {dt:.1f}s")


if __name__ == "__main__":
    main()
