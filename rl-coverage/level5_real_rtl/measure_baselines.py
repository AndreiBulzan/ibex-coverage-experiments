"""Quick baseline characterisation: how high does pure random push toggle/branch
coverage on this minimal Ibex config across N episodes?

Each episode is independent (CPU resets), so the per-episode max is bounded by
what one program of EP_STEPS instructions can exercise. The cumulative-across-
episodes number is the real ceiling for our 30-op action subset.
"""

import time, sys
import numpy as np
from real_rtl_env import RealRTLToggleEnv, COVDAT, LEVEL6_DIR
sys.path.insert(0, str(LEVEL6_DIR))
from cpu_env_l6 import N_OPS
from shadow_cpu_l6 import IMM_BUCKETS as N_IMM_BUCKETS

import cov_parser

EP_STEPS = 1024
N_EPISODES = 8


def main():
    env = RealRTLToggleEnv(episode_steps=EP_STEPS, seed=42, kind="toggle")
    rng = np.random.default_rng(42)
    cum_hits: set[str] = set()
    cum_branch: set[str] = set()
    print(f"{'ep':>3} | {'ep_toggle%':>10} | {'cum_toggle%':>11} | {'cum_branch%':>11} | {'wall':>5}")
    print("-" * 60)
    for ep in range(N_EPISODES):
        env.reset()
        t0 = time.time()
        for _ in range(EP_STEPS):
            a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
                 rng.integers(32), rng.integers(N_IMM_BUCKETS)]
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
        dt = time.time() - t0
        # accumulate
        s = cov_parser.parse(str(COVDAT))
        prefix_t = "\x01page\x02v_toggle/"
        prefix_b = "\x01page\x02v_branch/"
        toggle_hits = {k for k, v in s.points.items() if v > 0 and prefix_t in ("\x01" + k)}
        branch_hits = {k for k, v in s.points.items() if v > 0 and prefix_b in ("\x01" + k)}
        cum_hits |= toggle_hits
        cum_branch |= branch_hits
        ep_pct = info["ep_real_pct"]
        cum_t_pct = 100.0 * len(cum_hits) / s.by_kind["toggle"][1]
        cum_b_pct = 100.0 * len(cum_branch) / s.by_kind["branch"][1]
        print(f"{ep+1:>3} | {ep_pct:>9.2f}% | {cum_t_pct:>10.2f}% | {cum_b_pct:>10.2f}% | {dt:>4.1f}s")


if __name__ == "__main__":
    main()
