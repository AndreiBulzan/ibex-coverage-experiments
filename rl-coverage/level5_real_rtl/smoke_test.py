"""Smoke test: verify the real-RTL pipeline works end-to-end.

Runs three episodes:
  1. Random actions.
  2. Shadow-pretrained PPO (loaded from level4_full_5615bin/ppo_l6.zip).
  3. Same PPO again (sanity for variance).

Reports per-episode toggle / branch / line coverage, plus cumulative across the
three episodes. If the pretrained PPO meaningfully exceeds random on episode 1,
that's the warm-start working without any fine-tuning.
"""

import time
import numpy as np

from real_rtl_env import RealRTLToggleEnv, LEVEL6_DIR, ML4DV_DIR, COVDAT
import cov_parser

import sys
sys.path.insert(0, str(LEVEL6_DIR))
from cpu_env_l6 import N_OPS
from shadow_cpu_l6 import IMM_BUCKETS as N_IMM_BUCKETS

EP_STEPS = 1024


def random_episode(env, rng):
    env.reset()
    last_info = {}
    for _ in range(EP_STEPS):
        a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
             rng.integers(32), rng.integers(N_IMM_BUCKETS)]
        _, r, term, trunc, info = env.step(a)
        last_info = info
        if term or trunc:
            break
    return last_info, r


def ppo_episode(env, model):
    obs, _ = env.reset()
    last_info = {}
    for _ in range(EP_STEPS):
        action, _ = model.predict(obs, deterministic=False)
        obs, r, term, trunc, info = env.step(action)
        last_info = info
        if term or trunc:
            break
    return last_info, r


def main():
    print(f"Vtop:    {ML4DV_DIR / 'sim_build' / 'Vtop'}")
    print(f"covdat:  {COVDAT}")
    print(f"Episode steps: {EP_STEPS}")

    print("\n--- Episode 1: random actions ---")
    env = RealRTLToggleEnv(episode_steps=EP_STEPS, seed=42, kind="toggle")
    rng = np.random.default_rng(42)
    t0 = time.time()
    info, r = random_episode(env, rng)
    print(f"  ep took {time.time()-t0:.1f}s")
    print(f"  reward (new toggle hits): {r:.0f}")
    print(f"  toggle: {info['ep_real_covered']}/{info['ep_real_total']} "
          f"= {info['ep_real_pct']:.2f}%")

    # Also report branch + line for context
    s = cov_parser.parse(str(COVDAT))
    for k in ("toggle", "branch", "line"):
        c, t = s.by_kind[k]
        print(f"  {k:7s} (whole-file): {c}/{t} = {100*c/t:.2f}%")

    print("\n--- Loading ppo_l6.zip ---")
    from stable_baselines3 import PPO
    model = PPO.load(str(LEVEL6_DIR / "ppo_l6.zip"), device="cpu")
    print("  loaded.")

    for trial in (2, 3):
        print(f"\n--- Episode {trial}: PPO (warm-start) ---")
        t0 = time.time()
        info, r = ppo_episode(env, model)
        print(f"  ep took {time.time()-t0:.1f}s")
        print(f"  reward (new toggle hits): {r:.0f}")
        print(f"  toggle: {info['ep_real_covered']}/{info['ep_real_total']} "
              f"= {info['ep_real_pct']:.2f}%")
        s = cov_parser.parse(str(COVDAT))
        for k in ("toggle", "branch", "line"):
            c, t = s.by_kind[k]
            print(f"  {k:7s} (whole-file): {c}/{t} = {100*c/t:.2f}%")


if __name__ == "__main__":
    main()
