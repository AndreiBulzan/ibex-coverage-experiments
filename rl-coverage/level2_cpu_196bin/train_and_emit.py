"""Train PPO on the non-JAL env, roll out a program, save to /tmp/rl_program.json
for real-RTL validation."""
import argparse, json, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cpu_env_nojal import IbexCpuNoJalEnv, NON_JAL
from shadow_cpu import Op, bins_for_step, WRITERS, BIN_NAMES


def rollout_policy(model, n: int, seed: int):
    env = IbexCpuNoJalEnv(episode_steps=n, seed=seed)
    obs, _ = env.reset(seed=seed)
    seq, covered = [], set()
    prev_w, prev_r = None, None
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=False)
        op_idx = int(action[0]); rd = int(action[1]); rs1 = int(action[2]); rs2 = int(action[3])
        op_enum = NON_JAL[op_idx]
        for b in bins_for_step(int(op_enum), rd, rs1, rs2, 0, prev_w, prev_r):
            covered.add(b)
        seq.append((int(op_enum), rd, rs1, rs2))
        if op_enum in WRITERS:
            prev_w, prev_r = op_enum, rd
        else:
            prev_w, prev_r = None, None
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=seed+1); prev_w = prev_r = None
    return seq, sorted(BIN_NAMES[b] for b in covered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=150_000)
    ap.add_argument("--rollout-n", type=int, default=2000)
    ap.add_argument("--episode-steps", type=int, default=256)
    ap.add_argument("--out", default="/tmp/rl_program.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def make_env(): return IbexCpuNoJalEnv(episode_steps=args.episode_steps, seed=args.seed)
    vec = DummyVecEnv([make_env for _ in range(4)])
    model = PPO("MlpPolicy", vec,
                learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4,
                gamma=0.99, ent_coef=0.03, verbose=0, seed=args.seed, device="cpu")
    print(f"Training PPO (no-JAL) for {args.ppo_steps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, progress_bar=False)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Rolling out {args.rollout_n} instructions...")
    seq, shadow_hit = rollout_policy(model, args.rollout_n, seed=args.seed)
    print(f"  shadow predicts {len(shadow_hit)}/196 bins (non-JAL ceiling 179)")

    with open(args.out, "w") as f:
        json.dump({"n_instructions": len(seq), "seed": args.seed,
                   "sequence": seq, "shadow_hit_bins": shadow_hit}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
