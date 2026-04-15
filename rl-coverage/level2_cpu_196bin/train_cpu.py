"""Random baseline vs PPO on the 196-bin CPU coverage env.

Same structure as train.py but for the sequential CPU environment where
RL should have a real edge: the 143 RAW_HAZARD bins require chaining
writer->reader with matching register, something uniform random only
lucks into 1/32 of the time.
"""

import argparse, time, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cpu_env import IbexCpuEnv
from shadow_cpu import N_BINS, Op
N_OPS = len(Op)


def random_cpu(n_samples: int, seed: int = 0) -> np.ndarray:
    env = IbexCpuEnv(episode_steps=n_samples, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    curve = np.empty(n_samples, dtype=np.int32)
    for t in range(n_samples):
        a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32), rng.integers(32), rng.integers(3)]
        _, _, term, trunc, info = env.step(a)
        curve[t] = info["covered"]
        if term: curve[t:] = info["covered"]; break
    return curve


def ppo_eval(model, n_samples: int, seed: int = 0) -> np.ndarray:
    env = IbexCpuEnv(episode_steps=n_samples, seed=seed)
    obs, _ = env.reset(seed=seed)
    curve = np.empty(n_samples, dtype=np.int32)
    for t in range(n_samples):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(action)
        curve[t] = info["covered"]
        if term: curve[t:] = info["covered"]; break
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=300_000)
    ap.add_argument("--eval-samples", type=int, default=10_000)
    ap.add_argument("--episode-steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== Random CPU: {args.eval_samples} samples ===")
    t0 = time.time()
    rand = random_cpu(args.eval_samples, seed=args.seed)
    print(f"  final: {rand[-1]}/{N_BINS} = {100*rand[-1]/N_BINS:.2f}%  ({time.time()-t0:.1f}s)")

    print(f"\n=== PPO training: {args.ppo_steps} env steps ===")
    def make_env(): return IbexCpuEnv(episode_steps=args.episode_steps, seed=args.seed)
    vec_env = DummyVecEnv([make_env for _ in range(4)])
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4,
        gamma=0.99, ent_coef=0.03,
        verbose=0, seed=args.seed, device="cpu",
    )
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, progress_bar=False)
    print(f"  trained in {time.time()-t0:.1f}s")

    print(f"\n=== PPO eval: {args.eval_samples} samples ===")
    t0 = time.time()
    ppo = ppo_eval(model, args.eval_samples, seed=args.seed)
    print(f"  final: {ppo[-1]}/{N_BINS} = {100*ppo[-1]/N_BINS:.2f}%  ({time.time()-t0:.1f}s)")

    np.savez("curves_cpu.npz", random=rand, ppo=ppo)
    print("\nSaved curves_cpu.npz")

    print(f"\n{'samples':>8} | {'random':>10} | {'PPO':>10}")
    print("-" * 36)
    for n in [100, 500, 1000, 5000, 10000]:
        if n <= len(rand):
            print(f"{n:>8} | {100*rand[n-1]/N_BINS:>8.2f}%  | {100*ppo[n-1]/N_BINS:>8.2f}%")


if __name__ == "__main__":
    main()
