"""Same as cpu_env.py but with the action space restricted to the 13 non-JAL ops,
so the resulting programs run on real RTL without control-flow surprises.

Ceiling: 179/196 bins (drops jal_seen, jal_zero_dst, br_forwards, br_backwards,
and the 13 RAW hazards where JAL is the writer).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shadow_cpu import Op, N_BINS, bins_for_step, WRITERS

NON_JAL = [op for op in Op if op != Op.JAL]
N_OPS = len(NON_JAL)  # 13
N_WRITERS = len(WRITERS)


class IbexCpuNoJalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 512, seed: int | None = None):
        super().__init__()
        self.episode_steps = episode_steps
        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32])
        obs_dim = N_BINS + (N_WRITERS + 1) + 33
        self.observation_space = spaces.MultiBinary(obs_dim)
        self._rng = np.random.default_rng(seed)
        self.covered = np.zeros(N_BINS, dtype=np.int8)
        self.prev_writer = None
        self.prev_rd = None
        self.step_idx = 0

    def _obs(self):
        prev_w = np.zeros(N_WRITERS + 1, dtype=np.int8)
        if self.prev_writer is None:
            prev_w[-1] = 1
        else:
            prev_w[WRITERS.index(self.prev_writer)] = 1
        prev_r = np.zeros(33, dtype=np.int8)
        prev_r[self.prev_rd if self.prev_rd is not None else -1] = 1
        return np.concatenate([self.covered, prev_w, prev_r])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.covered[:] = 0
        self.prev_writer = None
        self.prev_rd = None
        self.step_idx = 0
        return self._obs(), {}

    def step(self, action):
        op_idx, rd, rs1, rs2 = [int(x) for x in action]
        op = NON_JAL[op_idx]  # map compact index back to Op enum
        new_hits = 0
        for b in bins_for_step(int(op), rd, rs1, rs2, 0, self.prev_writer, self.prev_rd):
            if not self.covered[b]:
                self.covered[b] = 1
                new_hits += 1
        if op in WRITERS:
            self.prev_writer, self.prev_rd = op, rd
        else:
            self.prev_writer, self.prev_rd = None, None
        self.step_idx += 1
        terminated = bool(self.covered.sum() == N_BINS)
        truncated = self.step_idx >= self.episode_steps
        return self._obs(), float(new_hits), terminated, truncated, {
            "covered": int(self.covered.sum()),
        }
