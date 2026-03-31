import random
import numpy as np


class ReliabilityEnv:
    def __init__(self, config):
        self.cfg = config
        self.rng = random.Random()
        reward_cfg = self.cfg["reward"]
        self.max_penalty = (
            reward_cfg["latency_weight"] * 2.0
            + reward_cfg["error_weight"] * 1.0
            + reward_cfg["cpu_weight"] * 2.0
        )
        self.reset()

    def seed(self, seed=None):
        self.rng.seed(seed)
        return seed

    def reset(self):
        self.latency = self.rng.uniform(0.3, 0.6)
        self.cpu = self.rng.uniform(0.3, 0.6)
        self.error_rate = self.rng.uniform(0.0, 0.1)
        self.traffic = self.rng.uniform(*self.cfg["traffic"]["base"])
        return self._get_obs()

    def step(self, action):
        self._apply_action(action)
        self._apply_traffic()
        self._apply_noise()
        self._apply_failures()
        self._clamp_state()

        reward = self._compute_reward()
        done = self.latency >= 2.0 or self.cpu >= 2.0 or self.error_rate >= 1.0

        return self._get_obs(), reward, done, {}

    # -----------------------------
    # Dynamics
    # -----------------------------

    def _apply_action(self, action):
        dyn = self.cfg["dynamics"]

        if action == 2:  # scale up
            self.latency *= self.rng.uniform(*dyn["latency_scale_up"])
            self.cpu *= self.rng.uniform(*dyn["cpu_scale_up"])

        elif action == 0:  # scale down
            self.latency *= self.rng.uniform(*dyn["latency_scale_down"])
            self.cpu *= self.rng.uniform(*dyn["cpu_scale_down"])

        # action == 1 → no-op

    def _apply_traffic(self):
        vol = self.cfg["traffic"]["volatility"]
        self.traffic *= self.rng.uniform(*vol)

        # traffic affects latency + cpu
        self.latency *= (1 + 0.5 * self.traffic)
        self.cpu *= (1 + 0.7 * self.traffic)

    def _apply_noise(self):
        noise = self.cfg["dynamics"]["noise"]

        self.latency *= self.rng.uniform(*noise["latency"])
        self.cpu *= self.rng.uniform(*noise["cpu"])

    def _apply_failures(self):
        failure_cfg = self.cfg["dynamics"]["failure"]

        if self.rng.random() < failure_cfg["prob"]:
            impact = self.rng.uniform(*failure_cfg["impact"])
            self.latency *= impact
            self.error_rate += 0.2

    def _clamp_state(self):
        self.latency = float(np.clip(self.latency, 0.0, 2.0))
        self.cpu = float(np.clip(self.cpu, 0.0, 2.0))
        self.error_rate = float(np.clip(self.error_rate, 0.0, 1.0))
        self.traffic = float(np.clip(self.traffic, 0.0, 1.5))

    # -----------------------------
    # Reward
    # -----------------------------

    def _compute_reward(self):
        r_cfg = self.cfg["reward"]

        penalty = (
            r_cfg["latency_weight"] * self.latency
            + r_cfg["error_weight"] * self.error_rate
            + r_cfg["cpu_weight"] * self.cpu
        )

        reward = 1.0 - (penalty / self.max_penalty)
        return float(np.clip(reward, 0.0, 1.0))

    # -----------------------------
    # Observation
    # -----------------------------

    def _get_obs(self):
        return np.array([
            self.latency,
            self.cpu,
            self.error_rate,
            self.traffic
        ], dtype=np.float32)
