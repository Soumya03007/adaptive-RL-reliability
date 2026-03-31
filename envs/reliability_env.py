import random
import numpy as np


class ReliabilityEnv:
    OBS_SIZE = 7

    def __init__(self, config):
        self.cfg = config
        self.rng = random.Random()
        reward_cfg = self.cfg["reward"]
        self.max_penalty = (
            reward_cfg["latency_weight"] * 2.0
            + reward_cfg["error_weight"] * 1.0
            + reward_cfg["cpu_weight"] * 2.0
            + reward_cfg.get("scale_action_weight", 0.0)
            + reward_cfg.get("action_change_weight", 0.0)
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
        self.last_action = None
        return self._get_obs()

    def step(self, action):
        previous_action = self.last_action
        self._apply_action(action)
        self._apply_traffic()
        self._apply_noise()
        self._apply_failures()
        self._apply_recovery(action)
        self._clamp_state()

        reward = self._compute_reward(action, previous_action)
        done = self.latency >= 2.0 or self.cpu >= 2.0 or self.error_rate >= 1.0
        self.last_action = action

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

    def _apply_recovery(self, action):
        recovery_cfg = self.cfg["dynamics"].get("recovery", {})
        base = recovery_cfg.get("error_base", [0.96, 1.0])
        scale_up = recovery_cfg.get("error_scale_up", [0.72, 0.9])
        no_op = recovery_cfg.get("error_no_op", base)
        scale_down = recovery_cfg.get("error_scale_down", [0.98, 1.05])

        if action == 2:
            factor = self.rng.uniform(*scale_up)
        elif action == 1:
            factor = self.rng.uniform(*no_op)
        else:
            factor = self.rng.uniform(*scale_down)

        self.error_rate *= factor

    def _clamp_state(self):
        self.latency = float(np.clip(self.latency, 0.0, 2.0))
        self.cpu = float(np.clip(self.cpu, 0.0, 2.0))
        self.error_rate = float(np.clip(self.error_rate, 0.0, 1.0))
        self.traffic = float(np.clip(self.traffic, 0.0, 1.5))

    # -----------------------------
    # Reward
    # -----------------------------

    def _compute_reward(self, action, previous_action):
        r_cfg = self.cfg["reward"]

        penalty = (
            r_cfg["latency_weight"] * self.latency
            + r_cfg["error_weight"] * self.error_rate
            + r_cfg["cpu_weight"] * self.cpu
        )

        if action in (0, 2):
            penalty += r_cfg.get("scale_action_weight", 0.0)

        if previous_action is not None and action != previous_action:
            penalty += r_cfg.get("action_change_weight", 0.0)

        reward = 1.0 - (penalty / self.max_penalty)
        return float(np.clip(reward, 0.0, 1.0))

    # -----------------------------
    # Observation
    # -----------------------------

    def _get_obs(self):
        last_action = np.zeros(3, dtype=np.float32)
        if self.last_action is not None:
            last_action[self.last_action] = 1.0

        return np.array(
            [
                self.latency,
                self.cpu,
                self.error_rate,
                self.traffic,
                *last_action,
            ],
            dtype=np.float32,
        )
