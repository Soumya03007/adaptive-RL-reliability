import random
from typing import Any

import numpy as np


ACTION_LABELS = {
    0: "scale_down",
    1: "no_op",
    2: "scale_up",
}


class ReliabilityEnv:
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

        self.last_reward_breakdown: dict[str, float] = {}
        self.last_info: dict[str, Any] = {}

        self.reset()

    def seed(self, seed=None):
        self.rng.seed(seed)
        return seed

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)

        self.latency = self.rng.uniform(0.3, 0.6)
        self.cpu = self.rng.uniform(0.3, 0.6)
        self.error_rate = self.rng.uniform(0.0, 0.1)
        self.traffic = self.rng.uniform(*self.cfg["traffic"]["base"])

        self.last_action = None
        self.last_reward_breakdown = {"penalty": 0.0, "reward": 0.0}

        self.last_info = {
            "metrics": self.snapshot(),
            "action": None,
            "previous_action": None,
            "reward_breakdown": dict(self.last_reward_breakdown),
            "done_reason": None,
        }

        return self._get_obs()

    def step(self, action):
        previous_action = self.last_action

        self._apply_action(action)
        self._apply_traffic()
        self._apply_noise()
        self._apply_failures()
        self._apply_recovery(action)
        self._clamp_state()

        reward, reward_breakdown = self._compute_reward(
            action,
            previous_action,
        )

        done = (
            self.latency >= 2.0
            or self.cpu >= 2.0
            or self.error_rate >= 1.0
        )

        self.last_action = action

        info = {
            "metrics": self.snapshot(),
            "action": ACTION_LABELS.get(action, str(action)),
            "previous_action": (
                ACTION_LABELS.get(previous_action, str(previous_action))
                if previous_action is not None
                else None
            ),
            "reward_breakdown": reward_breakdown,
            "done_reason": self._done_reason(done),
        }

        self.last_reward_breakdown = reward_breakdown
        self.last_info = info

        return self._get_obs(), reward, done, info

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

    def _apply_traffic(self):
        """
        Enhanced workload realism:
        traffic now causes non-linear latency + CPU stress
        so Locust spikes look dramatic in Grafana.
        """
        vol = self.cfg["traffic"]["volatility"]

        # baseline traffic evolution
        self.traffic *= self.rng.uniform(*vol)

        base_low, base_high = self.cfg["traffic"]["base"]
        base_mid = (base_low + base_high) / 2.0

        reversion = self.cfg["traffic"].get("reversion", 0.0)
        if reversion:
            self.traffic = (
                (1 - reversion) * self.traffic
                + reversion * base_mid
            )

        # 🔥 additional burstiness for Locust realism
        burst_factor = self.rng.uniform(0.9, 1.25)
        self.traffic *= burst_factor

        # 🔥 non-linear traffic pressure
        pressure = self.traffic ** 1.4

        # stronger latency sensitivity
        self.latency *= 1 + 0.8 * pressure

        # stronger CPU saturation curve
        self.cpu *= 1 + 0.9 * pressure

        # overload creates errors
        if self.traffic > 1.0:
            overload = self.traffic - 1.0
            self.error_rate += overload * 0.08

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
        recovery = self.cfg["dynamics"].get("recovery", {})
        if not recovery:
            return

        self.latency *= self.rng.uniform(
            *recovery.get("latency_decay", [1.0, 1.0])
        )

        self.cpu *= self.rng.uniform(
            *recovery.get("cpu_decay", [1.0, 1.0])
        )

        self.error_rate = max(
            0.0,
            self.error_rate
            - self.rng.uniform(
                *recovery.get("error_decay", [0.0, 0.0])
            ),
        )

        if action == 2:
            self.error_rate *= self.rng.uniform(
                *recovery.get("scale_up_error_multiplier", [1.0, 1.0])
            )

        elif action == 0:
            self.error_rate *= self.rng.uniform(
                *recovery.get("scale_down_error_multiplier", [1.0, 1.0])
            )

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

        latency_penalty = r_cfg["latency_weight"] * self.latency
        error_penalty = r_cfg["error_weight"] * self.error_rate
        cpu_penalty = r_cfg["cpu_weight"] * self.cpu

        penalty = latency_penalty + error_penalty + cpu_penalty

        scale_penalty = 0.0
        if action in (0, 2):
            scale_penalty = r_cfg.get("scale_action_weight", 0.0)
            penalty += scale_penalty

        action_change_penalty = 0.0
        if previous_action is not None and action != previous_action:
            action_change_penalty = r_cfg.get(
                "action_change_weight",
                0.0,
            )
            penalty += action_change_penalty

        reward = float(
            np.clip(1.0 - (penalty / self.max_penalty), 0.0, 1.0)
        )

        return reward, {
            "latency_penalty": float(latency_penalty),
            "error_penalty": float(error_penalty),
            "cpu_penalty": float(cpu_penalty),
            "scale_penalty": float(scale_penalty),
            "action_change_penalty": float(action_change_penalty),
            "penalty": float(penalty),
            "reward": reward,
        }

    # -----------------------------
    # Observation
    # -----------------------------

    def _get_obs(self):
        return np.array(
            [
                self.latency,
                self.cpu,
                self.error_rate,
                self.traffic,
            ],
            dtype=np.float32,
        )

    def snapshot(self):
        return {
            "latency": float(self.latency),
            "cpu": float(self.cpu),
            "error_rate": float(self.error_rate),
            "traffic": float(self.traffic),
        }

    def _done_reason(self, done):
        if not done:
            return None
        if self.latency >= 2.0:
            return "latency_limit"
        if self.cpu >= 2.0:
            return "cpu_limit"
        if self.error_rate >= 1.0:
            return "error_rate_limit"
        return "terminated"