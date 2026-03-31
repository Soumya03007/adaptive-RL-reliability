import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Categorical, Composite, Unbounded
from envs.reliability_env import ReliabilityEnv

class TorchRLEnvWrapper(EnvBase):
    def __init__(self, env, device="cpu"):
        super().__init__(device=device)
        self.env = env
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=torch.Size([ReliabilityEnv.OBS_SIZE]),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.action_spec = Categorical(
            n=3,
            shape=torch.Size([]),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=self.device,
        )
        self.done_spec = Categorical(
            n=2,
            shape=torch.Size([1]),
            dtype=torch.bool,
            device=self.device,
        )

    def _obs_tensor(self, obs):
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _reset(self, tensordict=None, **kwargs):
        obs = self.env.reset()
        return TensorDict(
            {
                "observation": self._obs_tensor(obs),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        action = int(tensordict["action"].item())

        obs, reward, done, _ = self.env.step(action)

        done_tensor = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict(
            {
                "observation": self._obs_tensor(obs),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": done_tensor,
                "terminated": done_tensor.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed=None):
        if seed is not None and hasattr(self.env, 'seed'):
            self.env.seed(seed)
        return
