from configs.config import CLIP_EPSILON, CRITIC_COEFF, ENTROPY_COEFF, GAMMA, LAMBDA
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

def build_ppo(policy_module, value_module):
    advantage = GAE(
        gamma=GAMMA,
        lmbda=LAMBDA,
        value_network=value_module,
        average_gae=False,
    )

    loss = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=CLIP_EPSILON,
        entropy_coeff=ENTROPY_COEFF,
        critic_coeff=CRITIC_COEFF,
    )

    return advantage, loss
