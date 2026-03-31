from configs.config import FRAMES_PER_BATCH, TOTAL_FRAMES
from torchrl.collectors import Collector

def build_collector(env, policy, device):
    return Collector(
        env,
        policy,
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=TOTAL_FRAMES,
        device=device,
    )
