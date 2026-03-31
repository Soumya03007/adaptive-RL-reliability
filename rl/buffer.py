from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage

def build_buffer(capacity):
    return ReplayBuffer(
        storage=LazyTensorStorage(capacity),
    )
