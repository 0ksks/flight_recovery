from TorchRL.Trainer import Trainer
import torch as th

# graph
NODE_NUM = 20
# network
MLP_HIDDEN_ARCH = [64, 64]
LR = 3e-4
MAX_GRAD_NORM = 1.0
# device
IS_FORK = th.multiprocessing.get_start_method() == "fork"
DEVICE = th.device(0) if th.cuda.is_available() and not IS_FORK else th.device("cpu")
# experience
FRAMES_PER_BATCH = 1000
TOTAL_FRAMES = 50_000
# PPO
SUB_BATCH_SIZE = 64
NUM_EPOCHS = 10
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_EPS = 1e-4
Trainer().train(
    TOTAL_FRAMES, NUM_EPOCHS, FRAMES_PER_BATCH, SUB_BATCH_SIZE, DEVICE, MAX_GRAD_NORM
)
