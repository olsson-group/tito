import warnings

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", ".*and is already saved during checkpointing.*")
warnings.filterwarnings("ignore", ".*There is a wandb run already in progress and.*")
warnings.filterwarnings("ignore", ".*does not have many workers which may be a bott.*")
