import numpy as np
import torch

def make_code_reproducible():
    np.random.seed(15)
    torch.manual_seed(15)
    torch.cuda.manual_seed(1)

def make_nvidia_faster_computation():
    torch.backends.cudnn.benchmark = True
