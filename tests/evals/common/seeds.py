import os
import random
import time
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: Optional[int] = 1337):
    s = int(seed if seed is not None else time.time() * 1000) % (2**31 - 1)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    return s


def fix_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
