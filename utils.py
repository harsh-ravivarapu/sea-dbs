import random
import os
import torch
import torch.nn as nn


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_weights_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
