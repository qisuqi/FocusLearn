from typing import List

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: make get_num_units to work with selected data. And subsequent functions e.g.,
# get new unique features etc to work.

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def get_num_units(config,
                  features: torch.Tensor) -> List:
    features = features.cpu()
    num_unique_vals = [len(np.unique(features[:, i])) for i in range(features.shape[1])]

    num_units = [min(config.num_basis_functions, i * config.units_multiplier) for i in num_unique_vals]

    return num_units


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

