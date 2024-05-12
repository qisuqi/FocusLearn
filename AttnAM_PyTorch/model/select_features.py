import torch
import torch.nn as nn


class SelectFeatures(nn.Module):
    def __init__(self, top, dim):
        super(SelectFeatures, self).__init__()

        self.top = top
        self.dim = dim

    def forward(self, attn, inputs, return_index=False):
        """ Use topk function from PyTorch to select top k features based on attention weights. To make gradient can
        flow through this operation, torch.gather is used to mask the input with the selected index. """

        top_attn, top_ind = torch.topk(attn, k=self.top, dim=self.dim)

        top_inp = torch.gather(inputs, dim=self.dim, index=top_ind).float().requires_grad_(True)

        if return_index:
            return top_attn, top_ind
        else:
            return top_attn, top_inp



