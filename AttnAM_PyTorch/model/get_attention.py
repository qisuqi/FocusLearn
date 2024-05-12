import torch
import torch.nn as nn

#from .multihead_attention import MultiheadAttention
from torch.nn import MultiheadAttention
from torch.nn import AdaptiveAvgPool1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(Attention, self).__init__()
        self.model_dim = model_dim

        self.self_attn = MultiheadAttention(embed_dim=model_dim,
                                            num_heads=num_heads)
        self.feature_pooling = AdaptiveAvgPool1d(1)

    def forward(self, x):
        """ A multi-head attention is used to with each head attend each feature independently. Computed attention
         weights are then combined across all features using feature-wise attention pooling so that order of input
         features can be ignored. """
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, num_features] -> [seq_len, batch_size, num_features]
        attention_output, _ = self.self_attn(x, x, x)  # Apply self attention across seq_len dimension
        attention_output = attention_output.permute(1, 2, 0)  # Convert back to [batch_size, num_features, seq_len]

        feature_attention_weights = self.feature_pooling(attention_output)  # Apply feature-wise pooling
        feature_attention_weights = feature_attention_weights.squeeze()  # Remove extra dimension
        # apply softmax to obtain normalised attention weights
        feature_attention_weights = nn.functional.softmax(feature_attention_weights, dim=-1)

        if len(feature_attention_weights.shape) == 1:
            feature_attention_weights = feature_attention_weights.unsqueeze(0)

        return feature_attention_weights



