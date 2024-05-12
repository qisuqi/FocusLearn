from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn

from AttnAM_PyTorch.model.encoder_decoder import EncoderRNNs
from AttnAM_PyTorch.model.get_attention import Attention
from AttnAM_PyTorch.model.select_features import SelectFeatures
from AttnAM_PyTorch.model.featurenn import FeatureNN


class AttnAM(nn.Module):
    def __init__(self, config):
        super(AttnAM, self).__init__()

        self.config = config

        self.rnn = EncoderRNNs(
            num_layers=self.config.encoder_num_layers,
            input_dim=self.config.encoder_input_dim,
            model_dim=self.config.encoder_hidden_unit,
            dropout=self.config.encoder_dropout
        )

        self.attention = Attention(
            model_dim=self.config.encoder_input_dim,
            num_heads=self.config.num_attn_heads
        )

        self.attention_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.config.encoder_input_dim, 1)
        )

        self.select_features = SelectFeatures(top=self.config.top_features, dim=-1)

        self.feature_nns = nn.ModuleList([
            FeatureNN(config=self.config,
                      name=f'FeatureNN_{i}',
                      input_shape=1)
            for i in range(self.config.top_features)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

        self.dropout = nn.Dropout(p=self.config.output_dropout)

    def calc_outputs(self, inputs, weights):
        """Returns the output computed by each feature net."""

        return [self.feature_nns[i](inputs[:, i], weights[:, i]) for i in range(self.config.top_features)]

    def forward(self, x):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
        """

        rnn = self.rnn(x)

        attention_weights = self.attention(rnn)

        attn_to_return = self.attention_out(attention_weights)

        new_weights, new_input = self.select_features(attention_weights, x)

        individual_outputs = self.calc_outputs(new_input, new_weights)

        conc_out = torch.cat(individual_outputs, dim=-1)

        dropped_out = self.dropout(conc_out)

        out = torch.sum(dropped_out, dim=-1).unsqueeze(dim=1)

        return attn_to_return, out + self._bias

    @torch.no_grad()
    def get_attention_maps(self, x):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        rnn = self.rnn(x)
        attention_maps = self.attention(rnn)

        return attention_maps

    @torch.no_grad()
    def get_top_feature_index(self, x):
        """Function for extracting top feature index.

        Input arguments same as the forward pass. """

        rnn = self.rnn(x)
        attention_weights = self.attention(rnn)
        top_ind = self.select_features(attention_weights, x, return_index=True)

        return top_ind

