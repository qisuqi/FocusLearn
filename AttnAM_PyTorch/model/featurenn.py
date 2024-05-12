import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .activation import ExU
from .activation import LinReLU

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeatureNN(nn.Module):
    """Neural Network model for each individual feature."""

    def __init__(self,
                 config,
                 name,
                 input_shape: int):

        """Initializes FeatureNN hyperparameters.

        Args:
        """
        super(FeatureNN, self).__init__()

        self.config = config
        self.name = name

        self._input_shape = input_shape

        self.act = LinReLU(in_features=input_shape,
                           out_features=self.config.feature_hidden_unit[0])

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=self.config.feature_dropout)

        self.fc1 = nn.Linear(in_features=self.config.feature_hidden_unit[0],
                             out_features=self.config.feature_hidden_unit[1])

        self.fc2 = nn.Linear(in_features=self.config.feature_hidden_unit[1],
                             out_features=1)

    def forward(self, inputs, weights):
        """Computes FeatureNN output with either evaluation or training
                mode."""

        inputs = inputs.unsqueeze(1)
        weights = weights.unsqueeze(1)

        outputs = self.relu(self.act(inputs, weights))
        outputs = self.dropout(outputs)
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)

        return outputs






