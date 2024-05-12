import torch
import torch.nn as nn

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, model_dim, dropout):
        super(EncoderLSTM, self).__init__()
        self.model_dim = model_dim

        self.fc = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=model_dim,
                            batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        # self.out1 = nn.Linear(in_features=model_dim*2, out_features=model_dim)
        self.out = nn.Linear(in_features=model_dim, out_features=input_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])

        x = self.fc(x)  # Added a linear layer to allow grad flow
        lstm_output, _ = self.lstm(x)
        norm = self.norm(lstm_output)  # Also allows grad flow better
        #out1 = self.out1(norm)
        out = self.out(norm)
        dropout = self.dropout(out)

        return dropout


class EncoderGRU(nn.Module):
    def __init__(self, input_dim, model_dim, dropout):
        super(EncoderGRU, self).__init__()
        self.model_dim = model_dim

        self.fc = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=model_dim,
                          batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        self.out = nn.Linear(in_features=model_dim, out_features=input_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])

        x = self.fc(x)  # Added a linear layer to allow grad flow
        gru_output, _ = self.gru(x)
        norm = self.norm(gru_output)  # Also allows grad flow better
        out = self.out(norm)
        dropout = self.dropout(out)

        return dropout


class EncoderRNNs(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLSTM(**block_args) for _ in range(num_layers)])
        #self.layers = nn.ModuleList([EncoderGRU(**block_args) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderRNN(nn.Module):
    def __init__(self, lstm_out, lstm_hidden, output_size):
        super(DecoderRNN, self).__init__()
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(lstm_out, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, output_size)

    def forward(self, inputs, hidden, cell):
        output = F.relu(inputs)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.out(output)
        return output, hidden, cell


