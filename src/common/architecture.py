import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        #batch_size = x.size()[0]
        #h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        #c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        #output, (hn, cn) = self.lstm(x, (h0, c0))

        output, (hn, cn) = self.lstm(x)
        logits = self.output_layer(output)
        return logits, (hn, cn)

    def forward_from_state(self, x, state):
        output, (hn, cn) = self.lstm(x, state)
        logits = self.output_layer(output)
        return logits, (hn, cn)
