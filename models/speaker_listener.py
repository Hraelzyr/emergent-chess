import torch.nn as nn


class Speaker(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Speaker, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_sentence, hidden):
        embedded = self.embedding(input_sentence)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden


class Listener(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Listener, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_sentence, hidden):
        output = self.embedding(input_sentence)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output))
        return output, hidden
