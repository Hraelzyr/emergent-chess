import torch
import torch.nn as nn
# h, x -> y


class SpeakerAndListener(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SpeakerAndListener, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_word, hidden):
        output = self.embedding(input_word)
        word = 1
        words = None
        logprobs = None
        while word != 0:
            output, hidden = self.gru(output, hidden)
            if logprobs is None:
                logprobs = self.softmax(self.out(output))
            else:
                logprobs = torch.cat([logprobs, self.softmax(self.out(output))], 0)
            word = torch.multinomial(torch.exp(logprobs[-1]), 1)
            if words is None:
                words = word
            else:
                words = torch.cat([words, word])
        return words, hidden