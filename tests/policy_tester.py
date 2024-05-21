import unittest
import torch
from models.speaker_listener import SpeakerAndListener

# x n
# Speaker processes one word -> output until NULL
# Listener hears except NULL -> some moves


class SpeakingAndListeningTest(unittest.TestCase):
    def setUp(self):
        self.speaker = SpeakerAndListener(64, 192, 17)
        self.listener = SpeakerAndListener(17, 192, 64)

    def test_speaker(self):
        x = torch.zeros((1, 1), dtype=torch.long)  # 1 sentences of 1 word
        hidden = torch.zeros((1, 1, 192))  # 1 hidden state
        output, hidden = self.speaker(x, hidden)
        assert (output.shape[1] == 1), f'Speaker output shape is not correct: {output.shape}'

    def test_listener(self):
        x = torch.zeros((1, 1), dtype=torch.long)
        hidden = torch.zeros((1, 1, 192))
        output, hidden = self.listener(x, hidden)
        assert (output.shape[1] == 1), f'Listener output shape is not correct: {output.shape}'

    def test_sequence(self):
        x = torch.zeros((3, 1), dtype=torch.long)
        hidden_s = torch.zeros((1, 1, 192))  # 1 hidden state
        hidden_l = hidden_s.clone()
        output = None
        intermediate = None
        for word in x:
            sentence, hidden_s = self.speaker(word.unsqueeze(0), hidden_s)
            if intermediate is None:
                intermediate = sentence
            else:
                torch.cat([intermediate, sentence])

        for word_ in intermediate:
            output_, hidden_l = self.listener(word_.unsqueeze(0), hidden_l)
            if output is None:
                output = output_
            else:
                torch.cat([output, output_])

        assert output.shape[1] == 1, f"Malformed output: {output.shape}"


if __name__ == '__main__':
    unittest.main()
