import unittest
import torch

from models.speaker_listener import Speaker, Listener


class SpeakingAndListeningTest(unittest.TestCase):
    def setUp(self):
        self.speaker = Speaker(64, 192, 17)
        self.listener = Listener(17, 192, 64)

    def test_speaker(self):
        x = torch.zeros((10, 1), dtype=torch.long)  # 10 sentences of 1 word each
        hidden = torch.zeros((1, 10, 192))  # 10 hidden states
        output, hidden = self.speaker(x, hidden)
        assert (output.shape == (10, 1, 17)), f'Speaker output shape is not correct: {output.shape}'

    def test_listener(self):
        x = torch.zeros((10, 1), dtype=torch.long)
        hidden = torch.zeros((1, 10, 192))
        output, hidden = self.listener(x, hidden)
        assert (output.shape == (10, 1, 64)), f'Listener output shape is not correct: {output.shape}'


if __name__ == '__main__':
    unittest.main()
