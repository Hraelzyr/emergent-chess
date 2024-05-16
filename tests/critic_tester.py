import unittest
import torch
import models.critic as critic


class CriticTester(unittest.TestCase):

    def setUp(self):
        self.critic = critic.Critic()

    def test_shape(self):
        x = torch.zeros(3, 192)
        output = self.critic(x)
        assert output.shape == (3, 1), f'Critic output shape is not correct: {output.shape}'


if __name__ == '__main__':
    unittest.main()
