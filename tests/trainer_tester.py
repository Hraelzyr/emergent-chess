import unittest
from methods.trainer import *
from chess import Board


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_reward(self):
        correct_match = reward("e4", "e4", self.board)
        self.board.reset()
        correct_not_match = reward("e4", "d4", self.board)
        self.board.reset()
        incorrect_move = reward("e5", "e4", self.board)
        self.assertGreater(correct_match, correct_not_match)
        self.assertLess(incorrect_move, 0)
        self.assertGreater(correct_not_match, 0)

    def test_actor_loss_fn(self):
        advantage = torch.tensor([1, 2, 3])
        log_probs = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        sample = torch.tensor([0, 1, 2])
        loss = actor_loss_fn(advantage, log_probs, sample)
        self.assertEqual(loss.shape, (3,))

    def test_critic_loss_fn(self):
        value = torch.tensor([1, 2, 3], dtype=torch.float)
        target = torch.tensor([0, 1, 2])
        loss = critic_loss_fn(value, target)
        self.assertEqual(loss.shape, (3,))


if __name__ == '__main__':
    unittest.main()
