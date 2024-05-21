import unittest
from methods.trainer import *
from chess import Board


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_reward(self):
        correct_match, ended_cm = reward("e4", "e4", self.board)
        self.board.reset()
        correct_not_match, ended_cnm = reward("e4", "d4", self.board)
        self.board.reset()
        incorrect_move, ended_i = reward("e5", "e4", self.board)
        self.board.set_board_fen("8/8/8/8/8/K7/7Q/k7")
        mate, ended_m = reward("Qb2#", "Qb2#", self.board)

        assert ended_m, "Mate does not terminate the game"
        assert not ended_cm, "Correct move that matches terminates the game"
        assert not ended_cnm, "Correct move that does not match terminates the game"
        assert ended_i, "Incorrect move doesn't end the game"

        self.assertGreater(correct_match, correct_not_match)
        self.assertLess(incorrect_move, 0)
        self.assertGreater(correct_not_match, 0)
        self.assertGreater(mate, correct_match)

    def test_actor_loss_fn(self):
        advantage = torch.tensor([1, 2, 3])
        log_probs = torch.log(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]))
        sample = torch.tensor([0, 1, 2])
        loss = actor_loss_fn(advantage, log_probs, sample)
        self.assertEqual(loss.shape, (3,))

    def test_critic_loss_fn(self):
        value = torch.tensor([1, 2, 3], dtype=torch.float)
        target = torch.tensor([0, 1, 2])
        loss = critic_loss_fn(value, target)
        self.assertEqual((3,), loss.shape)


if __name__ == '__main__':
    unittest.main()
