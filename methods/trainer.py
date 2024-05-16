import chess
import torch
import torch.nn.functional as F


def reward(sample, target, board):
    try:
        board.push_san(sample)
        if board.is_checkmate():
            return 10
        if sample == target:
            return 2
        return 1
    except chess.IllegalMoveError:
        return -10


def actor_loss_fn(advantage, log_probs, sample):
    return -advantage * torch.gather(log_probs, 1, sample.unsqueeze(1)).squeeze()


def critic_loss_fn(value, target):
    return F.huber_loss(value, target, reduction='none')
