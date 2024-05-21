import chess
import torch
import torch.nn.functional as F
# Concept space of speaker and listener
# Structure of words


# Listener sees board and ignores speaker

# 1. Tablebase -> sample a winning seq -> speaker -> listener -> reward -> train
# 2. stockfish -> evaluate -> q-value
# pi(lis) = pi(stock)

#prfix -> speaker -> listener -> speaker -> listener -> suffix : evaluate suffix

def reward(sample, target, board):
    try:
        board.push_san(sample)
        if board.is_checkmate():
            return 10, True
        if sample == target:
            return 2, False
        return 1, False
    except chess.IllegalMoveError:
        return -10, True

# H = -log pi(detached)*log(pi)


def actor_loss_fn(advantage, log_probs, sample):
    lp = torch.gather(log_probs, 1, sample.unsqueeze(1)).squeeze().detach()
    return -(advantage) * torch.gather(log_probs, 1, sample.unsqueeze(1)).squeeze() #+H


def critic_loss_fn(value, target):
    return F.huber_loss(value, target, reduction='none')
