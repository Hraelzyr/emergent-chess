pieces = ['K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

#Kh5
#1 word = 1 move
#3 word ; 1 char = 1 word

# 200 -> 20
# 6 -> piece
#6 -> file
#8 -> rank
#20 -> t-SNE embedding -> 2

#Listener overwhelms speaker

#Kh5-Rh5+Rh1 = Kh1

complete = pieces + files + ranks
vocab_size = len(pieces)*len(files)*len(ranks)
# Tokenisation by word


def map_token_to_id(token: str) -> int:
    tok_hash = 0
    for char in token:
        tok_hash += complete.index(char)
    return tok_hash


def tokenise_moves(text: str):
    return map(map_token_to_id, text.split(''))
