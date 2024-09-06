"""
The Agent class. Note that the agent has just a single minimax function.
The minimax function is used in a generic Game class
"""

class Agent:
    def __init__(self, game, selfPiece, opponentPiece):
        self.game = game
        self.MIN = -float('inf')
        self.MAX = float('inf')
        self.winscore =  10000000
        self.losescore = -10000000
        self.selfPiece = selfPiece          #AI
        self.opponentPiece = opponentPiece  #HUMAN


    '''
    This is the function you must complete
    Inputs -
    1. self (Agent object reference)
    2. board (2d array)
    3. depth (float)
    4. maximizingPlayer (bool)
    5. alpha (float)
    6. beta (float)

    Outputs -
    1. score (float)
    2. next_move (type discussed below)

    Notes
    1. Need to create general implementation that considers all move types - tuple or float
    2. We use depth here to measure how far down we need to keep going, we stop at depth = 0
    3. Need to consider the different situations when game is over
    4. Prioritize winning in fewest moves
    5. next_move is the best possible move from valid_moves based on alpha-beta pruning. 
       Do not need to focus on its type, since it differs based on game, just find best move from valid_moves
    '''
    def minimax(self, board, depth, maximizingPlayer, alpha, beta):
        #breakpoint()
        cur_player = 0
        valid_moves = self.game.get_valid_moves(board)
        if depth == 0 or self.game.game_over(board): 
            if self.game.is_winner(board,self.selfPiece):
                return (None,self.winscore + depth)
            if self.game.is_winner(board,self.opponentPiece):
                    return (None,self.losescore - depth)
            if self.game.is_full(board):
                return (None,0)
            else:
                return (None,self.game.heuristic_value(board,self.selfPiece))
        
        
        if maximizingPlayer:
            next_move = None            
            best_max = self.MIN
            for i in range(len(valid_moves)):
                temp_board = self.game.play_move(board, valid_moves[i],2)
                cur_max = self.minimax(temp_board, (depth-1), False, alpha,beta)
                #breakpoint()
                if cur_max[1] > best_max:
                    next_move = valid_moves[i]
                best_max = max(cur_max[1], best_max)
                alpha = max(alpha, best_max)
                if beta <= alpha:
                    break
            return next_move, best_max
        else:
            next_move = None            
            best_min = self.MAX
            for i in range(len(valid_moves)):
                temp_board = self.game.play_move(board, valid_moves[i],1)
                cur_min = self.minimax(temp_board, (depth-1), True, alpha,beta)
                #breakpoint()
                if cur_min[1] < best_min:
                    next_move = valid_moves[i]
                best_min = min(cur_min[1], best_min)
                beta = min(beta, best_min)
                if beta <= alpha:
                    break
            return next_move, best_min