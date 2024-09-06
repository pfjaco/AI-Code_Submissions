import logging
import math
import queue

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Vs = {}

        self.Qsa = {}
        self.Nsa = {}
        self.Ps = {}
        self.Ns = {}

        # this is the only member variable you'll have to use. It'll be used in select()
        self.visited = set() # all "state" positions we have seen so far

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def gameEnded(self, canonicalBoard):
      """
      This function determines if the current board position is the end of the game.

      Returns:
          gameReward: a value that returns 0 if the game hasn't ended, 1 if the player won, -1 if the player lost
      """

      gameReward = self.game.getGameEnded(canonicalBoard, 1)
      return gameReward

    def predict(self, state, canonicalBoard):
        """
        A wrapper to perform predictions and necessary policy masking for the code to work.
        The key idea is to call this function to return an initial policy vector and value from the neural network
        instead of needing a rollout

        Returns:
            r: the reward given by the neural network
        """
        self.Ps[state], val = self.nnet.predict(canonicalBoard)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        self.Ps[state] = self.Ps[state] * valids
        sum_Ps_s = np.sum(self.Ps[state])
        if sum_Ps_s > 0:
            self.Ps[state] /= sum_Ps_s
        else:
            log.error("All valid moves were masked, doing a workaround.")
            self.Ps[state] = self.Ps[state] + valids
            self.Ps[state] /= np.sum(self.Ps[state])

        self.Vs[state] = valids
        self.Ns[state] = 0
        return val

    def getValidActions(self, state):
        """
        Generates the valid actions from the avialable actions. Actions are given as a list of integers.
        The integers represent which spot in the board to place an Othello disc.
        To see a (x, y) representation of an action, you can do "x, y = (int(action/self.game.n), action%self.game.n)"

        Returns:
            validActions: all valid actions you can take in terms of a list of integers
        """

        validActions = []
        for action in range(self.game.getActionSize()):
            if self.Vs[state][action]:
                validActions.append(action)
        return validActions

    def nextState(self, canonicalBoard, action):
        """
        Gets the next board state given the action

        Returns:
            nextBoard: the next board state given the action
        """

        nextState, nextPlayer = self.game.getNextState(canonicalBoard, 1, action)
        nextState = self.game.getCanonicalForm(nextState, nextPlayer)
        return nextState

    def getConfidenceVal(self, state, action):
        if (state, action) not in self.Qsa:
            self.Qsa[(state, action)] = 0
            self.Nsa[(state, action)] = 0

        u = self.Qsa[(state, action)] + self.args.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state]) / (
                    1 + self.Nsa[(state, action)])

        return u

    def updateValues(self, r, state, action):
        self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + r) / (self.Nsa[(state, action)] + 1)
        self.Nsa[(state, action)] += 1
        self.Ns[state] += 1

    def expand(self, state):
        self.visited.add(state)

    def select(self, state, board):
        """Serves as the select phase of the MCTS algorithm, should return a tuple of (state, board, action, reward)"""
        r = self.gameEnded(board)
        # TODO: Handle cases where the reward (r) is not 0 or if
        # we have not visited the current state (in this case we should simulate rollouts)
        if r != 0:
          return None, None, None, -r
        elif state not in self.visited:
          self.expand(state)
          r = self.simulate(state,board)  
          return None, None , None, -r


        u = np.NINF
        bestAction = None
        for actionPrime in self.getValidActions(state):
            # TODO: Get the upper bound for a confidence value and adjust action accordingly
            # remember the goal of this function should be to return the state, board, action of the
            # highest value at this state given a set of actions
            up = self.getConfidenceVal(state,actionPrime)
            if up > u:
              u = up
              bestAction = actionPrime
        board = self.nextState(board, bestAction)
        state = self.game.stringRepresentation(board)
        return state, board, bestAction, 0

    def backpropagate(self, seq):
        """This function uses the seq that you build and maintain in self.search
        and iterates through it to propagate values into search tree
        """
        """Hint for first TODO - when r’ is not 0, this indicates that we have reached an end state (game
win or game loss) in the rollout. As a result, we can set the overall “accumulated” reward to
the current reward in the sequence since it is an end state. In the other case when r’ is 0, we
know that this must be an intermediate state, action pair that we have not discovered before.
11
When we come across such a pair, we must call self.updateValues with the current accumulated
reward which propagates rewards into the search tree. Afterwards, we must adjust the reward
to that of the other player (max → min, min → max).
      """
        r = 0
        while not seq.empty():
            # This method retrieves front of Lifo.Queue and pops, the structure for this tuple should be defined by you
            curr_state_tuple = seq.get()
            # TODO: Implement the cases for where R is 0 and when R is not 0
            # use self.updateValues when updating values in backprop step
            r_prime = curr_state_tuple[3]
            if r_prime != 0:
              # Do something
              # Potentially have to do update value until we reach the last visited node?
              #update values and set reward to 0?
              #self.updateValues(-(curr_state_tuple[3]),curr_state_tuple[0],curr_state_tuple[2])
              r = r_prime
            else:
              self.updateValues(r,curr_state_tuple[0],curr_state_tuple[2])
              r = -r
            #use our state' r' and action to update qsa nsa ns
        return

    def simulate(self, state, board):
        # TODO: This function should return a reward using self.predict
        r = self.predict(state, board)
        return r

    def search(self, initial_board):
        """
        This function performs MCTS. The action chosen at each node is one that
        has the maximum upper confidence bound.

        Once a leaf node is found, the neural network is called to return a
        reward r for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the reward of the current
        state. This is done since r is in [-1,1] and if r is the value of a
        state for the current player, then its value is -r for the other player.

        Returns:
            b0: the initial board state of the othello board
        """
        initial_state = self.game.stringRepresentation(initial_board)
        r = self.gameEnded(initial_board)
        for _ in range(self.args.numMCTSSims):
            state = initial_state
            board = initial_board
            sequence = queue.LifoQueue()
            r = 0
            while r == 0:
                # TODO: Use select to search through possible future states, remember that
                # select returns a tuple of (state', board', action, r')
                '''Hint for first TODO - Since we know that the selection phase of MCTS (and also self.select)
                    returns a tuple of (state’, board’, action, r’), think about how to leverage the returned values in
                    setting the new board, state, and r. In addition, at each selection step, you will need to update
                    the sequence properly with the newly generate r’.

                    Hint for second TODO - at this step, we have just obtained a board state that produces a reward
                    value that isn’t equal to 0 (meaning the rollout has either concluded in a loss or win). After we
                    complete a rollout in the MCTS algorithm, what should we do to update our values from all
                    the nodes we visited during the rollout ?'''
                temp = self.select(state,board)
                sequence.put((state,board,temp[2],temp[3]))
                state = temp[0]
                board = temp[1]
                r = temp[3]

            # TODO: After our selection process, think about what we need to do with our
            # built in sequence (hint: you should use one of the functions you implemented)
            self.backpropagate(sequence)

        return initial_board
