import numpy as np

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class LearnAgent(Agent):
       
    def __init__(self, game, display=None):
        self.game = game
        self.display = display
        from game2048.load import model_0, model_64, model_128, model_256, model_512
        self.model_0 = model_0
        self.model_64 = model_64
        self.model_128 = model_128
        self.model_256 = model_256
        self.model_512 = model_512

    def predict(self):
        if self.game.score <= 32:
            result = self.model_0.predict(np.expand_dims(self.game.ohe_board, axis=0))
        elif 32< self.game.score <= 64:
            result = self.model_64.predict(np.expand_dims(self.game.ohe_board, axis=0))
        elif 64< self.game.score <= 128:
            result = self.model_128.predict(np.expand_dims(self.game.ohe_board, axis=0)) 
        elif 128< self.game.score <= 256:
            result = self.model_256.predict(np.expand_dims(self.game.ohe_board, axis=0))
        elif self.game.score >= 512:
            result = self.model_512.predict(np.expand_dims(self.game.ohe_board, axis=0))    
        return result

    def step(self):
        direction = self.predict().argmax()
        return direction
