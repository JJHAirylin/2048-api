import numpy as np
from game2048.expectimax import board_to_move
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent, LearnAgent
from game2048.displays import Display
import sys
import csv, os

GAME_SIZE = 4
SCORE_TO_WIN = 1024
record = 64
iter_num = 1000
filename = 'TrainForAll.csv'


sys.path.append("./")
if os.path.exists(filename):
    new = False
else: 
    new = True

# ------------------------------------------------------
# save each board and its direction to a dict
# -------------------------------------------------------
with open(filename, "a", newline='') as f:
    writer = csv.writer(f)
    if new:
        writer.writerow(["R1C1","R1C2","R1C3","R1C4","R2C1","R2C2","R2C3","R2C4","R3C1","R3C2","R3C3","R3C4","R4C1","R4C2","R4C3","R4C4","direction"])

    i = 0
    while i < iter_num:
        game = Game(GAME_SIZE, SCORE_TO_WIN, random=False)
        agent_guide = ExpectiMaxAgent(game, Display())
        agent_learn = LearnAgent(game, Display())
        board = game.board
        print('Iter idx:', i)
        while(game.end == 0):
            board = game.board
            direction_learn = agent_learn.step()
            # direction_guide = agent_guide.step()
            if game.score >= record:
                board = board.flatten().tolist()
                direction_guide = agent_guide.step()
                data = np.int32(np.append(board, direction_guide))
                writer.writerow(data)   
            game.move(direction_learn)
            

        i = i + 1
    # f.close()