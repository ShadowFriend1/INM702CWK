from game import Game
from numpy import random

# Run the script
if __name__ == '__main__':
    seed = random.randint(1000)
    game = Game(10, 10, [0, 0], [9, 9], seed)
    print(game.game_array)
    print(game.position)
    print(game.game_array[game.position[0], game.position[1]])
