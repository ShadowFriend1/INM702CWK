from numpy import random, array

# A class representing the game board
class Game:

    # Initialises the game board with given values
    def __init__(self, width, depth, start_cell, end_cell, seed):
        # The width/depth defines the size of the board
        self.width = width
        self.depth = depth
        # size is the total size of the array
        self.size = width * depth
        # start and end cell are vectors for the start and end position
        self.start_cell = array(start_cell)
        self.end_cell = array(end_cell)
        # seed contains the random seed used by randint
        self.seed = seed
        random.seed(seed)
        # array contains the game board
        self.game_array = array(random.randint(9, size=(depth, width)))
        self.position = start_cell
        # contains current time elapsed
        self.time = 0
        # contains all cells moved through
        self.travelled = []

    # moves the current position and add the time of the new position to the current time value
    # only allows adjacent movement (no diagonals)
    def move(self, direction):
        if direction[0] + direction[1] == 1:
            # Adds the current location to the list of travelled locations
            self.travelled.append(self.position)
            # Moves the current position to the next
            self.position += direction
            # checks to see if the end tile is reached
            if self.position == self.end_cell:
                # if the end tile is reached the movement stops and the movement returns true to indicate that the
                # game is won
                print("End cell reached!")
                return True
            else:
                # Adds the time value of the new location to the current time value
                self.time += self.game_array[self.position[0], self.position[1]]
                return False
        elif direction[0] + direction[1] > 1:
            print("Tried to move too far")
            return False
        else:
            print("No Movement at all?")
            return False
