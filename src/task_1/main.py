import matplotlib.pyplot as plt
import numpy as np

from src.task_1.game import Game


# My algorithm to find the shortest path
def my_find_shortest(game):
    # horizontal distance between the start and end cell
    vert_diff = game.end_cell[1] - game.start_cell[1]
    # vertical distance between the start and end cell
    hori_diff = game.end_cell[0] - game.start_cell[0]

    # Starts the path of traversal at the starting cell
    path = [game.start_cell]

    # starts the time value at 0
    time = 0

    # The set of moves the algorithm makes
    moves = []

    # Runs until the path finds the end cell
    while path[-1] != game.end_cell:
        # The minimum move required to continue in a short path to the end cell
        min_move = [0, 0]
        # Checks to see if there are more moves left than are required to reach the end cell with no more moves
        # than necessary, if there are not the minimum move is set such that the path will always move towards the end
        if vert_diff <= abs(game.end_cell[1] - path[-1][1]):
            if vert_diff > 0:
                min_move[1] = 1
            elif vert_diff != 0:
                min_move[1] = -1
        if hori_diff <= abs(game.end_cell[0] - path[-1][0]):
            if hori_diff > 0:
                min_move[0] = 1
            elif hori_diff != 0:
                min_move[0] = -1
        # The maximum move that can be made, obeying the game rules (it wil be 0 if the current path is in line with
        # the end)
        max_move = [0, 0]
        if vert_diff > 0:
            max_move[1] = 1
        elif vert_diff != 0:
            max_move[1] = -1
        if hori_diff > 0:
            max_move[0] = 1
        elif hori_diff != 0:
            max_move[0] = -1
        # The range of movements that can be performed in each axis
        vert_range = [min_move[1], max_move[1]]
        hori_range = [min_move[0], max_move[0]]
        # The lowest time of an available move
        lowest_time = 10
        # The current lowest time move (defaults to the most movement towards the end)
        best_move = max_move
        # Repeats for all available moves
        for h in vert_range:
            for v in hori_range:
                # Checks to see if the checked move is within the array
                if 0 <= v + path[-1][0] <= game.depth - 1 and 0 <= h + path[-1][1] <= game.width - 1:
                    # Finds the time value of the proposed move
                    g = game.game_array[v + path[-1][0], h + path[-1][1]]
                    # If it is the lowest time value checked so far it makes it the best move
                    if g < lowest_time and [v, h] != [0, 0]:
                        lowest_time = g
                        best_move = [v, h]
                    # If the checked move is to the end cell it sets that to the best move and stops checking
                    elif [v + path[-1][0], h + path[-1][1]] == game.end_cell:
                        best_move = [v, h]
                        break
        time += lowest_time
        # Adds the best move to both the path and list of moves
        path.append([path[-1][0] + best_move[0], path[-1][1] + best_move[1]])
        moves.append(best_move)
    return path, moves, time


def dijkstra_shortest(game):
    # Array storing whether nodes have been visited as the current node
    visited = np.full((game.width, game.depth), False)
    # Array storing tentative distance values
    tent_distance = np.zeros(game.game_array.shape)
    path = np.empty(game.game_array.shape, dtype=object)
    for i in range(0, len(path)):
        for j in range(0, len(path[0])):
            path[i][j] = []
    # Sets tentative distance values to -1 (representing infinity), sets start node to 0
    for i in range(0, len(tent_distance)):
        for j in range(0, len(tent_distance[0])):
            if [i, j] == game.start_cell:
                tent_distance[i, j] = 0
            else:
                tent_distance[i, j] = -1
    current_node = game.start_cell
    # Repeats until all nodes have been visited
    while False in visited:
        # visits all adjacent nodes
        for h in (-1, 0, 1):
            for v in (-1, 0, 1):
                # The node currently being visited
                visit = [current_node[0] + h, current_node[1] + v]
                # if the node being visited exists, is not recorded as having been visited as a current node and has a
                # tentative distance value smaller than the new distance, record the new distance in that node
                if [h, v] != [0, 0] and (0 <= visit[0] <= game.width - 1) and (0 <= visit[1] <= game.depth - 1) and \
                        not visited[visit[0], visit[1]]:
                    dist = tent_distance[current_node[0], current_node[1]] + game.game_array[visit[0], visit[1]]
                    # Checks to see if the new distance value is smaller that the already stored value
                    if tent_distance[visit[0], visit[1]] == -1 or \
                            dist < tent_distance[visit[0], visit[1]]:
                        tent_distance[visit[0], visit[1]] = dist
                        path_temp = path[current_node[0], current_node[1]]
                        path_temp.append([visit[0], visit[1]])
                        path[visit[0], visit[1]] = path_temp
        # Set the current node as visited
        visited[current_node[0], current_node[1]] = True
        # Set the best distance to infinity
        best_distance = -1
        # Loop through all nodes and find the unvisited best distance to be the next current node, then set that node
        # to the current node
        for h in range(0, len(tent_distance)):
            for v in range(0, len(tent_distance[0])):
                if tent_distance[h, v] != -1 and (tent_distance[h, v] < best_distance or best_distance == -1) and \
                        not visited[h, v]:
                    best_distance = tent_distance[h, v]
                    current_node = [h, v]
    return tent_distance[game.end_cell[0], game.end_cell[1]], tent_distance, path[game.end_cell[0], game.end_cell[1]]

def distribution_graphs():
    my_shortest_paths_uniform = []
    my_shortest_paths_normal = []
    dijkstra_shortest_paths_uniform = []
    dijkstra_shortest_paths_normal = []
    # Produces graph of how algorithms perform on different distributions, the array size increases every 10 iteration
    # and the average path length is recorded against the array size, the start cell is always the top left corner
    # and the end cell is always the bottom right
    my_time_uni_list = []
    my_time_norm_list = []
    d_time_uni_list = []
    d_time_norm_list = []
    size_list = []
    for i in range(0, 200):
        # Randomises the array in a consistent way
        seed = i
        game = Game(10 + round(i / 10), 10 + round(i / 10), [0, 0], [9+round(i/10), 9 + round(i/10)], seed, "uniform")
        game_normal = Game(10 + round(i / 10), 10 + round(i / 10), [0, 0], [9+round(i/10), 9+round(i/10)], seed,
                           "normal", 2)
        _, _, my_time_uni = my_find_shortest(game)
        d_short_uni, _, _ = dijkstra_shortest(game)
        _, _, my_time_norm = my_find_shortest(game_normal)
        d_short_norm, _, _ = dijkstra_shortest(game_normal)
        my_time_uni_list.append(my_time_uni)
        my_time_norm_list.append(my_time_norm)
        d_time_uni_list.append(d_short_uni)
        d_time_norm_list.append(d_short_norm)
        if i % 10 == 0:
            my_shortest_paths_uniform.append(sum(my_time_uni_list) / len(my_time_uni_list))
            my_shortest_paths_normal.append(sum(my_time_norm_list) / len(my_time_norm_list))
            dijkstra_shortest_paths_uniform.append(sum(d_time_uni_list) / len(d_time_uni_list))
            dijkstra_shortest_paths_normal.append(sum(d_time_norm_list) / len(d_time_norm_list))
            size_list.append(game.size)
        if (i + 1) % 10 == 0:
            print(i + 1, " iterations finished")
    plt.plot(size_list, my_shortest_paths_uniform, label="Uniform Distribution My Algorithm")
    plt.plot(size_list, my_shortest_paths_normal, label="Normal Distribution My Algorithm")
    plt.plot(size_list, dijkstra_shortest_paths_uniform, label="Uniform Distribution Dijkstra's Algorithm")
    plt.plot(size_list, dijkstra_shortest_paths_normal, label="Normal Distribution Dijkstra's Algorithm")
    plt.legend()
    plt.xlabel("Array Size")
    plt.ylabel("Mean Shortest Found Path Length")
    plt.show()

# Produces a graph of how the algorithms do on the normal distribution with different standard deviations (0-4)
def plot_deviation():
    dev_list = []
    my_f_list = []
    d_f_list = []
    for j in np.arange(0, 4, 0.1):
        my_time_list = []
        d_time_list = []
        for i in range(0, 10):
            seed = i
            game = Game(10, 10, [0, 0], [9, 9], seed, "normal", j)
            _, _, my_time = my_find_shortest(game)
            d_time, _, _ = dijkstra_shortest(game)
            my_time_list.append(my_time)
            d_time_list.append(d_time)
        my_f_list.append(sum(my_time_list) / len(my_time_list))
        d_f_list.append(sum(d_time_list) / len(d_time_list))
        dev_list.append(j)
    plt.plot(dev_list, my_f_list, label="My Algorithm")
    plt.plot(dev_list, d_f_list, label="Dijkstra's Algorithm")
    plt.legend()
    plt.xlabel("Standard Deviation")
    plt.ylabel("Mean Shortest Found Path Length")
    plt.show()


# Run the script
if __name__ == '__main__':
    plot_deviation()
