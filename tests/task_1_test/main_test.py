from src.task_1.main import my_find_shortest, dijkstra_shortest
from src.task_1.game import Game


def test_my_alg():
    game = Game(10, 10, [0, 0], [9, 9], 0)
    path, moves = my_find_shortest(game)
    assert path == [[0, 0], [1, 1], [2, 2], [2, 3], [3, 3], [4, 4], [4, 5], [5, 5], [5, 6], [6, 6], [7, 7], [8, 7],
                    [9, 7], [9, 8], [9, 9]]
    assert moves == [[1, 1], [1, 1], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 1], [1, 0], [1, 0],
                     [0, 1], [0, 1]]


def test_dijkstra():
    game = Game(10, 10, [0, 0], [9, 9], 0)
    shortest, tent_distance = dijkstra_shortest(game)
    assert shortest == 22
    assert (tent_distance == [[0, 0, 3, 6, 11, 13, 17, 12, 14, 17],
                              [6, 8, 8, 4, 10, 14, 14, 15, 10, 15],
                              [14, 10, 7, 4, 7, 12, 7, 9, 10, 18],
                              [10, 10, 7, 7, 11, 7, 8, 7, 11, 17],
                              [12, 9, 14, 9, 7, 7, 11, 12, 12, 17],
                              [17, 13, 10, 11, 15, 8, 8, 15, 15, 18],
                              [19, 12, 10, 13, 13, 12, 12, 14, 18, 19],
                              [15, 14, 14, 18, 16, 15, 19, 17, 19, 18],
                              [15, 19, 17, 14, 19, 15, 16, 17, 21, 20],
                              [15, 18, 16, 14, 21, 20, 15, 17, 24, 22]]).all()


if __name__ == '__main__':
    test_my_alg()
    test_dijkstra()
