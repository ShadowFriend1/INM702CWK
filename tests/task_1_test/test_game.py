from src.task_1.game import Game


def test_game():
    game = Game(10, 10, [0, 0], [9, 9], 0)
    assert (game.game_array == [[5, 0, 3, 3, 7, 3, 5, 2, 4, 7],
                                [6, 8, 8, 1, 6, 7, 7, 8, 1, 5],
                                [8, 4, 3, 0, 3, 5, 0, 2, 3, 8],
                                [1, 3, 3, 3, 7, 0, 1, 0, 4, 7],
                                [3, 2, 7, 2, 0, 0, 4, 5, 5, 6],
                                [8, 4, 1, 4, 8, 1, 1, 7, 3, 6],
                                [7, 2, 0, 3, 5, 4, 4, 6, 4, 4],
                                [3, 4, 4, 8, 4, 3, 7, 5, 5, 0],
                                [1, 5, 3, 0, 5, 0, 1, 2, 4, 2],
                                [0, 3, 2, 0, 7, 5, 0, 2, 7, 2]]).all()
    ended, invalid = game.move([0, 0])
    assert invalid
    ended, invalid = game.move([-1, 0])
    assert invalid
    ended, invalid = game.move([0, -1])
    assert invalid
    ended, invalid = game.move([-1, -1])
    assert invalid
    ended, invalid = game.move([2, 2])
    assert invalid
    ended, invalid = game.move([2, 0])
    assert invalid
    ended, invalid = game.move([0, 2])
    assert invalid
    ended, invalid = game.move([-2, -2])
    assert invalid
    ended, invalid = game.move([1, 1])
    assert not invalid
    assert game.time == 8
    game.position = [9, 8]
    ended, invalid = game.move([0, 1])
    assert ended


if __name__ == '__main__':
    test_game()
