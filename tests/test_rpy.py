
"""
Test basic movements- Roll, Pitch and Yaw.
"""
import copy

import numpy as np

from cube_simulator import utils, cube_specs
from cube_simulator.rubiks_cube import RubiksCube


def test_yaw():
    """
    Test yaw movement.
    :return:
    """
    # create initial cube
    edge_size = 3
    test_cube = RubiksCube(edge_size=edge_size)
    original_cube = copy.deepcopy(test_cube)

    # do yaw four times to back to original state
    # do it on each row, and in both directions
    for direction in [cube_specs.direction_idx.CCW, cube_specs.direction_idx.CW]:
        for row_idx in range(edge_size):

            # do four moves
            for move_idx in range(4):
                test_cube.yaw(row_idx=row_idx, direction=direction)

            # validate the state is back to original
            assert (test_cube.get_cube() == original_cube.get_cube()).all(), \
                'Wrong move! {}'.format([direction, row_idx])


def test_roll():
    """
    Test roll movement.
    :return:
    """
    # create initial cube
    edge_size = 3
    test_cube = RubiksCube(edge_size=edge_size)
    original_cube = copy.deepcopy(test_cube)

    # do yaw four times to back to original state
    # do it on each row, and in both directions
    for direction in [cube_specs.direction_idx.CCW, cube_specs.direction_idx.CW]:
        for row_idx in range(edge_size):

            # do four moves
            for move_idx in range(4):
                test_cube.roll(row_idx=row_idx, direction=direction)

            # validate the state is back to original
            assert (test_cube.get_cube() == original_cube.get_cube()).all(), \
                'Wrong move! {}'.format([direction, row_idx])


def test_pitch():
    """
    Test pitch movement.
    :return:
    """
    # create initial cube
    edge_size = 3
    test_cube = RubiksCube(edge_size=edge_size)
    original_cube = copy.deepcopy(test_cube)

    # do yaw four times to back to original state
    # do it on each row, and in both directions
    for direction in [cube_specs.direction_idx.UP, cube_specs.direction_idx.DOWN]:
        for column_idx in range(edge_size):

            # do four moves
            for move_idx in range(4):
                test_cube.pitch(column_idx=column_idx, direction=direction)

            # validate the state is back to original
            assert (test_cube.get_cube() == original_cube.get_cube()).all(), \
                'Wrong move! {}'.format([direction, column_idx])


def test_all():
    """
    Test combined movement.
    :return:
    """
    # create initial cube
    edge_size = 3
    test_cube = RubiksCube(edge_size=edge_size)
    original_cube = copy.deepcopy(test_cube)

    # define scrambling moves order
    direction_map = {
        test_cube.roll: [cube_specs.direction_idx.CW, cube_specs.direction_idx.CCW],
        test_cube.pitch: [cube_specs.direction_idx.UP, cube_specs.direction_idx.DOWN],
        test_cube.yaw: [cube_specs.direction_idx.CW, cube_specs.direction_idx.CCW],
    }

    selected_moves = np.random.choice(list(direction_map.keys()), size=100)
    scrambling_moves = [[
        move_func,
        np.random.randint(edge_size),
        np.random.choice(direction_map[move_func])
    ] for move_func in selected_moves]

    # execute scrambling moves
    for move_func, idx, move_direction in scrambling_moves:
        move_func(idx, move_direction)

    # execute reversed scrambling moves in reversed order
    for move_func, idx, move_direction in scrambling_moves[::-1]:
        move_func(idx, utils.swap_direction(move_direction))

    # validate the state is back to original
    assert (test_cube.get_cube() == original_cube.get_cube()).all(), \
        'Scrambling not worked correctly!'
