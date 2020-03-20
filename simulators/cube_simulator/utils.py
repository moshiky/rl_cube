
from simulators.cube_simulator.cube_specs import direction_idx


def _swap_pair(original_direction, direction_pair):
    """

    :param original_direction:
    :param direction_pair:
    :return:
    """
    if original_direction == direction_pair[0]:
        return direction_pair[1]
    return direction_pair[0]


def swap_direction(original_direction):
    """
    Returns the opposite direction.

    :param original_direction:
    :return:
    """
    if original_direction in [direction_idx.CW, direction_idx.CCW]:
        return _swap_pair(original_direction, [direction_idx.CW, direction_idx.CCW])

    if original_direction in [direction_idx.UP, direction_idx.DOWN]:
        return _swap_pair(original_direction, [direction_idx.UP, direction_idx.DOWN])
