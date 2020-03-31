
import numpy as np

from framework.action import Action
from framework.state import State


def get_shaping_signal(s_t0: State, a_t: Action, s_t1: State) -> float:
    """
    Calculates the shaping signal for Tower Of Hanoi game.
    Based on PBRS - potential based reward shaping.

    The logic is: the potential of each state is the number of floors in the correct final position.

    :param s_t0: State. initial state.
    :param a_t: Action. selected action.
    :param s_t1: State. target state.
    :return: float- the shaping signal.
    """
    return _phi(s_t1) - _phi(s_t0)


def _phi(s_t: State) -> float:
    """
    The potential of a state is defined by the height of the tallest tower, not including the one with the widest base.
    In case the one with the widest base is on a pole different from the initial one- it is included (this is in fact
    the final state)

    :param s_t: State.
    :return:
    """
    # find the index of the pole with the largest floor
    pole_features = s_t.features[:-1]
    start_pole_idx = s_t.features[-1]
    largest_floor_size = pole_features.max()
    num_floors = largest_floor_size

    # get largest pole idx
    largest_pole_idx = np.where(pole_features == largest_floor_size)[0][0] // num_floors

    # calculate the height of the tower on each pole
    num_poles = pole_features.shape[0] // num_floors
    pole_tower_height = (pole_features.reshape([num_poles, num_floors]) > 0).sum(axis=1)
    height_dict = dict(zip(range(num_poles), pole_tower_height))

    # sort heights to get the potential value
    sorted_map = sorted(height_dict.items(), reverse=True, key=lambda x: x[1])
    if sorted_map[0][0] == start_pole_idx and largest_pole_idx == start_pole_idx:
        # the highest is on the start pole, and is with the widest base- so ignore it and return the size of the next
        # highest tower
        return sorted_map[1][1]
    else:
        # return the height of the highest tower
        return sorted_map[0][1]
