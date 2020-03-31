
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
    Calculates the potential value of a state.

    :param s_t: State.
    :return:
    """
    # find the index of the pole with the largest floor
    pole_features = s_t.features[:-1]
    start_pole_idx = s_t.features[-1]
    largest_floor_size = pole_features.max()
    num_floors = largest_floor_size

    target_pole_idx = np.where(pole_features == largest_floor_size)[0][0] // num_floors
    if target_pole_idx == start_pole_idx:
        return 0

    # iteratively search for smaller floors
    total_correct = 1
    next_to_find = largest_floor_size - 1
    while next_to_find > 0:
        next_pole_idx = np.where(pole_features == next_to_find)[0][0] // num_floors
        if next_pole_idx != target_pole_idx:
            break

        total_correct += 1
        next_to_find -= 1

    return total_correct
