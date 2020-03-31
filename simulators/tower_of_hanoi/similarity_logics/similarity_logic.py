
from typing import List, Tuple

import numpy as np

from framework.action import Action
from framework.state import State


"""
== 0 <-> 1 ==
action_values=[
                0,  # pole[1] -> pole[0]
                1,  # pole[1] -> pole[2]
                2,  # pole[0] -> pole[1]
                3,  # pole[0] -> pole[2]
                4,  # pole[2] -> pole[1]
                5   # pole[2] -> pole[0]
            ]
"""

"""
== 0 <-> 2 ==
action_values=[
                0,  # pole[2] -> pole[1]
                1,  # pole[2] -> pole[0]
                2,  # pole[1] -> pole[2]
                3,  # pole[1] -> pole[0]
                4,  # pole[0] -> pole[2]
                5   # pole[0] -> pole[1]
            ]
"""

"""
== 1 <-> 2 ==
action_values=[
                0,  # pole[0] -> pole[2]
                1,  # pole[0] -> pole[1]
                2,  # pole[2] -> pole[0]
                3,  # pole[2] -> pole[1]
                4,  # pole[1] -> pole[0]
                5   # pole[1] -> pole[2]
            ]
"""

"""
== ORG ==
action_values=[
                0,  # pole[0] -> pole[1]
                1,  # pole[0] -> pole[2]
                2,  # pole[1] -> pole[0]
                3,  # pole[1] -> pole[2]
                4,  # pole[2] -> pole[0]
                5   # pole[2] -> pole[1]
            ]
"""


ACTION_SWAP_MAP = {
    0: {    # pole[0]
        1: {    # swapped with pole[1]
            0: 2,   # action[0] is now action[2]
            1: 3,   # ...
            2: 0,
            3: 1,
            4: 5,
            5: 4
        },
        2: {  # swapped with pole[2]
            0: 5,
            1: 4,
            2: 3,
            3: 2,
            4: 1,
            5: 0
        }
    },
    1: {
        2: {
            0: 1,
            1: 0,
            2: 4,
            3: 5,
            4: 2,
            5: 3
        }
    },
}


def _swap_poles_data(s_t: State, pole_idx_a: int, pole_idx_b: int) -> None:
    """
    Swaps the data of the specified poles pair.

    :param s_t: State.
    :param pole_idx_a: int.
    :param pole_idx_b: int.
    :return:
    """
    # get the number of floors
    poles_data = s_t.features[:-1]
    num_floors = poles_data.max()

    # swap data
    tmp_pole_data = np.zeros(num_floors, dtype=poles_data.dtype)
    tmp_pole_data[:] = poles_data[num_floors * pole_idx_a: num_floors * (pole_idx_a + 1)]
    poles_data[num_floors * pole_idx_a: num_floors * (pole_idx_a + 1)] = \
        poles_data[num_floors * pole_idx_b: num_floors * (pole_idx_b + 1)]
    poles_data[num_floors * pole_idx_b: num_floors * (pole_idx_b + 1)] = tmp_pole_data[:]

    # handle the case the start pole is swapped
    start_pole_idx = s_t.features[-1]
    if start_pole_idx == pole_idx_a:
        s_t.features[-1] = pole_idx_b
    elif start_pole_idx == pole_idx_b:
        s_t.features[-1] = pole_idx_a


def get_similarity_group(s_t: State, a_t: Action) -> List[Tuple[State, Action, float]]:
    """
    Calculates a list of state-action pairs that are similar to the given pair, along with the similarity rate.
    Notice: the input pair isn't included in the output list.

    :param s_t: State.
    :param a_t: Action.
    :return:
    """
    # create initial list
    similarity_group = list()

    # add the cases from first order
    for pole_idx_a in range(2):
        for pole_idx_b in range(pole_idx_a + 1, 3):
            new_state = State(s_t.features.copy(), s_t.is_final)
            _swap_poles_data(new_state, pole_idx_a, pole_idx_b)

            new_action = Action(ACTION_SWAP_MAP[pole_idx_a][pole_idx_b][a_t.action_value], a_t.action_cls)

            similarity_group.append((new_state, new_action, 1.0))

    # return the similarity group
    return similarity_group
