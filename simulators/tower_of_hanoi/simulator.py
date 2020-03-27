from typing import Tuple, List

import numpy as np

from config import consts
from framework.action import Action
from framework.action_type import ActionType
from framework.simulator_interface import SimulatorInterface
from framework.state import State


class Simulator(SimulatorInterface):
    """
    Simulates the game of Tower Of Hanoi.
    """
    NUM_POLES = 3

    def __init__(self, num_floors: int, verbose: bool = False):
        """
        Initiates the environment simulator according to specified configuration.
        NOTICE: in order to start use the simulator, the method reset must be called manually.

        :param num_floors: positive integer. number of floors.
        :param verbose: whether or not to print actions description.
        """
        # verify input
        assert num_floors > 0, 'num floors must be positive number.'

        # store configuration
        self.__num_floors = num_floors

        # prepare members
        self.__state = None
        self.__start_pole = None
        self.__verbose = verbose

        assert Simulator.NUM_POLES == 3, 'currently only 3 poles are supported'
        self.__action_type = ActionType(
            action_name='transfer',
            values_type=consts.ActionTypeConsts.CATEGORICAL_VALUES_TYPE,
            action_values=[
                0,  # pole[0] -> pole[1]
                1,  # pole[0] -> pole[2]
                2,  # pole[1] -> pole[0]
                3,  # pole[1] -> pole[2]
                4,  # pole[2] -> pole[0]
                5   # pole[2] -> pole[1]
            ]
        )

    def get_state_feature_specs(self) -> List[int]:
        """
        Interface method implementation.
        """
        return [self.__num_floors + 1] * (self.__num_floors * Simulator.NUM_POLES) + [Simulator.NUM_POLES]

    def __log(self, msg):
        if self.__verbose:
            print(msg)

    def reset(self, verbose=None) -> State:
        """
        Interface method implementation.
        """
        # reset state
        self.__state = np.zeros(self.__num_floors * Simulator.NUM_POLES + 1, dtype=int)

        # select random starting pole
        self.__start_pole = np.random.randint(Simulator.NUM_POLES)
        self.__state[-1] = self.__start_pole

        # place the floors on the selected start pole
        start_idx = self.__start_pole * self.__num_floors
        self.__state[start_idx:start_idx + self.__num_floors] = np.arange(self.__num_floors) + 1

        # reset verbose mode
        if verbose is not None:
            self.__verbose = verbose

        # wrap with state and return
        return State(self.__state, False)

    def act(self, action: Action) -> Tuple[State, float]:
        """
        Interface method implementation.

        Rewards:
        * 1 for win + final
        * -1 for illegal move + final
        * 0 otherwise
        """
        # find source and target poles
        source_pole = action.action_value // 2

        order_idx = action.action_value - source_pole * 2
        target_pole = sorted({0, 1, 2} - {source_pole})[order_idx]

        self.__log('action: pole[{}] -> pole[{}]'.format(source_pole, target_pole))

        # execute action and return new state and reward
        return self.__transfer(source_pole, target_pole)

    def __transfer(self, source_pole_idx: int, target_pole_idx: int) -> Tuple[State, float]:
        """
        Rewards:
        * 1 for win + final
        * -1 for illegal move + final
        * 0 otherwise

        State is defined final in the following cases:
        * the floors are arranged correctly on a pole different from the starting pole.
        * illegal move performed.

        :param source_pole_idx: integer. the pole to take floor from.
        :param target_pole_idx: integer. the pole to transfer the floor into.
        :return:
        """
        # get floor to move
        source_pole_arr = self.__get_pole_arr(source_pole_idx)
        if (source_pole_arr == 0).all():
            self.__log('game over! source pole all zeros')
            return State(self.__state, is_final=True), -1

        source_floor_idx = np.where(source_pole_arr > 0)[0][0]
        floor_to_move = source_pole_arr[source_floor_idx]

        # get target pole arr and target slot idx
        target_pole_arr = self.__get_pole_arr(target_pole_idx)
        if (target_pole_arr != 0).any():
            highest_target_floor_idx = np.where(target_pole_arr > 0)[0][0]
            highest_target_floor = target_pole_arr[highest_target_floor_idx]

            if highest_target_floor < floor_to_move:
                self.__log('game over! highest top floor is smaller than floor to move: {} < {}'.format(
                    highest_target_floor, floor_to_move))
                return State(self.__state, is_final=True), -1

            target_slot_idx = highest_target_floor_idx - 1

        else:   # no floor exists on target pole
            target_slot_idx = -1

        # move floor
        target_pole_arr[target_slot_idx] = floor_to_move
        source_pole_arr[source_floor_idx] = 0

        # check whether or not this is a final state
        is_final_state = self.__is_final_state()
        if is_final_state:
            self.__log('Win!')

        # calculate reward
        reward = 1 if is_final_state else 0

        # return new state and reward
        return State(self.__state, is_final=is_final_state), reward

    def __is_final_state(self) -> bool:
        """
        Checks whether or not the game is over.
        :return:
        """
        # iterate relevant poles
        valid_pole = np.arange(self.__num_floors) + 1
        for pole_idx in list(set(range(Simulator.NUM_POLES)) - {self.__start_pole}):
            pole_arr = self.__get_pole_arr(pole_idx)
            if (pole_arr == valid_pole).all():
                return True

        return False

    def __get_pole_arr(self, pole_idx: int) -> np.ndarray:
        """
        Returns selected pole state slice.

        :param pole_idx: integer.
        :return:
        """
        start_idx = pole_idx * self.__num_floors
        return self.__state[start_idx:start_idx+self.__num_floors]

    def get_state(self) -> State:
        """
        Interface method implementation.
        """
        return State(self.__state, is_final=self.__is_final_state())

    def get_actions(self) -> ActionType:
        """
        Interface method implementation.
        """
        return self.__action_type

    def visualize(self):
        """
        Interface method implementation.
        """
        output_str = str()
        pole_str_width = 2 * self.__num_floors + 1

        for floor_idx in range(self.__num_floors):
            for pole_idx in range(Simulator.NUM_POLES):
                pole_arr = self.__get_pole_arr(pole_idx)
                pole_floor = pole_arr[floor_idx]

                pole_floor_str_arr = ['.'] * pole_str_width
                if pole_floor == 0:
                    pole_floor_str_arr[self.__num_floors] = '|'
                else:
                    for i in range(self.__num_floors - (pole_floor - 1), self.__num_floors + pole_floor):
                        pole_floor_str_arr[i] = '#'

                output_str += ''.join(pole_floor_str_arr)

            output_str += '\n'

        for pole_idx in range(Simulator.NUM_POLES):
            pole_txt_str_arr = ['-'] * pole_str_width
            pole_txt_str_arr[self.__num_floors] = str(pole_idx)

            if pole_idx == self.__start_pole:
                pole_txt_str_arr[self.__num_floors - 1] = '['
                pole_txt_str_arr[self.__num_floors + 1] = ']'

            output_str += ''.join(pole_txt_str_arr)

        print(output_str)
