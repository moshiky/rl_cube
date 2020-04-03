
import numpy as np
from typing import List, Tuple

from framework.action import Action
from framework.state import State
from solvers.rl_agent.logics.dqn import state_utils


class ReplayMemoryHandler:
    """
    Handles replay memory for fast access.
    """
    def __init__(self, max_size: int, state_feature_specs: List[int]):
        """
        Init memory handler instance.

        :param max_size: integer. max number of records to store.
        :param state_feature_specs: list of ints. specs of the state features.
        """
        # store configuration
        self.__max_size = max_size
        self.__state_feature_specs = state_feature_specs
        self.__total_num_features = sum(self.__state_feature_specs)
        self.__next_empty_idx = 0

        # init memory arrays
        self.__states_t0 = np.zeros([self.__max_size, self.__total_num_features], dtype=np.float32)
        self.__actions = np.zeros(self.__max_size, dtype=int)
        self.__rewards = np.zeros(self.__max_size, dtype=np.float32)
        self.__states_t1 = self.__states_t0.copy()
        self.__states_t1_is_final = np.zeros(self.__max_size, dtype=int)

    def add_sample(self, s_t0: State, a: Action, r: float, s_t1: State) -> None:
        """
        Store experience sample.

        :param s_t0: State. initial state.
        :param a: Action. selected action.
        :param r: float. experienced reward signal.
        :param s_t1: State. target state.
        :return:
        """
        # select record index to store
        if self.__next_empty_idx < self.__max_size:
            selected_idx = self.__next_empty_idx
            self.__next_empty_idx += 1
        else:
            selected_idx = np.random.randint(self.__max_size)

        # convert states according to specified state feature specs
        self.__states_t0[selected_idx, :] = state_utils.state_to_array(s_t0, self.__state_feature_specs)
        self.__states_t1[selected_idx, :] = state_utils.state_to_array(s_t1, self.__state_feature_specs)
        self.__states_t1_is_final[selected_idx] = s_t1.is_final

        # store reward and action
        self.__rewards[selected_idx] = r
        self.__actions[selected_idx] = a.action_value

    def get_batch(self, batch_size: int) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """
        Create random batch of samples.

        :param batch_size: integer. number of samples.
        :return: batches of:
            s_t0
            action
            reward
            s_t1
            s_t1_is_final
        """
        # select indexes
        batch_idxs = np.random.choice(range(self.__next_empty_idx), batch_size, replace=False)

        # return samples
        return \
            self.__states_t0[batch_idxs], \
            self.__actions[batch_idxs], \
            self.__rewards[batch_idxs], \
            self.__states_t1[batch_idxs], \
            self.__states_t1_is_final[batch_idxs]

    def is_batch_ready(self, batch_size: int) -> bool:
        """
        Check whether or not there are enough samples for batch.

        :param batch_size: integer.
        :return: boolean.
        """
        return self.__next_empty_idx >= batch_size
