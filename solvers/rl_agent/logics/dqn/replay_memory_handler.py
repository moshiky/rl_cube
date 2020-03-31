
import numpy as np
from typing import List, Tuple

from framework.action import Action
from framework.state import State


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
        self.__states_t0[selected_idx, :] = self._state_to_array(s_t0)
        self.__states_t1[selected_idx, :] = self._state_to_array(s_t1)

        # store reward and action
        self.__rewards[selected_idx] = r
        self.__actions[selected_idx] = a.action_value

    def get_batch(self, batch_size: int) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Create random batch of samples.

        :param batch_size: integer. number of samples.
        :return: batches of:
            s_t0
            action
            reward
            s_t1
        """
        # select indexes
        batch_idxs = np.random.choice(range(self.__next_empty_idx), batch_size, replace=False)

        # return samples
        return self.__states_t0[batch_idxs], self.__actions[batch_idxs], self.__rewards[batch_idxs], self.__states_t1

    @staticmethod
    def _one_hot_encode(class_idx: int, num_classes: int) -> np.array:
        """
        Converts class index to the one-hot representation.

        :param class_idx: int
        :param num_classes: int
        :return: np.ndarray
        """
        vec = np.zeros(num_classes)
        vec[class_idx] = 1
        return vec

    def _state_to_array(self, state: State) -> np.array:
        """
        Converts a state to the representing np.array, according to self.__state_feature_specs configuration.

        :param state: State instance.
        :return: np.array.
        """
        # extract state features
        state_features = state.features
        if state_features.shape[0] == self.__total_num_features:
            return state_features

        # prepare output vector
        feature_vector = np.zeros(self.__total_num_features, dtype=np.float32)

        # fill feature_vector with feature values
        current_idx = 0
        for feature_idx in range(len(state_features)):
            num_feature_inputs = self.__state_feature_specs[feature_idx]
            feature_value = state_features[feature_idx]

            if num_feature_inputs == 1:
                feature_vector[current_idx] = feature_value
            else:
                feature_vector[current_idx:current_idx+num_feature_inputs] = \
                    self._one_hot_encode(feature_value, num_feature_inputs)

            current_idx += num_feature_inputs

        return feature_vector
