from typing import List

import numpy as np

from framework.state import State


def one_hot_encode(class_idx: int, num_classes: int) -> np.array:
    """
    Converts class index to the one-hot representation.

    :param class_idx: int
    :param num_classes: int
    :return: np.ndarray
    """
    vec = np.zeros(num_classes)
    vec[class_idx] = 1
    return vec


def state_to_array(state: State, state_feature_specs: List[int]) -> np.array:
    """
    Converts a state to the representing np.array, according to self.__state_feature_specs configuration.

    :param state: State instance.
    :param state_feature_specs: list of ints.
    :return: np.array.
    """
    # extract state features
    state_features = state.features
    if state_features.shape[0] == sum(state_feature_specs):
        return state_features

    # prepare output vector
    feature_vector = np.zeros(sum(state_feature_specs), dtype=np.float32)

    # fill feature_vector with feature values
    current_idx = 0
    for feature_idx in range(len(state_features)):
        num_feature_inputs = state_feature_specs[feature_idx]
        feature_value = state_features[feature_idx]

        if num_feature_inputs == 1:
            feature_vector[current_idx] = feature_value
        else:
            feature_vector[current_idx:current_idx+num_feature_inputs] = \
                    one_hot_encode(feature_value, num_feature_inputs)

            current_idx += num_feature_inputs

        return feature_vector
