from typing import List
import torch
import torch.nn as nn
import numpy as np

from framework.state import State


class QNet(nn.Module):
    """
    Implements q net module.
    """
    def __init__(self, state_feature_specs, num_actions, layers, dropout_rate):
        """
        Initiate the module.

        :param state_feature_specs: list of positive non-zero integers. [k0, k1, ..., kn]
            if ki = 1 - numeric feature
            if ki >= 2 - categorical feature with k classes
            otherwise- error!
        :param num_actions: integer. number of actions.
        :param layers: layers configuration. list of integers.
        :param dropout_rate: float.
        """
        # call base c'tor
        super().__init__()

        # store configuration
        self.__state_feature_specs = state_feature_specs
        self.__input_shape = sum(self.__state_feature_specs)

        self.__num_actions = num_actions

        # prepare output structure
        nn_layers = list()

        # construct nn hidden layers
        next_input_size = sum(self.__state_feature_specs)
        for layer_size in layers:
            next_layer = nn.Linear(
                in_features=next_input_size,
                out_features=layer_size,
                bias=True
            )
            next_input_size = layer_size
            nn_layers.append(next_layer)

            nn_layers.append(nn.BatchNorm1d(next_input_size))
            nn_layers.append(nn.ReLU())
            if dropout_rate > 0:
                nn_layers.append(nn.Dropout(dropout_rate))

        # add final layer
        next_layer = nn.Linear(
            in_features=next_input_size,
            out_features=self.__num_actions,
            bias=True
        )
        nn_layers.append(next_layer)
        nn_layers.append(nn.Softmax())

        # convert to sequential model and store
        self.__model = nn.Sequential(*nn_layers)

    @staticmethod
    def _one_hot_encode(class_idx: int, num_classes: int):
        """
        Converts class index to the one-hot representation.
        :param class_idx: int
        :param num_classes: int
        :return: np.ndarray
        """
        vec = np.zeros(num_classes)
        vec[class_idx] = 1
        return vec

    def _state_to_tensor(self, state: State) -> torch.Tensor:
        """
        Converts a state to the representing tensor, according to self.__state_feature_specs configuration.

        :param state: State instance.
        :return: torch.Tensor instance.
        """
        # extract state features
        state_features = state.features

        # prepare output vector
        feature_vector = np.zeros(self.__input_shape, dtype=np.float32)

        # fill feature_vector with feature values
        current_idx = 0
        for feature_idx in range(len(state_features)):
            feature_inputs = self.__state_feature_specs[feature_idx]
            feature_value = state_features[feature_idx]

            if feature_inputs == 1:
                feature_vector[current_idx] = feature_value
            else:
                feature_vector[current_idx:current_idx+feature_inputs] = \
                    QNet._one_hot_encode(feature_value, feature_inputs)

            current_idx += feature_inputs

        return torch.from_numpy(feature_vector)

    def forward(self, input_states: List[State]):
        """
        Execute forward pass and return logits.

        :param input_states: list of State elements.
        :return: list of logits vectors.
        """
        # convert each state to input tensor
        input_tensors = [
            self._state_to_tensor(state).unsqueeze(0) for state in input_states
        ]
        input_tensors_batch = torch.cat(input_tensors, dim=0)

        # feed model and return logits
        return self.__model(input_tensors_batch)
