from typing import List
import torch
import torch.nn as nn
import numpy as np

from framework.state import State


class QNet(nn.Module):
    """
    Implements q net module.
    """
    def __init__(self, state_feature_specs, num_actions, layers, dropout_rate, use_gpu):
        """
        Initiate the module.

        :param state_feature_specs: list of positive non-zero integers. [k0, k1, ..., kn]
            if ki = 1 - numeric feature
            if ki >= 2 - categorical feature with k classes
            otherwise- error!
        :param num_actions: integer. number of actions.
        :param layers: layers configuration. list of integers.
        :param dropout_rate: float.
        :param use_gpu: boolean. whether or not to use the gpu.
        """
        # call base c'tor
        super().__init__()

        # store configuration
        self.__state_feature_specs = state_feature_specs
        self.__input_shape = sum(self.__state_feature_specs)
        self.__use_gpu = use_gpu

        self.__num_actions = num_actions

        # prepare output structure
        nn_layers = list()

        # construct nn hidden layers
        print('Network structure:')
        next_input_size = sum(self.__state_feature_specs)
        for layer_size in layers:
            print('>> layer: {} -> {}'.format(next_input_size, layer_size))
            next_layer = nn.Linear(
                in_features=next_input_size,
                out_features=layer_size,
                bias=True
            )
            next_input_size = layer_size
            nn_layers.append(next_layer)

            print('>> layer: batch-norm')
            nn_layers.append(nn.BatchNorm1d(next_input_size))

            print('>> layer: relu')
            nn_layers.append(nn.ReLU())

            if dropout_rate > 0:
                print('>> layer: dropout {}'.format(dropout_rate))
                nn_layers.append(nn.Dropout(dropout_rate))

        # add final layer
        print('>> layer: {} -> {}'.format(next_input_size, self.__num_actions))
        next_layer = nn.Linear(
            in_features=next_input_size,
            out_features=self.__num_actions,
            bias=True
        )
        nn_layers.append(next_layer)

        # convert to sequential model and store
        self.__model = nn.Sequential(*nn_layers)
        self.__model.train()

        if self.__use_gpu:
            self.__model.cuda()

    def forward(self, input_states: np.array):
        """
        Execute forward pass and return logits.

        :param input_states: np.array representation of state batch.
        :return: list of logits vectors.
        """
        input_states = torch.from_numpy(input_states)

        if self.__use_gpu:
            input_states = input_states.cuda()

        # feed model and return logits
        return self.__model(input_states)
