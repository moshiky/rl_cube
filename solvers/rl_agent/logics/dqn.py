
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from config import consts
from solvers.rl_agent.agent_logic_interface import AgentLogicInterface
from solvers.rl_agent.logics import logics_config


class DQN(AgentLogicInterface):
    """
    Implements Deep Q-Learning algorithm.
    """

    def __init__(self, state_feature_specs, action_type, rs_logic=None, similarity_logic=None):
        """
        Initiate logic instance.

        :param state_feature_specs: list of positive non-zero integers. [k0, k1, ..., kn]
            if ki = 1 - numeric feature
            if ki >= 2 - categorical feature with k classes
            otherwise- error!
        :param action_type: instances of ActionType.
        :param rs_logic: reward shaping logic to apply.
            prototype: func(s_t, a_t, s_t1) -> float
        :param similarity_logic: similarity logic to apply.
            prototype: func(State, Action) -> list[(State, Action, float)]
        """
        # verify configuration
        assert (np.array(state_feature_specs) > 0).all(), \
            'invalid feature configuration! {}'.format(state_feature_specs)

        assert action_type.values_type == consts.ActionTypeConsts.CATEGORICAL_VALUES_TYPE, \
            'categorical action supported only in tabular q learning.'

        # store configuration
        self.__state_feature_specs = state_feature_specs
        self.__action_type = action_type
        self.__rs_logic = rs_logic
        self.__similarity_logic = similarity_logic

        # initiate q network
        self.__q_net = self.__construct_network()

    def __construct_network(self) -> nn.Module:
        """
        Construct and return internal q network according to configuration.
        :return: nn.Module
        """
        # prepare output structure
        nn_layers = list()

        # construct nn hidden layers
        next_input_size = sum(self.__state_feature_specs)
        for layer_size in logics_config.dqn.layers:
            next_layer = nn.Linear(
                in_features=next_input_size,
                out_features=layer_size,
                bias=True
            )
            next_input_size = layer_size
            nn_layers.append(next_layer)

            nn_layers.append(nn.BatchNorm1d(next_input_size))
            nn_layers.append(nn.ReLU())
            if logics_config.dqn.dropout_rate > 0:
                nn_layers.append(nn.Dropout(logics_config.dqn.dropout_rate))

        # add final layer
        num_actions = len(self.__action_type.action_values)
        next_layer = nn.Linear(
            in_features=next_input_size,
            out_features=num_actions,
            bias=True
        )
        nn_layers.append(next_layer)
        nn_layers.append(nn.Softmax())

        # convert to sequential model and return
        return nn.Sequential(*nn_layers)

    def new_epoch(self):
        """
        Interface method implementation.
        """
        pass

    def update(self, s_t0, a, r, s_t1):
        """
        Interface method implementation.
        """
        pass

    def next_action(self, s_t, is_train_mode):
        """
        Interface method implementation.
        """
        pass
