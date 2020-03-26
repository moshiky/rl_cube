import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import consts
from solvers.rl_agent.agent_logic_interface import AgentLogicInterface
from solvers.rl_agent.logics import logics_config


class DQN(AgentLogicInterface):
    """
    Implements Deep Q-Learning algorithm.
    """

    def __init__(self, state_feature_specs, action_type, train_dir_path, rs_logic=None, similarity_logic=None):
        """
        Initiate logic instance.

        :param state_feature_specs: list of positive non-zero integers. [k0, k1, ..., kn]
            if ki = 1 - numeric feature
            if ki >= 2 - categorical feature with k classes
            otherwise- error!
        :param action_type: instances of ActionType.
        :param train_dir_path: string. path to dir to store the checkpoint files.
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
        self.__train_dir_path = train_dir_path
        self.__rs_logic = rs_logic
        self.__similarity_logic = similarity_logic
        self.__step_idx = 0

        if not os.path.exists(self.__train_dir_path):
            os.makedirs(self.__train_dir_path)

        # initiate q network members
        self.__memory = list()

        self.__q_net = self.__construct_network()
        self.__target_net = self._store_q_net_and_load()

    def __construct_network(self) -> nn.Module:
        """
        Construct and return internal q network according to configuration.
        :return: nn.Module
        """


    def new_epoch(self):
        """
        Interface method implementation.
        """
        pass

    def update(self, s_t0, a, r, s_t1):
        """
        Interface method implementation.
        """
        # store experience to memory
        self.__memory.append((s_t0, a, r, s_t1))
        if len(self.__memory) > logics_config.dqn.memory_size:
            self.__memory = self.__memory[1:]

        # create train batch
        if len(self.__memory) >= logics_config.dqn.batch_size:

            # select samples for batch
            batch_sample_idxs = np.random.randint(len(self.__memory))
            batch_samples = np.array(self.__memory, dtype=object)[batch_sample_idxs]

            # construct input
            input_vecs = list()
            for sample in batch_samples:
                in_vec, out_vec = self._produce_sample_vecs(sample)

            # increase step idx
            self.__step_idx += 1

    def next_action(self, s_t, is_train_mode):
        """
        Interface method implementation.
        """
        pass

    def _store_q_net_and_load(self) -> nn.Module:
        """
        Store current q net to disk and load into target net.

        :return: nn.Module.
        """
        # store q net to disk
        ckpt_file_path = os.path.join(self.__train_dir_path, 'model__step_{}.pt'.format(self.__step_idx))
        torch.save(self.__q_net, ckpt_file_path)

        # load saved model and return
        return torch.load(ckpt_file_path)
