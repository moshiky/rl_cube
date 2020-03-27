import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import consts
from framework.action import Action
from framework.state import State
from solvers.rl_agent.agent_logic_interface import AgentLogicInterface
from solvers.rl_agent.logics import logics_config
from solvers.rl_agent.logics.dqn.q_net import QNet


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

        self.__q_net = QNet(
            state_feature_specs=self.__state_feature_specs,
            num_actions=len(self.__action_type.action_values),
            layers=logics_config.dqn.layers,
            dropout_rate=logics_config.dqn.dropout_rate
        )
        self.__target_net = self._store_q_net_and_load()

        # define optimizer
        self.__optimizer = torch.optim.Adam(
            params=self.__q_net.parameters(),
            lr=logics_config.dqn.lr
        )

    def new_epoch(self):
        """
        Interface method implementation.
        """
        pass

    def update(self, s_t0: State, a: Action, r: float, s_t1: State) -> float:
        """
        Interface method implementation.
        """
        # store experience to memory
        self.__memory.append((s_t0, a, r, s_t1))
        if len(self.__memory) > logics_config.dqn.memory_size + 200:
            self.__memory = self.__memory[200:]

        # create train batch
        batch_size = logics_config.dqn.batch_size
        if len(self.__memory) >= batch_size:

            # select samples for batch
            batch_sample_idxs = np.random.randint(len(self.__memory), size=batch_size)
            batch_samples = np.array(self.__memory, dtype=object)[batch_sample_idxs]

            # extract batch elements
            s_t0_batch = [x[0] for x in batch_samples]
            a_batch = [int(x[1].action_value) for x in batch_samples]
            r_batch = [x[2] for x in batch_samples]
            s_t1_batch = [x[3] for x in batch_samples]

            # get q(s_t0, a) for the batch
            s_t0_q_net_outputs = self.__q_net(s_t0_batch).cpu()
            y_hat_values = s_t0_q_net_outputs[range(batch_size), a_batch]

            # get target values
            s_t1_q_net_outputs = self.__target_net(s_t1_batch).cpu()
            best_a_values = s_t1_q_net_outputs.max(dim=1)
            y_values = torch.from_numpy(np.array(r_batch)) + logics_config.common.gamma * best_a_values.values

            # calculate loss and perform back-propagation
            loss = F.mse_loss(y_hat_values, y_values)
            loss.backward()
            self.__optimizer.step()
            self.__optimizer.zero_grad()

            # increase step idx
            self.__step_idx += 1

        # update target network in intervals
        if self.__step_idx % logics_config.dqn.target_update_interval == 0:
            self.__target_net = self._store_q_net_and_load()

        return r

    def next_action(self, s_t, is_train_mode) -> Action:
        """
        Interface method implementation.
        """
        # check which mode it is and apply epsilon-greedy policy
        if is_train_mode and np.random.rand() < logics_config.common.epsilon:
            # select random action
            selected_value = np.random.choice(self.__action_type.action_values)

        else:
            # select best action value
            s_t_q_net_output = self.__target_net([s_t]).cpu()[0]
            best_q_value = s_t_q_net_output.max()
            best_a_idxs = np.where(s_t_q_net_output == best_q_value)[0].tolist()
            selected_value = np.random.choice(best_a_idxs)

        return Action(selected_value, self.__action_type)

    def _store_q_net_and_load(self) -> nn.Module:
        """
        Store current q net to disk and load into target net.

        :return: nn.Module.
        """
        # store q net to disk
        ckpt_file_path = os.path.join(self.__train_dir_path, 'model__step_{}.pt'.format(str(self.__step_idx).zfill(5)))
        torch.save(self.__q_net, ckpt_file_path)

        # load saved model and return
        loaded_model = torch.load(ckpt_file_path)
        loaded_model.eval()
        loaded_model.cuda()
        return loaded_model
