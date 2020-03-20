
import numpy as np

from config import consts
from framework.action import Action
from solvers.rl_agent.agent_logic_interface import AgentLogicInterface


class TabularQLearning(AgentLogicInterface):
    """
    Implements Tabular Q Learning logic.
    """

    def __init__(self, action_type, epsilon, alpha, gamma):
        """
        Initiate logic instance.

        :param action_type: instances of ActionType.
        :param epsilon: float. exploration rate parameter.
        :param alpha: float. learning rate.
        :param gamma: float. horizon scaling parameter.
        """
        # verify configuration
        assert action_type.values_type == consts.ActionTypeConsts.CATEGORICAL_VALUES_TYPE, \
            'categorical action supported only in tabular q learning.'

        # store configuration
        self.__action_type = action_type
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__gamma = gamma

        # initiate q table
        self.__q_table = dict()

    def new_epoch(self):
        """
        Interface method implementation.
        """
        # no action required
        pass

    def update(self, s_t0, a, r, s_t1):
        """
        Interface method implementation.
        """
        # get current state q value
        s_t0_key = str(s_t0)
        s_t0_q_value = self.__q_table[s_t0_key][a]

        # get next best action q value
        best_next_action_q_value = self._get_best_action_value(s_t1)[1]

        # calculate new q value for s_t0
        new_s_t0_q_value = s_t0_q_value + self.__alpha * (r + self.__gamma * best_next_action_q_value - s_t0_q_value)

        # set new q value
        self.__q_table[s_t0_key][a] = new_s_t0_q_value

    def next_action(self, s_t, is_train_mode):
        """
        Interface method implementation.
        """
        # check which mode it is and apply epsilon-greedy policy
        if is_train_mode and np.random.rand() < self.__epsilon:
            # select random action
            selected_value = np.random.choice(self.__action_type.action_values)

        else:
            # select best action value
            selected_value = self._get_best_action_value(s_t)[0]

        # wrap value with action class
        return Action(selected_value, self.__action_type)

    def _get_best_action_value(self, s_t):
        """
        Returns the optimal action for the given state.
        :param s_t: state.
        :return: best action for given state, and q value for the specified action.
        """
        # verify state in table
        action_values = self.__action_type.action_values
        state_key = str(s_t)
        if state_key not in self.__q_table.keys():
            self.__q_table[state_key] = {
                action_value: 0.0 for action_value in action_values
            }

        # get q values for each of state actions
        action_q_values = self.__q_table[str(s_t)]

        # get best q value
        best_q = max(action_q_values.values())

        # select matching actions
        best_action_values = list(filter(lambda a: action_q_values[a] == best_q, action_values))

        # select random action value among best ones
        return np.random.choice(best_action_values), best_q
