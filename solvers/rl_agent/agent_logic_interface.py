
from abc import ABCMeta, abstractmethod

from framework.action import Action
from framework.state import State


class AgentLogicInterface(metaclass=ABCMeta):
    """
    Agent logic interface.
    """

    @abstractmethod
    def new_epoch(self):
        """
        Perform required operations on epoch start.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, s_t0: State, a: Action, r: float, s_t1: State) -> float:
        """
        Updates agent's logic using experience information.

        :param s_t0: current state.
        :param a: current action.
        :param r: current reward.
        :param s_t1: next state.
        :return: integer- reward signal
        """
        raise NotImplementedError()

    @abstractmethod
    def next_action(self, s_t, is_train_mode):
        """
        Returns action for given state, according to current logic.

        :param s_t: current state.
        :param is_train_mode: boolean.
        :return: action.
        """
        raise NotImplementedError()
