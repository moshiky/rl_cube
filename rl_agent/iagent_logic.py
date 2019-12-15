
from abc import ABC, abstractmethod


class IAgentLogic(metaclass=ABC):
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
    def update(self, s_t0, a, r, s_t1):
        """
        Updates agent's logic using experience information.

        :param s_t0: current state.
        :param a: current action.
        :param r: current reward.
        :param s_t1: next state.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def next_action(self, s_t0):
        """
        Returns action for given state, according to current logic.

        :param s_t0: current state.
        :return: action.
        """
        raise NotImplementedError()
