from typing import Tuple, List

import numpy as np
from abc import ABCMeta, abstractmethod

from framework.action import Action
from framework.action_type import ActionType
from framework.state import State


class SimulatorInterface(metaclass=ABCMeta):
    """
    Environment simulator interface.
    """

    @abstractmethod
    def reset(self) -> State:
        """
        Resets the environment and returns the initial state.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, action: Action) -> Tuple[State, float]:
        """
        Simulates the specified action and returns the reward signal.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def get_state(self) -> State:
        """
        Returns environment's current state.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def get_actions(self) -> ActionType:
        """
        Returns action type configuration.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_state_feature_specs(self) -> List[int]:
        """
        Returns the state feature specs.
        """
        raise NotImplementedError()

    @abstractmethod
    def visualize(self):
        """
        Produce environment visualization.
        :return:
        """
        raise NotImplementedError()
