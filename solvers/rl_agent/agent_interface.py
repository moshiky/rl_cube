
from abc import ABC, abstractmethod


class AgentInterface(metaclass=ABC):
    """
    Agent interface.
    """

    @abstractmethod
    def train(self, env, train_config, **kwargs):
        """
        Train the agent according to provided configuration.

        :param env: Environment class instance.
        :param train_config: training configuration.
        :return: list with rewards total for each epoch
        """
        raise NotImplementedError()

    @abstractmethod
    def eval(self, env, eval_config, **kwargs):
        """
        Evaluate the agent according to provided configuration.

        :param env: Environment class instance.
        :param eval_config: training configuration.
        :return: mean and std of epoch total rewards.
        """
        raise NotImplementedError()
