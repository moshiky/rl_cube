
import numpy as np

from solvers.rl_agent.agent_interface import AgentInterface


class Agent(AgentInterface):
    """
    This class represents an RL agent.
    Uses an IAgentLogic instance to learn.
    """
    def __init__(self, agent_logic):
        """
        Initiate agent instance.

        :param agent_logic: IAgentLogic instance.
        """
        self.__agent_logic = agent_logic

    def train(self, env, train_config, **kwargs):
        """
        Interface method implementation.
        """
        # execute required number of epochs in train mode
        num_epochs = train_config.num_epochs
        max_epoch_steps = train_config.max_epoch_steps
        return self._run_epochs(env, num_epochs, max_epoch_steps, is_train_mode=True)

    def eval(self, env, eval_config, **kwargs):
        """
        Interface method implementation.
        """
        # execute required number of epochs in eval mode
        num_epochs = eval_config.num_epochs
        max_epoch_steps = eval_config.max_epoch_steps
        epoch_reward = self._run_epochs(env, num_epochs, max_epoch_steps, is_train_mode=False)

        return epoch_reward.mean(), epoch_reward.std()

    def _run_epochs(self, env, num_epochs, max_epoch_steps, is_train_mode):
        """
        Execute multiple epoch and return rewards total for each epoch.

        :param env: Environment to act in
        :param max_epoch_steps: integer. max steps for epoch.
        :param is_train_mode: boolean.
        :return: np.ndarray- rewards total for each epoch.
        """
        # prepare output
        epoch_reward = np.zeros(num_epochs, dtype=np.float64)

        # iterate num epochs
        for epoch_idx in range(num_epochs):

            # reset environment state
            s_t = env.reset()

            # act and learn until epoch end- final state or max epoch steps
            epoch_steps = 0
            total_reward = 0
            while not s_t.is_final and epoch_steps < max_epoch_steps:
                # get next action
                a_t = self.__agent_logic.next_action(s_t, is_train_mode=is_train_mode)

                # act in environment
                s_next, r_t = env.act(a_t)
                total_reward += r_t

                # in train mode- update logic
                if is_train_mode:
                    self.__agent_logic.update(s_t, a_t, r_t, s_next)

                # prepare for next iteration
                s_t = s_next
                epoch_steps += 1

            # epoch ended, prepare for next epoch
            epoch_reward[epoch_idx] = total_reward

        return epoch_reward
