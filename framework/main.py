from config import general_config
from simulators.tower_of_hanoi.rs_logic import get_shaping_signal
from simulators.tower_of_hanoi.similarity_logic import get_similarity_group
from simulators.tower_of_hanoi.simulator import Simulator
from solvers.rl_agent.agent import Agent
from solvers.rl_agent.logics.tabular_q_learning import TabularQLearning


def main():
    # create environment
    env = Simulator(num_floors=5, verbose=False)

    # create agent logic
    agent_logic = TabularQLearning(
        action_type=env.get_actions(),
        epsilon=0.05,
        alpha=0.1,
        gamma=0.999,
        rs_logic=get_shaping_signal,
        similarity_logic=get_similarity_group
    )

    # create agent
    agent = Agent(
        agent_logic=agent_logic
    )

    # iterate train and eval
    for train_iter_idx in range(10000):
        print('== train iteration #{}'.format(train_iter_idx))

        epoch_history = agent.train(
            env=env,
            train_config=general_config.train_config
        )
        print('> train scores: {:.4f} [{}]'.format(epoch_history.mean(), epoch_history))

        mean, std = agent.eval(
            env=env,
            eval_config=general_config.eval_config,
            verbose=False
        )
        print('> eval scores: {:.4f} [{:.4f}]'.format(mean, std))

        if mean == 1:
            break


if __name__ == '__main__':
    main()
