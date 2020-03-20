from config import general_config
from simulators.tower_of_hanoi.simulator import Simulator
from solvers.rl_agent.agent import Agent
from solvers.rl_agent.logics.tabular_q_learning import TabularQLearning


def main():
    # create environment
    env = Simulator(num_floors=3, verbose=False)

    # create agent logic
    agent_logic = TabularQLearning(
        action_type=env.get_actions(),
        epsilon=0.05,
        alpha=0.1,
        gamma=1.0
    )

    # create agent
    agent = Agent(
        agent_logic=agent_logic
    )

    # iterate train and eval
    for train_iter_idx in range(150):
        print('== train iteration #{}'.format(train_iter_idx))

        epoch_history = agent.train(
            env=env,
            train_config=general_config.train_config
        )
        print('> train scores: {}'.format(epoch_history))

        mean, std = agent.eval(
            env=env,
            eval_config=general_config.eval_config
        )
        print('> eval scores: {:.4f} [{:.4f}]'.format(mean, std))


if __name__ == '__main__':
    main()
