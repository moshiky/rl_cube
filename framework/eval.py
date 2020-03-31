from simulators.tower_of_hanoi.simulator import Simulator
from solvers.rl_agent.logics.dqn.dqn import DQN


def visualize_epoch(env, policy_method, max_epoch_steps):
    """
    Visualize single epoch.

    :param env:
    :param policy_method:
    :param max_epoch_steps:
    :return:
    """
    # initiate environment
    s_t = env.reset(verbose=True)
    print('Initial state:')
    env.visualize()

    # act and learn until epoch end- final state or max epoch steps
    epoch_steps = 0
    while not s_t.is_final and epoch_steps < max_epoch_steps:
        # get next action
        a_t = policy_method(s_t)
        print('[{}] selected action: {}'.format(epoch_steps, a_t.action_value))

        # act in environment
        s_next, r_t = env.act(a_t)
        print('[{}] reward: {}'.format(epoch_steps, r_t))

        env.visualize()

        # prepare for next iteration
        s_t = s_next
        epoch_steps += 1

    print('Game over.')


if __name__ == '__main__':
    # create environment
    env = Simulator(num_floors=3, verbose=False)

    # get policy method
    policy_method = DQN.get_policy_func(
        r"C:\files\train_dir\dqn_01\model__step_04700.pt",
        env.get_state_feature_specs(),
        env.get_actions()
    )

    visualize_epoch(env, policy_method, max_epoch_steps=20)
