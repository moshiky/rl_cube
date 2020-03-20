
import numpy as np

from config import general_config
from simulators.cube_simulator.rubiks_cube import RubiksCube
from framework.plot_manager import PlotManager


def run_epoch(agent, is_train_epoch):
    """

    :param agent:
    :param is_train_epoch:
    :return:
    """
    # initiate cube
    cube = RubiksCube(edge_size=general_config.game.cube_edge_size)

    # scramble cube
    cube.scramble(general_config.game.num_scramble_moves)
    while cube.is_solved():
        cube.scramble(general_config.game.num_scramble_moves)

    # set training mode
    agent.set_mode(is_train_epoch)

    # run game moves until cube is solved, or max epoch steps is reached
    epoch_step = 0
    is_solved = False
    epoch_rewards = list()
    while not is_solved and epoch_step < general_config.epoch.max_steps:
        # perform next action
        reward = agent.act_and_update(cube)

        # store reward
        epoch_rewards.append(reward)

        # update iteration locals
        epoch_step += 1
        is_solved = cube.is_solved()

    return is_solved, sum(epoch_rewards)


def evaluate(agent):
    """

    :param agent:
    :return:
    """
    # run evaluation epochs
    epoch_rewards = list()
    for epoch_idx in range(general_config.eval.evaluation_epochs):
        epoch_rewards.append(run_epoch(agent, is_train_epoch=False))

    # return evaluation score
    return np.mean(epoch_rewards)


def train(agent):
    """

    :param agent:
    :return:
    """
    # initiate locals
    train_epoch_score_log = list()
    eval_score_log = list()

    max_intervals = general_config.plot_confs.max_intervals
    interval_length = general_config.train.evaluation_interval

    # initiate plot manager
    plot_manager = PlotManager(max_intervals, interval_length)

    # run train epochs
    for epoch_idx in range(general_config.train.train_epochs):

        # execute evaluation in intervals
        if epoch_idx % interval_length == 0:
            eval_score = evaluate(agent)
            eval_score_log.append(eval_score)

            plot_manager.add_eval_score(epoch_idx, eval_score_log)

        # execute train epoch
        epoch_score = run_epoch(agent, is_train_epoch=True)
        train_epoch_score_log.append(epoch_score)

        plot_manager.add_eval_score(epoch_idx, epoch_score)

        # plot current state
        plot_manager.update_plots()
