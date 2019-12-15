from config import general_config
from cube_simulator.rubiks_cube import RubiksCube


def run_epoch(agent, is_train_epoch):
    """

    :param agent:
    :param is_train_epoch:
    :return:
    """
    # initiate cube
    cube = RubiksCube(edge_size=general_config.game.cube_edge_size)

    #