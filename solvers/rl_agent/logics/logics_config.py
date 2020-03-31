
class common:
    epsilon = 0.5
    gamma = 0.99


class dqn:

    lr = 0.01

    layers = [10] * 5
    dropout_rate = 0.1

    batch_size = 512
    memory_size = 100000

    target_update_interval = 100
