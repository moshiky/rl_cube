
class common:
    epsilon = 0.5
    gamma = 0.99


class dqn:

    lr = 0.01

    layers = [100] * 5
    dropout_rate = 0.1

    batch_size = 256
    memory_size = 100000

    target_update_interval = 50
