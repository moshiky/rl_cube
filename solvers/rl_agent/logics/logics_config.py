
class common:
    epsilon = 0.05


class dqn:

    lr = 0.1

    layers = [10] * 5
    dropout_rate = 0.1

    batch_size = 32
    memory_size = 10000

    target_update_interval = 10
