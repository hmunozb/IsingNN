

class IsingConfig:
    def __init__(self, ising_l=None, training=None, decay=None):
        self.l = ising_l
        self.is_training = training
        self.weight_decay = decay


class TrainingConfig:
    def __init__(self, method='GRAD', rate=0.1):
        self.method = method
        self.learn_rate = rate


class MomentumTrainingConfig(TrainingConfig):
    def __init(self, rate=0.1, p=0.5):
        self.method = 'MOMNT'
        self.learn_rate = rate
        self.momentum = p

