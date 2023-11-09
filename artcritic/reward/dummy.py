from artcritic.reward.reward import Reward


class DummyReward(Reward):
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, x, y):
        return x.mean(), -x.mean()
