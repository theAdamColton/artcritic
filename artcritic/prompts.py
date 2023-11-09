import datasets

class DiffusionDBPromptUpscaled():
    def __init__(self, seed, split='train'):
        self.ds = datasets.load_dataset("adams-story/diffusiondb-prompt-upscale")[split].shuffle(seed=seed)
        self.iter = iter(self.ds)

    def __call__(self):
        return next(self.iter), {}
