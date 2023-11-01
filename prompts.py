import datasets

class diffusiondb_upscaled():
    def __init__(self):
        self.ds = datasets.load_dataset(path)['train']['prompt_upscaled'].shuffle()
        self.iter = iter(self.ds)

    def __call__(self):
        return next(self.iter)
