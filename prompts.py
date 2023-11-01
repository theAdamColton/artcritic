import datasets

class diffusiondb_upscaled():
    def __init__(self):
        self.ds = datasets.load_dataset("adams-story/diffusiondb-prompt-upscale")['train'].shuffle()['prompt_upscaled']
        self.iter = iter(self.ds)

    def __call__(self):
        return next(self.iter), {}
