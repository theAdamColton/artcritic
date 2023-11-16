import datasets

class DiffusionDBPromptUpscaled():
    def __init__(self, seed=42, split='train'):
        self.ds = datasets.load_dataset("adams-story/diffusiondb-prompt-upscale")[split].shuffle(seed=seed)
        self.iter = iter(self.ds)

    def __call__(self):
        # returns a dict with keys:
        # prompt
        # prompt_upscaled
        # prompt_qa_plan 
        # prompt_qa_options 
        # prompt_qa_answers
        # prompt_qa_questions
        return next(self.iter)
