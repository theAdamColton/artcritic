import datasets
import random

class DiffusionDB:
    def __init__(self, seed=42, split="train"):
        self.ds = datasets.load_dataset("poloclub/diffusiondb", "2m_text_only", split="train").shuffle(seed=seed)
        if split=='train':
            self.ds = self.ds.train_test_split(test_size = 0.05, seed=seed)['train']
        elif split == 'test':
            self.ds = self.ds.train_test_split(test_size = 0.05, seed=seed)['test']
        self.iter = iter(self.ds)

    def __call__(self):
        return next(self.iter)

class DiffusionDBPromptQA:
    def __init__(self, seed=42, split="train"):
        self.ds = datasets.load_dataset("adams-story/diffusiondb-prompt-upscale")[
            split
        ].shuffle(seed=seed)
        # filters
        self.ds = self.ds.filter(lambda x: len(x['prompt_qa_questions']) > 0)

        # select random questions, and thier answers and options
        
        def select_random_question(row):
            i = random.randint(0, len(row['prompt_qa_questions'])-1)
            row['prompt_qa_questions'] = [row['prompt_qa_questions'][i]]
            row['prompt_qa_options'] = [row['prompt_qa_options'][i]]
            row['prompt_qa_answers'] = [row['prompt_qa_answers'][i]]
            return row

        self.ds = self.ds.map(select_random_question)

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
