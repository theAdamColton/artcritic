import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
import datasets

from artcritic.prompts import DiffusionDBPromptUpscaled

prompt_sys = \
"""You are an expert in evaluating AI image generation models. Respond to the following requests.
"""

prompt_qa = \
"""Your task is to write questions that can evalutate the instruction following capabilities of an AI image generator. The AI model is given a text instruction then generating an image that fufils the outlined instructions in the prompt. Your task is to use the text instructions to write a list of multiple-choice questions that will be used to evaluate the generated images.
Write challenging questions that can ensure that the composition and scene details are exactly as specified in the instructions.
Before you write the question, first write a plan for questions, what would be potential details that the AI image generator would not accurately portray? Think about challenging questions to evaluate the generated image.
Don't reuse the options from the previous example! The options should be difficult to answer! If you are asking what main objects are in the images, use options that are similar what is in the instructions for the AI image generator. In your plan for the questions, do not simply repeat the instruction prompt. Instead, try to think about the elements of the prompt that would be difficult for the AI generator to generate. Don't write questions asking about the resolution.
Write a plan for the questions, and then three or four multiple choice questions and thier answers for the following instruction. Do not number the questions.
"""

example_image_prompt_upscaled = \
"""In a fantastical world, we see a horse with a friendly, curious face wearing a spacesuit. The suit is a perfect fit, covering its entire body including its tail and mane. A clear glass dome protects the horse's head, showing its expression clearly. The suit is painted with a colorful design, making the horse look even more majestic. The horse is riding an astronaut-like vehicle. The vehicle has a high-tech, futuristic design. The two of them are speeding through outer space, the horse's hooves leaving trails behind them. Surrounding the scene are countless stars and galaxies, painting the space with a sense of grandeur. Digital painting created by an artist named Kirby, featuring high-quality, vibrant colors, and intricate detailing."""

assistant_start_response = "Plan for questions:\n"

example_assistant_response = \
"""Plan for questions:
What makes this prompt unique is the that the horse is riding on the astronaut, not the astronaut on the horse. The prompt states that the horse is riding on the astronaut like a vehicle, which is unexpected. The AI image generator might make a mistake and portray the astronaut riding the horse, instead of the horse riding the astronaut. Additionally, the image is a digital painting, and should have a smooth and sharp style with the clear focus being the horse and astronaut.

Question: What subjects are present in this image?
Options: 
A. A rocket ship and an astronaut
B. A space alien and a unicorn
C. An astronaut and a donkey
D. An astronaut and a horse
Answer: D

Question: What is being ridden in this image?
Options:
A. The horse
B. The astronaut
D. The rocket ship
C. Nothing is being ridden
Answer: B

Question: What medium is this art piece?
Options:
A. Digital painting
B. Oil on canvas
C. 3D rendering
D. Photograph
Answer: A"""

final_user_prompt = \
"""
Instruction:
{prompt}
"""



def parse_output_questions(output):
    """
    returns a list of questions as dicts
    """
    questions = []

    first_question_i = output.find("Question:")
    plan, qas = output[:first_question_i], output[first_question_i: ]

    plan = plan.strip()

    questions = []
    answers = []
    options = []

    for qa in qas.split("Question:"):
        qa = qa.strip()
        if len(qa) == 0:
            continue
        answer = qa[-1]

        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"{answer} is not a valid uppercase letter in question: {qa} in output {output}")

        # removes the options from the question

        question = qa.split("Options:")[0].strip()

        assert question[-1] == "?", question

        question_options = qa.split("Options:")[1].strip()
        # removes the answer from the options
        question_options = question_options[:-1].strip().removesuffix("Answer:").strip()

        questions.append(question)
        answers.append(answer)
        options.append(question_options)

    return {"prompt_qa_raw_output": output, "prompt_qa_questions": questions, "prompt_qa_answers": answers, "prompt_qa_plan":plan, "prompt_qa_options": options}


def main(
    model_url = "TheBloke/Mistral-7B-OpenOrca-AWQ",
    request_batch_size=128,
    max_n=500000,
    max_prompt_upscaled_str_len = 200,
    ):
    #diffusion_db = load_dataset("poloclub/diffusiondb", "2m_text_only", split="train").shuffle(1000)
    prompt_ds = DiffusionDBPromptUpscaled().ds

    max_n = min(max_n, len(prompt_ds))
    prompt_ds = iter(prompt_ds)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    print('loading model')
    model = LLM(model_url, quantization="awq", dtype="half", gpu_memory_utilization=0.90, swap_space=16, max_model_len=4096, )
    print('model loaded')

    output_data = []
    prog_bar = tqdm(total=max_n)


    while len(output_data) < max_n:
        chat_templates = []
        prompts_upscaled = []
        prompts= []
        while len(chat_templates) < request_batch_size:
            d = next(prompt_ds)
            prompt_upscaled = d['prompt_upscaled']
            prompt = d['prompt_original']

            prompt_upscaled = prompt_upscaled.strip().replace("\n", "")

            prompts.append(prompt)
            prompts_upscaled.append(prompt_upscaled)

            chat = [
                    {"role":"system", "content": prompt_sys},
                    {"role":"user", "content": prompt_qa + final_user_prompt.format(prompt=example_image_prompt_upscaled)},
                    {"role":"assistant", "content": example_assistant_response},
                    {"role": "user", "content": final_user_prompt.format(prompt=prompt_upscaled[:max_prompt_upscaled_str_len])},
                    {"role": "assistant", "content": assistant_start_response},
            ]
            chat_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            chat_templates.append(chat_template)

        outputs = model.generate(chat_templates, sampling_params)

        _added_n = 0
        for prompt_upscaled, prompt, output in zip(prompts_upscaled, prompts, outputs):

            output_text = output.outputs[0].text

            try:
                d = parse_output_questions(output_text)
            except Exception as e:
                print("error parsing output!", e)
                continue

            d.update({"prompt":prompt, "prompt_upscaled": prompt_upscaled})

            output_data.append(d)
            print(d)
            _added_n += 1

        prog_bar.update(_added_n)

    d = datasets.Dataset.from_list(output_data)
    os.makedirs("out/", exist_ok=True)
    d.save_to_disk(f"out/prompt_qa_dataset{max_n:08}")

if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
