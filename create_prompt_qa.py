import os
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import datasets

# Upscale prompts
upscale_prompt_sys = "You are an expert in art composition. Respond to the following requests."

prompt_upscale = \
"""You are an employee at an art/photography seminar, of the visual medium. Pieces are inspired by short text prompts. These prompts are intended to be interpreted open-endedly. Your job is to interpret these prompts, and write detailed descriptions that provide explicit compositions and art styles for the artists to use.

The artists need to be given detailed descriptions of all of the details they need to include in the piece. You need to describe every salient feature that goes into the finished work.

Dive into colors, facial expressions, metaphors, funny observations or jokes about what the piece looks like, comparisons, whatever you think will add vividness and detail to the piece. 

Caption:
"abstract painting with various colors and shapes, flat pastel, neutral color, graffiti, abstract art, greg rutkowski, artgerm, ross tran, chinese style"
Detailed Composition:
The painting is composed of rectangles, long and thin, all stacked horizontally except for the long verticals at the left and right edges of the painting - a large blue rectangle is nearly a squat square, taking up the lower portion of the painting from the bottom edge to just above center, stopping about 4 inches before the left edge of the canvas, and about 1 inch before the right edge of the canvas. Near it stops lay perfectly straight yet slightly uneven in black charcoal lines. Blue paint goes up to the black line, sometimes overlapping the line, sometimes touching the line, sometimes stopping right before the line. It leaves a gap where the pale yellow paint underneath shows through the blue. There is a blue stripe that is almost exactly the same width as the non-blue vertical rectangles on the left. This is blue vertical stripe is separated from the rest of the square by a bold charcoal line, again appearing and disappearing with the paint. To the right of the vertical blue stripe is one inch of pale pink. To the left of the blue squat square is a subdivided tall rectangle  The colors have been painted with very thin paint, and in various hues of one color per block, so the surface is not even, but modulated within the sometimes W-directional loose brushstrokes. The previous layers of color show through the top color as if it was tissue paper. This thin paint layering within straight rectangles - some edged with straight but uneven charcoal lines - continues throughout the painting.

Caption:
"3 d character design, male pop singer, vector art, digital art, portrait, 4 k, 8 k, sharp focus, smooth, music artist"
Detailed Composition:
"""

prompt_recaption_request = \
"""Good. Another work needs a detailed composition:

Caption:
"{caption}"
Detailed Composition:
"""

recaption_assistant_response_jss = \
"""A striking portrait, collaborative effort of John Singer Sargent and J.C. Leyendecker. Center of the frame are two women, identical twins, standing side by side in a frontal view. The twins appear to be in their early thirties and wear clothing from the early 20th century.
Both women have long, auburn hair cascading down their shoulders, with curls framing their faces. Their expressions convey a sense of worry and weariness. Their eyes are a deep, thoughtful brown, and their eyebrows are slightly furrowed, giving them a concerned appearance. Their lips are soft and downturned, adding to the sense of anxiety.
The women's skin is pale and flawless, a common trait in Sargent's portraits, and their cheeks carry a hint of rosy color. They wear conservative, high-collared blouses with lace detailing, buttoned up to their necks. Their attire is elegant and modest, reflecting the fashion of the era.
The background of the painting is somewhat muted, allowing the focus to remain on the twin women. The lighting is soft but highlights their faces, casting subtle shadows that add depth to the composition.
Sargent and Leyendecker's collaborative masterpiece captures the essence of the twin women's lives, portraying their inner turmoil and concerns, while showcasing the timeless elegance and grace that these renowned artists are known for."""

recaption_assistant_response_pop_singer = \
"""Intricately detailed male pop singer character, 3D render. Psychonautic strobe stage lighting. The character is a vibrant and dynamic lead singer. His face is locked into a wickedly contorted grin. He holds a microphone in his hands, and is mid-jump doing the splits in the air above the stage as he screams into the mic. The camera is positioned low to the stage, and uses a wide angle lens, the singer's legs extend almost from one side of the image to the other. He exudes charisma and confidence. He is wearing a rainbow pattern cross stiched in richly colored textures, and tight glistening leather pants, his eyes are wild and striking.
The background of the stage is filled with vivid, swirling lights that add a sense of movement and energy to the scene. Smoke curls up from the edges of the frame. The fellow band mates are visible, locked in the culmination of some great chord that seems to be held up wavering in the air, the drummer is furiously hitting the shining crash. The exertion is visible from their sweat and heaving chests and taut musculatures."""

recaption_assistant_response_diebenkorn = \
"""The painting is composed of rectangles that are long and thin, all stacked horizontally except for the long verticals at the left and right edges of the painting - the large blue rectangle is nearly a squat square, taking up the lower portion of the painting from the bottom edge to just above center, stopping about 4 inches before the left edge of the canvas, and about 1 inch before the right edge of the canvas. Near it stops lay perfectly straight yet slightly uneven in black charcoal lines. Blue paint goes up to the black line, sometimes overlapping the line, sometimes touching the line, sometimes stopping right before the line. It leaves a gap where the pale yellow paint underneath shows through the blue. There is a blue stripe that is almost exactly the same width as the non-blue vertical rectangles on the left. This is blue vertical stripe is separated from the rest of the square by a bold charcoal line, again appearing and disappearing with the paint. To the right of the vertical blue stripe is one inch of pale pink. To the left of the blue squat square is a subdivided tall rectangle  The colors have been painted with very thin paint, and in various hues of one color per block, so the surface is not even, but modulated within the sometimes W-directional loose brushstrokes. The previous layers of color show through the top color as if it was tissue paper. This thin paint layering within straight rectangles - some edged with straight but uneven charcoal lines - continues throughout the painting."""


# QA prompts
prompt_sys_qa = "You are an expert in evaluating AI image generation models. Respond to the following requests."

prompt_qa = \
"""Your task is to write a short multiple-choice quiz that can evalutate the capabilities of an AI image generator to generate an image given some certain image prompt. 
Write questions that, when answered in context with the final generated image, can ensure that the subject(s) and composition and scene details are exactly as specified in the image prompt. The questions should not be about abstract concepts, but should instead verify the objects and composition.
Before you write the question, first make a short plan for questions. Think about which crucial details the AI image generator might not accurately portray. The questions must test the fundemental visual elements of the work.
AI image generators consistantly fail to capture object permanance, and object placement and composition.
If the question you are asking is what main subjects and details exist in the image, try and use alternative subjects that are similar, but not exactly the same, as what is in the image prompt. In your plan for the questions, do not simply repeat the image prompt. Instead, try to think about the elements of the image prompt that would be difficult for a non-attentive AI generator. Don't write questions asking about the resolution, or artists, (unless the artists are very famous). Instead focus on specific objects, their exact positions/layouts, and the overall composition and spacial orientations of the objects in the image. The provided image prompt may contain some superfluous details concerning the art styles and artist information, which can be ignored.
"""

example_image_prompt = """horse in a spacesuit riding on astronaut - like vehicle. digital painting by kirby"""

example_image_prompt_upscaled = """In a fantastical world, we see a horse with a friendly, curious face wearing a spacesuit. The suit is a perfect fit, covering its entire body including its tail and mane. A clear glass dome protects the horse's head, showing its expression clearly. The suit is painted with a colorful design, making the horse look even more majestic. The horse is riding an astronaut-like vehicle. The vehicle has a high-tech, futuristic design. The two of them are speeding through outer space, the horse's hooves leaving trails behind them. Surrounding the scene are countless stars and galaxies, painting the space with a sense of grandeur. Digital painting created by an artist named Kirby, featuring high-quality, vibrant colors, and intricate detailing."""

assistant_start_response_qa = "Plan for questions:\n"

example_assistant_response_qa = \
"""Plan for questions:
The horse is riding on the astronaut, not the astronaut on the horse. The AI image generator might mistakenly portray the astronaut riding the horse, instead of the horse riding the astronaut. What details should the image have in order to emphasize that the horse is the one doing the riding? The horses legs, which are clothed in the space suit, should be draped around the astronaut. The image is in the digital art style, and should employ a wide-angle field of view to show the speed of the horse riding the astronaut-like vehicle, on which the horse is riding. The rider subject of the image should be heading approximately towards the camera in an exaggerated loping arc, with a trail of exotic space ship exhaust behind it.

Question: What is the subject of the image?
Options: 
A. An astronaut riding a rocket ship
B. A space alien riding a unicorn
C. An astronaut riding a donkey
D. A horse riding an astronaut-like vehicle
E. An astronaut riding a horse
Answer: D

Question: What is being ridden in this image?
Options:
A. The horse
B. The astronaut
D. The space ship
C. Nothing
Answer: B

Question: What is covering the horse?
Options:
A. The saddle and the astronaut's legs
B. A well-fitted space suit
C. Armored horse barding
D. The horse is bare skinned
Answer: B"""

first_user_prompt_qa = \
"""Image Prompt:
\"{prompt}\"
"""

final_user_prompt_qa = \
"""

Make sure to write a short plan for the questions, prefixed by "Plan for questions:", then three or four multiple choice questions and thier answers, prefixed with "Question:". Answer with the option's letter from the given choices directly. There is no case where the answer is undeterminable or not specified by the prompt.

Image Prompt:
\"{prompt}\"

Plan for Questions:
"""


def parse_output_questions(output):
    """
    returns a list of questions as dicts
    """
    questions = []

    first_question_i = output.find("Question:")
    plan, qas = output[:first_question_i], output[first_question_i:]

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
            raise ValueError(
                f"{answer} is not a valid uppercase letter in question: {qa} in output {output}\n\n\n\n"
            )

        # removes the options from the question

        question = qa.split("Options:")[0].strip()

        if question[-1] != "?":
            raise ValueError(f"question:{question}, output: {output}")

        question_options = qa.split("Options:")[1].strip()
        # removes the answer from the options
        question_options = question_options[:-1].strip().removesuffix("Answer:").strip()

        questions.append(question)
        answers.append(answer)
        options.append(question_options)

    if len(questions) ==0:
        raise ValueError("No questions found!", output, "\n\n\n\n\n")

    return {
        "prompt_qa_raw_output": output,
        "prompt_qa_questions": questions,
        "prompt_qa_answers": answers,
        "prompt_qa_plan": plan,
        "prompt_qa_options": options,
    }

def create_prompt_upscale_prompt(image_prompt, tokenizer):
    chat = [
        {"role": "system", "content": upscale_prompt_sys},
        {"role": "user", "content": prompt_upscale},
        {
            "role": "assistant",
            "content": recaption_assistant_response_pop_singer,
        },
        {
            "role": "user",
            "content": prompt_recaption_request.format(caption=image_prompt),
        },
    ]
    chat_template = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    return chat_template

def create_prompt_qa_prompt(image_prompt_upscaled, tokenizer):
    chat = [
        {"role": "system", "content": prompt_sys_qa},
        {
            "role": "user",
            "content": prompt_qa
            + first_user_prompt_qa.format(prompt=example_image_prompt_upscaled) + \
            f"\nHere is an an example response: \n{example_assistant_response_qa}\n\n{final_user_prompt_qa.format(prompt=image_prompt_upscaled)}",
         },
    ]
    chat_template = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    return chat_template


def main(
    model_url: str = "Open-Orca/Mistral-7B-OpenOrca",
    tokenizer_url: Optional[str] = None,
    quantization: Optional[str] = None,
    request_batch_size=512,
    max_n=500000,
    max_prompt_str_len=300,
):
    diffusion_db = datasets.load_dataset("poloclub/diffusiondb", "2m_text_only", split="train").shuffle(1000)

    max_n = min(max_n, len(diffusion_db))
    diffusion_db = iter(diffusion_db)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
    if not tokenizer_url:
        tokenizer_url = model_url
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)

    print("loading model")
    model = LLM(
        model_url,
        quantization=quantization,
        dtype="half",
        gpu_memory_utilization=0.90,
        swap_space=16,
        max_model_len=4096,
    )
    print("model loaded")

    output_data = []
    prog_bar = tqdm(total=max_n)
    used_image_prompts = set()

    while len(output_data) < max_n:

        # first makes a batch of upscaled image prompts
        print("Making upscaled image prompts...")
        prompt_upscale_prompts = []
        image_prompts = []
        while len(prompt_upscale_prompts) < request_batch_size:
            image_prompt = next(diffusion_db)["prompt"]

            if image_prompt in used_image_prompts:
                continue

            used_image_prompts.add(image_prompt)
            image_prompts.append(image_prompt)

            prompt_upscale_prompt = create_prompt_upscale_prompt(image_prompt.strip().replace("\n", "")[:max_prompt_str_len],
                                                                  tokenizer)
            prompt_upscale_prompts.append(prompt_upscale_prompt)

        outputs = model.generate(prompt_upscale_prompts, sampling_params)

        upscaled_image_prompts = []

        for output in outputs:
            upscaled_image_prompts.append(
                    output.outputs[0].text.strip().replace('"', "").replace("\n", " "),
            )

        # then uses the upscaled_image_prompts to make a batch of
        # multiple-choice qas
        print("Making image qa prompts...")
        chat_prompt_qa_prompts = []
        for prompt_upscaled in upscaled_image_prompts:
            chat_prompt_qa_prompt = create_prompt_qa_prompt(prompt_upscaled[:max_prompt_str_len], tokenizer)
            chat_prompt_qa_prompts.append(chat_prompt_qa_prompt)


        outputs = model.generate(chat_prompt_qa_prompts, sampling_params)

        _added_n = 0
        for output, image_prompt, upscaled_image_prompt in zip(
                outputs, image_prompts, upscaled_image_prompts):

            output_text = output.outputs[0].text

            try:
                d = parse_output_questions(output_text)
            except Exception as e:
                print("error parsing output!", e)
                continue

            output_data.append({
                "prompt": image_prompt,
                "prompt_upscaled": upscaled_image_prompt,
                **d
                })
            _added_n += 1

        prog_bar.update(_added_n)

        print("Done with batch", "" if len(output_data) == 0 else output_data[-1])

        d = datasets.Dataset.from_list(output_data)
        os.makedirs("out/", exist_ok=True)
        d.save_to_disk(f"out/prompt_qa_dataset{max_n:08}")


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
