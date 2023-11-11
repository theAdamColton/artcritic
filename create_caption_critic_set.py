import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
import datasets



prompt_sys = \
"""
You are an expert on art and accessibility. Respond to the following requests.
"""

prompt_upscale = \
"""
The below is a short caption, describing a image on the internet. Some of the details are missing from the caption. It is lacking details about composition, subject, technique, line, shape and form, and details of the creation and medium. Using the imperative-voice, using a couple sentences of brief language, provide a more detailed caption, fixing the lack of details. It is important that every feature of the image is thouroughly described; the description will be used to provide accessibility captioning for those with impared vision.
"""

prompt_recaption = \
"""
You are an employee at an art/photography seminar, of the visual medium. Works have short captions. These existing captions are not detailed enough. Your job is to write object centric descriptions for the blind.

Even though the patron's requests are not detailed, there are many details in the pieces. The descriptions need to describe every salient feature in the images.

Dive into colors, facial expressions, metaphors, funny observations or jokes about what the piece looks like, comparisons, whatever you think will add vividness and detail to the non-seeing person’s experience. This is also the part where they may ask specific or general questions and you can get the m the info they might want or need to fully wrap their mind around the work.

Caption:
"abstract painting with various colors and shapes, and notice the blue stripe in the middle"
Description for the blind:
The painting is composed of rectangles, long and thin, all stacked horizontally except for the long verticals at the left and right edges of the painting - a large blue rectangle is nearly a squat square, taking up the lower portion of the painting from the bottom edge to just above center, stopping about 4 inches before the left edge of the canvas, and about 1 inch before the right edge of the canvas. Near it stops lay perfectly straight yet slightly uneven in black charcoal lines. Blue paint goes up to the black line, sometimes overlapping the line, sometimes touching the line, sometimes stopping right before the line. It leaves a gap where the pale yellow paint underneath shows through the blue. There is a blue stripe that is almost exactly the same width as the non-blue vertical rectangles on the left. This is blue vertical stripe is separated from the rest of the square by a bold charcoal line, again appearing and disappearing with the paint. To the right of the vertical blue stripe is one inch of pale pink. To the left of the blue squat square is a subdivided tall rectangle  The colors have been painted with very thin paint, and in various hues of one color per block, so the surface is not even, but modulated within the sometimes W-directional loose brushstrokes. The previous layers of color show through the top color as if it was tissue paper. This thin paint layering within straight rectangles - some edged with straight but uneven charcoal lines - continues throughout the painting.

Caption:
"3 d character design, male pop singer, vector art, digital art, portrait, 4 k, 8 k, sharp focus, smooth, music artist"
Description for the blind:
"""

prompt_recaption_request = \
"""
Good. Another work needs detailed captioning:
"{caption}"
Description for the blind:
"""

recaption_assistant_response_jss = \
"""
This striking portrait, collaborative effort of John Singer Sargent and J.C. Leyendecker. Center of the frame are two women, identical twins, standing side by side in a frontal view. The twins appear to be in their early thirties and wear clothing from the early 20th century.
Both women have long, auburn hair cascading down their shoulders, with curls framing their faces. Their expressions convey a sense of worry and weariness. Their eyes are a deep, thoughtful brown, and their eyebrows are slightly furrowed, giving them a concerned appearance. Their lips are soft and downturned, adding to the sense of anxiety.
The women's skin is pale and flawless, a common trait in Sargent's portraits, and their cheeks carry a hint of rosy color. They wear conservative, high-collared blouses with lace detailing, buttoned up to their necks. Their attire is elegant and modest, reflecting the fashion of the era.
The background of the painting is somewhat muted, allowing the focus to remain on the twin women. The lighting is soft but highlights their faces, casting subtle shadows that add depth to the composition.
Sargent and Leyendecker's collaborative masterpiece captures the essence of the twin women's lives, portraying their inner turmoil and concerns, while showcasing the timeless elegance and grace that these renowned artists are known for.
"""

recaption_assistant_response_pop_singer = \
"""
"Intricately detailed male pop singer character, 3D render. Psychonautic strobe stage lighting. The character is a vibrant and dynamic lead singer. His face is locked into a wickedly contorted grin. He holds a microphone in his hands, and is mid-jump doing the splits in the air above the stage as he screams into the mic. He exudes charisma and confidence. The character is wearing a rainbow pattern cross stiched in richly colored textures. Tight glistening leather pants, his eyes are striking.
The stage is filled with vivid, swirling lights that add a sense of movement and energy to the scene. Smoke curls up from the edges of the stage. The fellow band mates are pased to resume at this culminating moment, the chord seems to be held up wavering in the air, waiting for them to let it come crashing down. The exertion is visible from their sweat and heaving chests. 
High definition realistic 3D art, soft vector design and smooth digital shading, professional 3D render
"""

recaption_assistant_response_diebenkorn = \
"""
The painting is composed of rectangles that are long and thin, all stacked horizontally except for the long verticals at the left and right edges of the painting - the large blue rectangle is nearly a squat square, taking up the lower portion of the painting from the bottom edge to just above center, stopping about 4 inches before the left edge of the canvas, and about 1 inch before the right edge of the canvas. Near it stops lay perfectly straight yet slightly uneven in black charcoal lines. Blue paint goes up to the black line, sometimes overlapping the line, sometimes touching the line, sometimes stopping right before the line. It leaves a gap where the pale yellow paint underneath shows through the blue. There is a blue stripe that is almost exactly the same width as the non-blue vertical rectangles on the left. This is blue vertical stripe is separated from the rest of the square by a bold charcoal line, again appearing and disappearing with the paint. To the right of the vertical blue stripe is one inch of pale pink. To the left of the blue squat square is a subdivided tall rectangle  The colors have been painted with very thin paint, and in various hues of one color per block, so the surface is not even, but modulated within the sometimes W-directional loose brushstrokes. The previous layers of color show through the top color as if it was tissue paper. This thin paint layering within straight rectangles - some edged with straight but uneven charcoal lines - continues throughout the painting.
"""

def main(
    model_url = "TheBloke/Mistral-7B-OpenOrca-AWQ",
    request_batch_size=1024,
    max_n=50000,
    ):

    diffusion_db = load_dataset("poloclub/diffusiondb", "2m_text_only", split="train").shuffle(1000)
    max_n = min(max_n, len(diffusion_db))
    diffusion_db = iter(diffusion_db)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    print('loading model')
    model = LLM(model_url, quantization="awq", dtype="half", gpu_memory_utilization=0.90, swap_space=16, max_model_len=4096, )
    print('model loaded')
    
    output_data = []
    i_start = 0

    prog_bar = tqdm(total=max_n)

    used_prompts = set()

    while len(output_data) < max_n:
        chat_templates = []
        prompts = []
        while len(chat_templates) < request_batch_size:
            prompt = next(diffusion_db)['prompt']
            prompt = prompt.strip()

            if prompt in used_prompts:
                continue

            prompts.append(prompt)

            used_prompts.add(prompt)

            chat = [
                    {"role":"system", "content": prompt_sys},
                    {"role":"user", "content": prompt_recaption},
                    {"role":"assistant", "content": recaption_assistant_response_pop_singer},
                    {"role":"user", "content": prompt_recaption_request.format(caption=prompt)},
            ]

            chat_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            chat_templates.append(chat_template)

        outputs = model.generate(chat_templates, sampling_params)

        for output, prompt in zip(outputs, prompts):
            output_data.append(
                    dict(
                        prompt_original=prompt,
                        prompt_upscaled=output.outputs[0].text.strip().replace("\"", ""),
                        )
                    )

        prog_bar.update(request_batch_size)

    d = datasets.Dataset.from_list(output_data)
    os.makedirs("out/", exist_ok=True)
    d.save_to_disk(f"out/dataset{max_n:08}")

    prog_bar.close()

if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
