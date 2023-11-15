from llava.model.builder import load_pretrained_model
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from llava.conversation import conv_templates

from transformers import TextStreamer
from PIL import Image
import torch

model_path = "teowu/llava_v1.5_7b_qinstruct_preview_v0.1"
model_name = get_model_name_from_path(model_path)

device='cuda'

tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name, load_4bit=True, device=device)

image = Image.open("./images/ladyai.png")

pixel_values = process_images([image], image_processor, model.config)
if type(pixel_values) is list:
    pixel_values = [image.to(model.device, dtype=torch.float16) for image in pixel_values]
else:
    pixel_values = pixel_values.to(model.device, dtype=torch.float16)

inp = "Describe and evaluate the quality of the image. Also give an analysis of the content of the image, as if you were describing it to a blind person at an art exhibit."
inp = "Rate the quality of the image as either amazing, good, bad or terrible."
inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

if 'llama-2' in model_name.lower():
    conv_mode = "llava_llama_2"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


with torch.inference_mode():
    output_ids = model.generate(
    input_ids,
    images=pixel_values,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=512,
    streamer=streamer,
    use_cache=True,
    )

outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
print("outputs:",outputs)
