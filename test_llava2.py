from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
model_path = "teowu/llava_v1.5_7b_qinstruct_preview_v0.1" 
prompt = "Rate the quality of the image. Think step by step."
image_file = "./images/ladyai.png"
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature":0.0,
    "top_p": 0,
    "num_beams": 1,
    "max_new_tokens": 256,
})()
eval_model(args)
