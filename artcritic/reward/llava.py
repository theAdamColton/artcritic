from typing import List
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling,DataCollatorForSeq2Seq, AutoTokenizer, DataCollatorWithPadding
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import torch
from llava import LlavaLlamaForCausalLM
import torchvision
import llava.mm_utils as llava_utils
from llava.conversation import conv_templates as llava_conv_templates

from artcritic.reward.reward import Reward


# exactly the same as the original https://github.com/haotian-liu/LLaVA/blob/785f766fcddc86ffeaa62cd51cf7834a11c04e6d/llava/model/multimodal_encoder/clip_encoder.py#L40C10-L40C10
# but with the @no_grad() removed
def _clip_vision_tower_forward(self_vision_tower, images):
    if type(images) is list:
        image_features = []
        for image in images:
            image_forward_out = self_vision_tower.vision_tower(image.to(device=self_vision_tower.device, dtype=self_vision_tower.dtype).unsqueeze(0), output_hidden_states=True)
            image_feature = self_vision_tower.feature_select(image_forward_out).to(image.dtype)
            image_features.append(image_feature)
    else:
        image_forward_outs = self_vision_tower.vision_tower(images.to(device=self_vision_tower.device, dtype=self_vision_tower.dtype), output_hidden_states=True)
        image_features = self_vision_tower.feature_select(image_forward_outs).to(images.dtype)

    print("_clip_vision_tower_forward called")
    return image_features

class LlavaReward(Reward):
    def __init__(self,
        inference_dtype=torch.float16,
                 device='cuda',
        model_path = "liuhaotian/llava-v1.5-13b",
                 max_seq_len: int = 256,
                 torch_compile=False,
                     ):
        print("LOADING LLAVA MODEL")

        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=inference_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                llm_int8_enable_fp32_cpu_offload=True,
                load_in_8bit_fp32_cpu_offload=True,
            )


        # put everything on gpu, but specify max_memory, which should automate offloading
        # TODO try to have auto device map
        device_map = {"":0,}
        model:LlavaLlamaForCausalLM = LlavaLlamaForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, torch_dtype=torch.float16, device_map=device_map, max_memory={0:'6GB', "cpu": "24GB"},)
        #model:LlavaLlamaForCausalLM = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=inference_dtype, device_map="auto", max_memory={0:'1GB', "cpu": "24GB"}, offload_folder="/tmp/huggingface/")

        model = model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model.get_vision_tower().load_model()
        self.image_processor = model.get_vision_tower().image_processor

        for mod in model.modules():
            mod.requires_grad_(False)

        model.gradient_checkpointing_enable()
        model.model.gradient_checkpointing_enable()

        # needs to monkey patch the vision tower so it doesn't have the @no_grad decorator
        model.model.vision_tower.forward = lambda images: _clip_vision_tower_forward(model.model.vision_tower, images)

        if torch_compile:
            model = torch.compile(model)

        print("DONE LOADING LLAVA MODEL")

        model_name = llava_utils.get_model_name_from_path(model_path)
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        self.max_seq_len = max_seq_len

        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.tokenizer = tokenizer
        self.conv_template = llava_conv_templates[conv_mode].copy()
        self.captioning_prompt = "Provide a detailed caption for the following image. Describe all of the objects in the image in painstaking detail."
        self.device = device
        self.dtype = inference_dtype

    def _tokenize(self, assistant_output:str):
        inp = DEFAULT_IMAGE_TOKEN + '\n' + self.captioning_prompt
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], assistant_output)
        prompt = conv.get_prompt()
        # seperately tokenizes the user's caption, and the assistant's answer
        sep = conv.sep + conv.roles[1] + ": "

        user_prompt, assistant_response = prompt.split(sep)
        # adds back the sep
        user_prompt = user_prompt + sep

        input_ids_user = llava_utils.tokenizer_image_token(user_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        # gets rid of EOS token
        input_ids_user = input_ids_user[:-1]

        # no images in the assistant_response, so there is no need to use the
        # special tokenizer_image_token
        input_ids_assistant = self.tokenizer(assistant_response, return_tensors='pt', add_special_tokens=False).input_ids[0]

        labels_user = torch.LongTensor([-100]*len(input_ids_user))
        labels_assistant = input_ids_assistant.clone()

        input_ids = torch.concat([input_ids_user, input_ids_assistant])
        labels = torch.concat([labels_user, labels_assistant])

        return input_ids, labels

    def __call__(self, pixel_values: torch.Tensor, captions: List[str]):
        pixel_values = self._process_image_pixels(pixel_values).to(self.device).to(self.dtype)

        batched_input_ids = []
        batched_labels = []
        for cap in captions:
            input_ids, targets = self._tokenize(cap)
            batched_input_ids.append(input_ids)
            batched_labels.append(targets)

        # pads
        collator = DataCollatorForSeq2Seq(self.tokenizer, None, padding=True, )
        features = [{"labels":torch.LongTensor(labels), "input_ids": torch.LongTensor(input_ids)} for input_ids, labels in zip(batched_input_ids, batched_labels)]

        model_inputs = collator(features)

        # clips to max seq length
        model_inputs = {k:v[:,:self.max_seq_len] for k,v in model_inputs.items()}

        model_inputs = {k:v.to(self.device) for k,v in model_inputs.items()}

        outputs = self.model(images=pixel_values, return_dict=True, **model_inputs)

        return outputs.loss, -outputs.loss


    def _process_image_pixels(self,x: torch.Tensor):
        """
        Does everything that the image_processor (CLIPImageProcessor)
        does, but using native pytorch differentiable operations
        """
        #x = ((x / 2) + 0.5).clamp(0, 1) 
        to_h, to_w = self.image_processor.crop_size['height'], self.image_processor.crop_size['width']
        x= torchvision.transforms.Resize((to_h, to_w), antialias=True)(x)

        # normalizes
        image_mean = self.image_processor.image_mean
        image_std = self.image_processor.image_std

        x = torchvision.transforms.functional.normalize(x, image_mean, image_std)

        return x

