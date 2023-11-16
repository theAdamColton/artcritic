import random
from torch.nn import functional as F
from typing import Dict, List
from llava.model.builder import load_pretrained_model
from torchvision.transforms import InterpolationMode
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling,DataCollatorForSeq2Seq, AutoTokenizer, DataCollatorWithPadding
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import torch
from llava import LlavaLlamaForCausalLM
import torchvision
import llava.mm_utils as llava_utils
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
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

    return image_features


class LlavaReward(Reward):
    def __init__(self,
        inference_dtype=torch.float16,
                 device='cuda',
                 model_path ="teowu/llava_v1.5_7b_qinstruct_preview_v0.1",
                 max_seq_len: int = 256,
                 torch_compile=False,
                     ):
        print("LOADING LLAVA MODEL")


        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path, model_base = None, model_name=model_path.split("/")[-1], load_4bit=True)

        model = model.eval()

        self.model = model
        self.image_processor = image_processor


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

        self.max_seq_len = min(max_seq_len, context_len)

        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.tokenizer = tokenizer
        self.conv_template = llava_conv_templates[conv_mode].copy()
        self.captioning_prompt = "Describe this image in detail. It is a work in an art installation, and you are a art critic. Describe the composition, colors, and salient items that are portrayed."
        self.device = device
        self.dtype = inference_dtype

        self.collator = DataCollatorForSeq2Seq(self.tokenizer, None, padding=True, pad_to_multiple_of=64)

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

    def __call__(self, pixel_values: torch.Tensor, batched_prompt_d):
        pixel_values = self._process_image_pixels(pixel_values).to(self.device).to(self.dtype)

        batched_input_ids = []
        batched_labels = []
        for cap in detailed_captions:
            input_ids, targets = self._tokenize(cap)
            batched_input_ids.append(input_ids)
            batched_labels.append(targets)

        # pads
        features = [{"labels":torch.LongTensor(labels), "input_ids": torch.LongTensor(input_ids)} for input_ids, labels in zip(batched_input_ids, batched_labels)]

        model_inputs = self.collator(features)

        # clips to max seq length
        model_inputs = {k:v[:,:self.max_seq_len] for k,v in model_inputs.items()}

        model_inputs = {k:v.to(self.device) for k,v in model_inputs.items()}

        outputs = self.model(images=pixel_values, return_dict=True, **model_inputs)

        loss = self._get_loss(outputs)

        return loss, -loss

    def _get_loss(self, outputs):

        loss = outputs.loss

        return loss


    def _process_image_pixels(self,x: torch.Tensor):
        """
        Does everything that the image_processor (CLIPImageProcessor)
        does, but using native pytorch differentiable operations
        """
        #x = ((x / 2) + 0.5).clamp(0, 1) 
        to_h, to_w = self.image_processor.crop_size['height'], self.image_processor.crop_size['width']
        x = F.interpolate(x, (to_h, to_w), antialias=False, mode='nearest')

        # normalizes
        mean = self.image_processor.image_mean
        std = self.image_processor.image_std

        x = torchvision.transforms.Normalize(mean=mean, std=std)(x)

        return x

class LlavaRewardSimpleRater(LlavaReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, pixel_values: torch.Tensor, batched_prompt_d):
        pixel_values = self._process_image_pixels(pixel_values).to(self.device).to(self.dtype)

        b = pixel_values.shape[0]

        batched_input_ids = []
        for prompt_d in batched_prompt_d:
            cap = prompt_d['prompt']
            inp = f"This AI generated image is generated from the prompt, '{cap[:150]}'. The AI generator might not have completely adhered to all of the details in the prompt, in which case it is a poor image. If the quality is good, and the generated image correctly corresponds with the user's instruction prompt, then the rating should be \"Good\"."
            inp = inp + "\nRate the quality of the image as either \"Poor\" or \"Good\"."
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(self.device)
            batched_input_ids.append({'input_ids':input_ids})

        model_inputs = self.collator(batched_input_ids)
        model_inputs = {k:v.to(self.device) for k,v in model_inputs.items()}

        outputs = self.model(images=pixel_values, return_dict=True, **model_inputs)

        good_token = self.tokenizer("Good", add_special_tokens=False)['input_ids']
        good_token = good_token[0]

        poor_token = self.tokenizer("Poor", add_special_tokens=False)['input_ids']
        poor_token = poor_token[0]

        loss = 0.0
        for batch_i, input_ids in enumerate(batched_input_ids):
            input_len = len(input_ids['input_ids'])

            # LLaVA adds visual tokens corresponding to the number of ViT
            # patches subtracted by 1

            # so the last actual last token the model produces per input element is
            # different from the input sequence length

            input_len += self.model.model.vision_tower.num_patches -1


            # we want the good token to be really positive
            # and we want the bad token to be really negative
            good_logit = outputs.logits[batch_i, input_len-1, poor_token]
            poor_logit = outputs.logits[batch_i, input_len-1, good_token]
            # c.3 from https://arxiv.org/abs/2311.06783 q-instruct paper
            reward = good_logit / (good_logit + poor_logit)
            loss = loss - reward

        loss = loss / b

        return loss, -loss



class LlavaRewardSimpleRater(LlavaReward):
    def __call__(self, pixel_values: torch.Tensor, batched_prompt_d: List[Dict],):
        pixel_values = self._process_image_pixels(pixel_values).to(self.device).to(self.dtype)

        batched_input_ids = []
        answer_ids = []
        for prompt_d in batched_prompt_d:
            questions = prompt_d['prompt_qa_questions']
            answers = prompt_d['prompt_qa_answers']
            all_options = prompt_d['prompt_qa_options']


            # picks a random question
            len_q = len(questions)
            q_i = random.randint(0, len_q-1)

            question, answer, options = questions[q_i], answers[q_i], all_options[q_i]

            inp = f"{question}\nAnswer with the option’s letter from the given choices directly.\n{options}"
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(self.device)
            batched_input_ids.append({'input_ids':input_ids})
            answer_id = self.tokenizer(answer, add_special_tokens=False,)['input_ids'][0]
            answer_ids.append(answer_id)

        model_inputs = self.collator(batched_input_ids)
        model_inputs = {k:v.to(self.device) for k,v in model_inputs.items()}
        outputs = self.model(images=pixel_values, return_dict=True, **model_inputs)
        logits = outputs.logits

        loss = 0.0
        for i, (input_ids, answer_id) in enumerate(zip(batched_input_ids, answer_ids)):
            last_token_i = len(input_ids['input_ids'])

            # LLaVA adds visual tokens corresponding to the number of ViT
            # patches subtracted by 1

            # so the last actual last token the model produces per input element is
            # different from the input sequence length

            last_token_i += self.model.model.vision_tower.num_patches -1 - 1

            # we want the answer token to be really positive
            # and we want the bad token to be really negative
            reward = logits[i, last_token_i, answer_id]
            loss = loss - reward

        loss = loss / pixel_values.shape[0]
        return loss, -loss

