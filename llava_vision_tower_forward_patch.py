# exactly the same as the original https://github.com/haotian-liu/LLaVA/blob/785f766fcddc86ffeaa62cd51cf7834a11c04e6d/llava/model/multimodal_encoder/clip_encoder.py#L40C10-L40C10
# but with the @no_grad() removed
def forward(self_vision_tower, images):
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
