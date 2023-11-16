import ml_collections
import os


def general():
    config = ml_collections.ConfigDict()
    return config


def aesthetic():
    config = general()
    config.num_epochs = 200
    config.prompt_fn = "simple_animals"

    config.eval_prompt_fn = "eval_simple_animals"

    config.reward_fn = "aesthetic"  # CLIP or imagenet or .... or ..
    config.train.max_grad_norm = 5.0
    config.train.loss_coeff = 0.01
    config.train.learning_rate = 1e-3
    config.max_vis_images = 4
    config.train.adam_weight_decay = 0.1

    config.save_freq = 1
    config.num_epochs = 7
    config.num_checkpoint_limit = 14
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (0, 50)
    config.trunc_backprop_timestep = 40
    config.truncated_backprop = True
    config = set_config_batch(
        config, total_samples_per_epoch=256, total_batch_size=128, per_gpu_capacity=4
    )
    return config


def llava():
    config = general()
    config.num_epochs = 20
    config.prompt_fn = "diffusiondb_upscaled"
    config.eval_prompt_fn = "diffusiondb_upscaled"
    config.reward_fn = "llava"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    config.train.max_grad_norm = 5.0
    config.train.loss_coeff = 0.01
    config.train.learning_rate = 1e-3
    config.train.adam_weight_decay = 0.1
    config.train.gradient_accumulation_steps = 2

    config.visualize_train = True

    config.lora_rank = 1
    config.train.use_8bit_adam = True

    config.steps = 20
    config.trunc_backprop_timestep = 15
    config.truncated_backprop = True
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (0, 20)
    config = set_config_batch(
        config, total_samples_per_epoch=16, total_batch_size=8, per_gpu_capacity=1
    )
    return config


def hps():
    config = general()
    config.num_epochs = 200
    config.prompt_fn = "hps_v2_all"
    config.eval_prompt_fn = "eval_hps_v2_all"
    config.reward_fn = "hps"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    config.train.max_grad_norm = 5.0
    config.train.loss_coeff = 0.01
    config.train.learning_rate = 1e-3
    config.train.adam_weight_decay = 0.1

    config.trunc_backprop_timestep = 40
    config.truncated_backprop = True
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (0, 50)
    config = set_config_batch(
        config, total_samples_per_epoch=256, total_batch_size=128, per_gpu_capacity=2
    )
    return config


def evaluate_soup():
    config = general()
    config.only_eval = True

    config.reward_fn = "aesthetic"
    config.prompt_fn = "simple_animals"
    config.debug = False
    config.same_evaluation = True
    config.max_vis_images = 10

    config.soup_inference = True
    config.resume_from = "<CHECKPOINT_NAME>"
    # Use checkpoint name for resume_from_2 as stablediffusion to interpolate between stable diffusion and resume_from
    config.resume_from_2 = "<CHECKPOINT_NAME>"
    config.mixing_coef_1 = 0.0
    config = set_config_batch(
        config, total_samples_per_epoch=256, total_batch_size=128, per_gpu_capacity=4
    )
    return config


def evaluate():
    config = general()
    config.reward_fn = "aesthetic"
    config.prompt_fn = "eval_simple_animals"
    config.only_eval = True
    config.same_evaluation = True
    config.max_vis_images = 10
    config = set_config_batch(
        config, total_samples_per_epoch=256, total_batch_size=128, per_gpu_capacity=4
    )
    return config


def get_config(name):
    return globals()[name]()
