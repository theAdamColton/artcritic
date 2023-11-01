#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=u1377031@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j


source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

conda info

CACHE_DIR=/scratch/general/vast/u1377031/huggingface_cache
mkdir -p ${CACHE_DIR}
export TRANSFORMER_CACHE=${CACHE_DIR}
export HF_DATASETS_CACHE=${CACHE_DIR}
export HF_HOME=${CACHE_DIR}

OUT_DIR=./out/
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES=0 accelerate launch sd_ft_reward_backprop.py --config config/align_prop.py:llava
