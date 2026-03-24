#!/bin/bash

export TOKENIZERS_PARALLELISM=false

export DEBUG_MODE=true  
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1
export MAIN_PROCESS_PORT=29508
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
REASONER_MODEL="/mnt/nfs_project_a/xinyi/models/Qwen3-VL-8B-Instruct"
# WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="/mnt/nfs_project_a/xinyi/models/Qwen3-VL-8B-Instruct"
# TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="/mnt/nfs_project_a/xinyi/models/Qwen3-VL-8B-Instruct"
TRIGGER_ACTIVE=False

DATASET_NAME="kodcode"

MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=1
PROMPT_LATENTS_LEN=4
INFERENCE_LATENTS_LEN=4

BATCH_SIZE=4

LOAD_MODEL_PATH="/mnt/nfs_project_a/xinyi/personal_memory/MemGen/.cache/train/kodcode/Qwen3-VL-8B-Instruct/pn=1_pl=4_in=1_il=4_20260323-202148/model"
# LOAD_MODEL_PATH=null

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main_auto.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active ${TRIGGER_ACTIVE} \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \