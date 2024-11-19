#! /bin/bash

WORKING_DIR=$PWD
# STORAGE_DIR=$PWD

# Local Models

# MODEL=${STORAGE_DIR}/models/llava-mlan-vicuna-7b
# MODEL=${STORAGE_DIR}/models/llava-mlan-v-llama2-7b

# HuggingFace Checkpoints

# MODEL=ToviTu/llava-mlan-vicuna-7b
# MODEL=ToviTu/llava-mlan-llama2-7b
# MODEL=ToviTu/llava-mlan-v-vicuna-7b
# MODEL=ToviTu/llava-mlan-v-llama2-7b

#---------------------------------------------------------
# Tasks (Specify one or all in TASK below)
#---------------------------------------------------------

FOLDER=l_eval
TASK="race_em,openbookqa_em,boolq_em,hellaswag_em"  

# Run command
echo "Running evaluation on model: $MODEL with task: $TASK"

lm_eval \
    --model hf \
    --model_args pretrained=$MODEL \
    --include_path ./custom \
    --tasks $TASK \
    --device cuda:0 \
    --batch_size auto \
    --gen_kwargs max_new_tokens=20,max_length=None,do_sample=False\
    --num_fewshot 0 \
    --log_samples \
    --output_path ${WORKING_DIR}/playground/${FOLDER} \
