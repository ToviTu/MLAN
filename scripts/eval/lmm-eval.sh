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

FOLDER=vl_eval
TASK="realworldqa_llava_plain,ai2d_llava_plain,pope_llava_plain,gqa_llava_plain,sciq_llava_plain"

# Run command
echo "Running evaluation on model: $MODEL with task: $TASK"

python -m accelerate.commands.launch \
        --num_processes=1 \
        -m lmms_eval \
        --model llava_plain \
        --model_args pretrained=$MODEL \
        --include_path ${WORKING_DIR}/scripts/eval/custom/ \
        --tasks $TASK \
        --batch_size 1 \
        --output_path ${WORKING_DIR}/playground/${FOLDER}
