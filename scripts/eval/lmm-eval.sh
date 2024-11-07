#! /bin/bash

WORKING_DIR=$PWD
STORAGE_DIR=/ib-scratch/chenguang03/vision_share

# Pretrain Models
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-pretrain-fb
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-pretrain-fb

# CoT Finetunded Models
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit

# Flan Finetuned Models
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit-flan
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan2
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan2
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan


# Mix Finetuned Models
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-flan
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-mix-flan
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-cot
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-mix-cot

# Mix Scaling Finetuned Models
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-125l
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-250l
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-375l
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-500l
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-625l
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-750l
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-875l

# Num Tokens Scaling VIT Models
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-short
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-mid 

# Num Instances Scaling Models
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-25
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-mid
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-75
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan-25
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan-50
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan-75

# Low Resolution Variants
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-pretrain-lowres
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan-lowres
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-lowres

# Cap Token Budget
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-long
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-flan-short

# Stability Test
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan2-seed42
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan2-seed43
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan2-seed44

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-seed42
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-seed43
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-seed44

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan-seed42
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan-seed43
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan-seed44

#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan2-seed42
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan2-seed43
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan2-seed44

#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan-seed42
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan-seed43
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan-seed44

#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit-flan-seed42
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit-flan-seed43
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit-flan-seed44

#MODELS=/storage1/chenguangwang/Active/t.tovi/models/llava-llama2-7b-mix-cot
#MODELS=/storage1/chenguangwang/Active/t.tovi/models/llava-vicuna-7b-mix-cot

# LLaVA mimic
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-flan-6l
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-mix-flan-6l

# Cambrian-1 mimic
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-mix-flan-250l

# Our mixture
MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-mix-flan-875l

################################################################
# Pretrain 
#TASKS="realworldqa_llama_plain,ai2d_llama_plain,pope_llama_plain,gqa_llama_plain"

# Llava Plain
FOLDER=extra
TASKS="realworldqa_llava_plain,ai2d_llava_plain,pope_llava_plain,gqa_llava_plain,sciq_llava_plain"

python -m accelerate.commands.launch \
        --num_processes=1 \
        -m lmms_eval \
        --model llava_plain \
        --model_args pretrained=$MODELS \
        --include_path ${WORKING_DIR}/scripts/eval/custom/ \
        --tasks $TASKS \
        --batch_size 1 \
        --output_path ${WORKING_DIR}/playground/${FOLDER} \
        #--limit 100 \
        #--verbosity DEBUG

