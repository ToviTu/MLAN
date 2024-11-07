#! /bin/bash

WORKING_DIR=$PWD
STORAGE_DIR=/ib-scratch/chenguang03/vision_share

# MODELS
#MODELS=meta-llama/Llama-2-7b-hf
#MODELS=lmsys/vicuna-7b-v1.5
#MODELS=liuhaotian/llava-v1.5-7b-hf

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-pretrain-fb
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-pretrain-fb

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit-flan

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan2
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan2

#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-125l
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-250l
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-375l
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-625l
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-750l
#MODELS=${STORAGE_DIR}/modelsv3/llava-llama2-7b-mix-875l

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-flan-short
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan-long

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan2-seed42
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan2-seed42-test
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

# LLaVA mimic
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-mix-flan-6l
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-mix-flan-6l

# Cambrian-1 mimic
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-mix-flan-250l

# Our mixture
MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-mix-flan-875l

#---------------------------------------------------------
# Tasks (Specify one in TASK below)
#---------------------------------------------------------
# arc_easy - arc_easy - arc_easy_prompt_em
# arc_challenge - arc_challenge - arc_challenge_prompt_em
# commonsenseqa - commonsense_qa_loglikelihood - commonsense_qa_em
# OpenBookQA - openbookqa - openbookqa_em
# BoolQ - boolq_log - boolq_em
# RACE - race - race_em
# hellaswag - hellaswag - hellaswag_em
# cosmosqa - cosmosqa - cosmosqa_em
# SQuADv2 - squadv2
#---------------------------------------------------------

TASK="race_em,openbookqa_em,boolq_em,hellaswag_em"  

# Run command
echo "Running evaluation on model: $PRETRAINED_MODEL with task: $TASK"

lm_eval \
    --model hf \
    --model_args pretrained=$MODEL \
    --include_path ./ \
    --tasks $TASK \
    --device cuda:0 \
    --batch_size auto \
    --gen_kwargs max_new_tokens=20,max_length=None,do_sample=False\
    --num_fewshot 0 \
    --log_samples \
    --output_path ${WORKING_DIR}/playground/test \