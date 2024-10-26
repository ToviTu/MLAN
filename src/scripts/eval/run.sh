#!/bin/bash

#---------------------------------------------------------
# Models (Specify one in MODELS below)
#---------------------------------------------------------
# Pretrained Models
# LLaMA2-7B-pretrain
# Vicuna-1.5-7B-pretrain

# Main Result Models
# LLaMA2-7B-lit
# LLaMA2-7B-vit
# Vicuna-1.5-7B-lit
# Vicuna-1.5-7B-vit

# Flan Models
# llava-llama2-7b-lit-flan
# llava-llama2-7b-vit-flan
# llava-llama2-7b-mix-flan
# llava-vicuna-7b-lit-flan
# llava-vicuna-7b-vit-flan
# llava-vicuna-7b-mix-flan

# CoT Models (Chain of Thought)
# LLaMA2-7B-lit (CoT)
# LLaMA2-7B-vit (CoT)
# Vicuna-1.5-7B-lit (CoT)
# Vicuna-1.5-7B-vit (CoT)

# Mixture Scaling Models
# llama2-7b-mix-flan-000l
# llama2-7b-mix-flan-125l
# llama2-7b-mix-flan-250l
# llama2-7b-mix-flan-375l
# llama2-7b-mix-flan-500l
# llama2-7b-mix-flan-625l
# llama2-7b-mix-flan-750l
# llama2-7b-mix-flan-875l
# llama2-7b-mix-flan-100l
# llama2-7b-mix-flan-short

# Low Res Ablation Models
# llava-llama2-7b-pretrain-lowres
# llava-llama2-7b-lit-flan-lowres
# llava-llama2-7b-vit-flan-lowres

# Instance Scaling (VIT) Models
# llava-llama2-7b-vit-flan-0
# llava-llama2-7b-vit-flan-25
# llava-llama2-7b-vit-flan-50
# llava-llama2-7b-vit-flan-75
# llava-llama2-7b-vit-flan-100
# llava-llama2-7b-vit-flan-400

# Instance Scaling (LIT) Models
# llava-llama2-7b-lit-flan-0
# llava-llama2-7b-lit-flan-25
# llava-llama2-7b-lit-flan-50
# llava-llama2-7b-lit-flan-75
# llava-llama2-7b-lit-flan-100

# Cap Token Budget Models
# llavallama2-7b-vit-flan
# llava-llama2-7b-lit-flan
# llava-llama2-7b-vit-flan-short (20%)
# llava-llama2-7b-vit-flan-400
# llama2-7b-mix-flan-short

# Stability Analysis Models
# llama2-7b-lit-flan-seed42
# llama2-7b-lit-flan-seed43
# llama2-7b-lit-flan-seed44
# llama2-7b-vit-flan2-seed42
# llama2-7b-vit-flan2-seed43
# llama2-7b-vit-flan2-seed44
# llama2-7b-vit-flan-seed42
# llama2-7b-vit-flan-seed43
# llama2-7b-vit-flan-seed44
# vicuna-7b-lit-flan-seed42
# vicuna-7b-lit-flan-seed43
# vicuna-7b-lit-flan-seed44
# vicuna-7b-vit-flan2-seed42
# vicuna-7b-vit-flan2-seed43
# vicuna-7b-vit-flan2-seed44
# vicuna-7b-vit-flan-seed42
# vicuna-7b-vit-flan-seed43
# vicuna-7b-vit-flan-seed44
#---------------------------------------------------------

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



MODELS=${STORAGE_DIR}/models/llava-llama2-7b-pretrain-fb
TASK="arc_easy"  

# Run command
echo "Running evaluation on model: $PRETRAINED_MODEL with task: $TASK"

lm_eval \
    --model hf \
    --model_args pretrained=$PRETRAINED_MODEL \
    --include_path ./ \
    --tasks $TASK \
    --device cuda:0 \
    --batch_size 32 \
    --gen_kwargs max_new_tokens=20,max_length=None,do_sample=False\
    --num_fewshot 0 \
    --log_samples \
    --output_path ${WORKING_DIR}/playground/test \