# MLAN: Language-Based Instruction Tuning Improves Zero-Shot Generalization of Multimodal Large Language Models

[MLAN arxiv]() / [MLAN Huggingface](https://huggingface.co/datasets/ToviTu/MLAN) 

## 💡 Introduction

MLAN explores a language-heavy approach in visual instruction-tuning, 
which enables fine-tuned language model to excel in both vision-language and language-only tasks.
Our empirical experiments find that: 
(1) fine-tuning on pure visual instruction data does not yield the best performance on unseen benchmarks.
(2) language instruction-following abilities can be efficiently transfered to the vision domain by substituting only ~10% of the language data with visual data.
A key advantage of primarily using language-only data is the significantly decreased training cost 
thanks to the vastly reduced use of token-rich image inputs.

## ⚒️ Installation

Our training code is built upon the [LLaVA repo](https://github.com/haotian-liu/LLaVA).

1. Clone this repository
```
git clone https://github.com/ToviTu/MLAN.git
cd MLAN
```

2. Install the training packages
```
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

3. Install our modified evaluation packages
```
pip install git+https://github.com/ZhuohaoNi/lm_eval.git
pip install git+https://github.com/ToviTu/lmms-eval.git@llava_plain
```

## 📖 Data Preperation

The text and image data can be accessed directly through our Huggingface repository. You should download them into the `playground/data` folder. 

[MLAN_80k](https://huggingface.co/datasets/ToviTu/MLAN/resolve/main/MLAN_80k.json): contains 80k **language-only** instruction-tuning data collected from public datasets.

[MLAN_v_80k](https://huggingface.co/datasets/ToviTu/MLAN/resolve/main/MLAN_v_80k.json): contains 70k **language-only** and 10k **vision-language** instruction following data.

[images_mlan_v](https://huggingface.co/datasets/ToviTu/MLAN/resolve/main/images_mlan_v.zip): contains the corresponding images for MLAN_v_80k.

## 🏋️‍♂️ Train

MLAN training consists of 2 phases: 
(1) feature alignment: we use the LLaVA-CC3M-Pretrain-595K to make the visual encoder outputs comptatible with the base language model.
(2) supervised fine-tuning: use our MLAN_80k or MLAN_v_80k to instruction-tune the language model and the projector.

### Pretraining

Pretraining takes around 3.5 hours for a 7B model. Our experiments are conducted on single nodes with 8xA6000 (48G) or 4xA100 (80G). Please note that the global batch size (num_gpus * per_device_batchsize * gradient_accumulation_steps) needs to be kept the same.

```
bash scripts/pretrain.sh
```

### Instruction Tuning

Thanks to the reduced usage of image inputs, MLAN takes under 1 hour and MLAN_v takes under 2 hours on 8xA6000. 

```
bash scripts/finetune.sh
```

## 📝 Evaluation

Our testing environments are built upon lm-eval and lmms-eval platforms, for language-only and vision-language tasks respectively. We use customized answer parsers to extract short answers. Take a look at the task definitions written in `scripts/eval/custom` directory. 

To run evaluation, use the following commands
```
bash scripts/eval/lm-eval
bash scripts/eval/lmm-eval
```

## Citations
```
@article{name_pending,
  title={title_pending},
  author={author_pending},
  journal={arXiv preprint arXiv:2406.04325},
  year={2024}
}
```

# Acknowledgement
1. [LLaVA](https://github.com/haotian-liu/LLaVA): our code is built upon their wonderful scripts.
2. [LM-EVAL](https://github.com/EleutherAI/lm-evaluation-harness): we customized their pipeline for language evaluation.
3. [LMMS-EVAL](https://github.com/EvolvingLMMs-Lab/lmms-eval): we customized their pipeline for vision evaluation. 
