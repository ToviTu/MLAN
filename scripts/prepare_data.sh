#! /bin/bash

mkdir -p playground/data

# Pretraining data
wget -O playground/data/blip_laion_cc_sbu_558k.json https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
wget -O playground/data/images.zip https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip
unzip playground/data/images.zip -d playground/data/images
rm playground/data/images.zip

# Fine-tuning data
# The actual links will be updated after the review period.
wget -O playground/data/MLAN_80k.json https://huggingface.co/datasets/account/MLAN/resolve/main/MLAN_80k.json
wget -O playground/data/MLAN_v_88l_80k.json https://huggingface.co/datasets/account/MLAN/resolve/main/MLAN_v_88l_80k.json
wget -O playground/data/MLAN_v_50l_80k.json https://huggingface.co/datasets/account/MLAN/resolve/main/MLAN_v_50l_80k.json

wget -O playground/data/images_mlan_v.zip https://huggingface.co/datasets/account/MLAN/resolve/main/images_mlan_v.zip
unzip playground/data/images_mlan_v.zip -d playground/data/images_mlan_v
rm playground/data/images_mlan_v.zipc
