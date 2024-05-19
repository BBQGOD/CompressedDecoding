#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
python para_decode_alpaca_gpt4_prefiltering.py \
    --model google/gemma-7b-it \
    --assist_model google/gemma-2b-it \
    --max_new_tokens 96 --top_k 100 \
    --data_path ../GPT-4-LLM/data/ \
    --data_filename alpaca_gpt4_data.json alpaca_gpt4_data_zh.json \
    --rand_seed 17 --lang en --dataset_size 10 \
    --batch_size 3 \
    --assist_token_conf_threshold 0.95 \
    --compressed_token_conf_threshold 5 \
    --overlap_ratio_threshold 0.085 \
    > tmp_prefiltering.log
