#!/bin/bash

base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory
model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b
template=llama2
output_model=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_exp_output/bactrian-crosslingual_llama2-7b
dataset_dir=${base_path}/data
dataset=bactrian-crosslingual
# dataset=alpaca_en

if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi

# Deepspeed params
random_num=$(( RANDOM % 5001 ))
master_port=$(( random_num + 25000 ))

deepspeed --master_port ${master_port} ${base_path}/src/train_bash.py \
    --deepspeed ${base_path}/ds_configs/stage2_no_offloading.conf \
    --stage sft \
    --finetuning_type lora \
    --lora_dropout 0 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_target q_proj,v_proj \
    --do_train \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --model_name_or_path ${model_path} \
    --dataset_dir ${dataset_dir} \
    --dataset ${dataset} \
    --template ${template} \
    --output_dir ${output_model} \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 5 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --flash_attn \
    --bf16 true \
    --tf32 true \
    --ddp_find_unused_parameters false \
    --seed 42 2>&1 | tee $(dirname "${output_model}")/sft_lora_${dataset}.log