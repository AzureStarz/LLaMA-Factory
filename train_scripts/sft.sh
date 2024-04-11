#!/bin/bash
# constant directory
base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM
code_path=${base_path}/LLaMA-Factory
dataset_dir=${code_path}/data
# pretrained LLM model path
model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b
# template used in sft
template=alpaca
# output model name
output_model_name=sft_llama2-7b
output_model=${base_path}/LLM-SFT_exp_output/${output_model_name}
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
# dataset used in sft
dataset=alpaca_en
# dataset=alpaca_en
# Deepspeed params
random_num=$(( RANDOM % 5001 ))
master_port=$(( random_num + 25000 ))

deepspeed --master_port ${master_port} ${code_path}/src/train_bash.py \
    --deepspeed ${code_path}/ds_configs/stage2_no_offloading.conf \
    --stage sft \
    --finetuning_type full \
    --do_train \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --model_name_or_path ${model_path} \
    --dataset_dir ${dataset_dir} \
    --dataset ${dataset} \
    --cutoff_len 512 \
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
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --flash_attn true \
    --bf16 true \
    --tf32 true \
    --ddp_find_unused_parameters false \
    --seed 42 2>&1 | tee $(dirname "${output_model}")/${output_model_name}.log