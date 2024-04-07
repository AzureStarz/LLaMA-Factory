#!/bin/bash

base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory
model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b
lora_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_exp_output/alpaca_en_llama2-7b

template=llama2
dataset_dir=${base_path}/data
dataset=alpaca_gpt4_en

output_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/alpaca_en_llama2-7b

if [ ! -d ${output_dir} ];then
    mkdir ${output_dir}
fi

for lang in ${langs[*]}

CUDA_VISIBLE_DEVICES=0 python ${base_path}/src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ${model_path} \
    --adapter_name_or_path ${lora_path} \
    --dataset_dir ${dataset_dir} \
    --dataset ${dataset} \
    --template ${template} \
    --finetuning_type lora \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 1 \
    --max_samples 100 \
    --predict_with_generate \
    --fp16