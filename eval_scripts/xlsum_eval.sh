#!/bin/bash
# constant path
base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory
model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b
# task-specific info
task=xlsum
eval_class=ats
task_dir=${base_path}/evaluation
template=llama2
test_split=test
# param path
lora_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_exp_output/alpaca_en_llama2-7b
save_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/xlsum_alpaca_en_llama2-7b
mkdir -p ${save_path}

langs=('indonesian' 'japanese' 'korean' 'portuguese' 'vietnamese' 'ukrainian')
for lang in ${langs[*]}
do
CUDA_VISIBLE_DEVICES=0 python ${base_path}/src/evaluate.py \
    --model_name_or_path ${model_path} \
    --adapter_name_or_path ${lora_path} \
    --template ${template} \
    --eval_template ${task} \
    --eval_class ${eval_class} \
    --save_dir ${save_path}/${lang} \
    --lang ${lang} \
    --finetuning_type lora \
    --task_dir ${task_dir} \
    --task ${task} \
    --split ${test_split} \
    --n_shot 1 \
    --batch_size 1 \
    --seed 42
done