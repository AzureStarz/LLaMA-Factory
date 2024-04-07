#!/bin/bash

task=belebele
eval_class=mrc
task_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation
template=llama2
test_split=test

base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory
model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b
lora_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_exp_output/alpaca_en_llama2-7b
save_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/belebele_alpaca_en_llama2-7b/fewshot_eng_examples
mkdir -p ${save_path}

langs=('ces_Latn' 'dan_Latn' 'ukr_Cyrl' 'bul_Cyrl' 'fin_Latn' 'hun_Latn' 'nob_Latn' 'ind_Latn' 'jpn_Jpan' 'kor_Hang' 'por_Latn' 'slv_Latn' 'vie_Latn' 'pol_Latn')
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
    --n_shot 5 \
    --batch_size 8 \
    --seed 42
done