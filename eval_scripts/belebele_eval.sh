#!/bin/bash
# constant path
base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory
model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b
# task-specific info
task=belebele
eval_class=mrc
task_dir=${base_path}/evaluation
template=llama2
test_split=test
# param path
eval_model=bactrian-x_llama2-7b
lora_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_exp_output/${eval_model}
save_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/${task}_${eval_model}/fewshot_eng_examples
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