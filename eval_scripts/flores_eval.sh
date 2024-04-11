#!/bin/bash
# comet_model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/wmt22-comet-da/checkpoints/model.ckpt
# base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory
# flores_data_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/data/translationData/flores101_dataset/devtest
# langs=('ces' 'dan' 'ukr' 'bul' 'fin' 'hun' 'nob' 'ind' 'jpn' 'kor' 'por' 'slv' 'vie' 'pol')

# for lang in ${langs[*]}
# do
# echo ======
# # to_english
# dataset=${lang}_to_eng
# output_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/flores_alpaca_en_llama2-7b/${dataset}
# if [ -e ${output_dir}/labels.txt ];then
#     echo ${dataset}
#     cat ${output_dir}/result.txt
#     # if [ -e ${output_dir}/result.txt ];then
#     #     continue
#     # fi
#     # sacrebleu ${output_dir}/labels.txt -i ${output_dir}/predictions.txt --tokenize flores101 -m bleu -b -w 4 >> ${output_dir}/result.txt
#     # comet-score --model ${comet_model_path} -s ${flores_data_dir}/${lang}.devtest -t ${output_dir}/predictions.txt -r ${output_dir}/labels.txt --quiet --only_system >> ${output_dir}/result.txt
# fi
# echo ------
# # from_english
# dataset=eng_to_${lang}
# output_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/flores_alpaca_en_llama2-7b/${dataset}
# if [ -e ${output_dir}/labels.txt ];then
#     echo ${dataset}
#     cat ${output_dir}/result.txt
#     # if [ -e ${output_dir}/result.txt ];then
#     #     continue
#     # fi
#     # sacrebleu ${output_dir}/labels.txt -i ${output_dir}/predictions.txt --tokenize flores101 -m bleu -b -w 4 >> ${output_dir}/result.txt
#     # comet-score --model ${comet_model_path} -s ${flores_data_dir}/eng.devtest -t ${output_dir}/predictions.txt -r ${output_dir}/labels.txt --quiet --only_system >> ${output_dir}/result.txt
# fi
# done
# constant path
base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory
model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b
# task-specific info
task=flores
eval_class=mmt
task_dir=${base_path}/evaluation
template=llama2
test_split=test
# params path
eval_model=bactrian-x_llama2-7b
lora_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_exp_output/${eval_model}
save_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/${task}_${eval_model}
mkdir -p ${save_path}

langs=('ces_Latn' 'dan_Latn' 'ukr_Cyrl' 'bul_Cyrl' 'fin_Latn' 'hun_Latn' 'nob_Latn' 'ind_Latn' 'jpn_Jpan' 'kor_Hang' 'por_Latn' 'slv_Latn' 'vie_Latn' 'pol_Latn')
for lang in ${langs[*]}
do
# from_english
lang_pair=eng_Latn_to_${lang}
CUDA_VISIBLE_DEVICES=0 python ${base_path}/src/evaluate.py \
    --model_name_or_path ${model_path} \
    --adapter_name_or_path ${lora_path} \
    --template ${template} \
    --eval_template ${task} \
    --eval_class ${eval_class} \
    --save_dir ${save_path}/${lang_pair} \
    --lang_pair ${lang_pair} \
    --finetuning_type lora \
    --task_dir ${task_dir} \
    --task ${task} \
    --split ${test_split} \
    --n_shot 5 \
    --batch_size 8 \
    --seed 42
# to_english
lang_pair=${lang}_to_eng_Latn
CUDA_VISIBLE_DEVICES=0 python ${base_path}/src/evaluate.py \
    --model_name_or_path ${model_path} \
    --adapter_name_or_path ${lora_path} \
    --template ${template} \
    --eval_template ${task} \
    --eval_class ${eval_class} \
    --save_dir ${save_path}/${lang_pair} \
    --lang_pair ${lang_pair} \
    --finetuning_type lora \
    --task_dir ${task_dir} \
    --task ${task} \
    --split ${test_split} \
    --n_shot 5 \
    --batch_size 8 \
    --seed 42
done