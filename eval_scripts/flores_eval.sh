#!/bin/bash
comet_model_path=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/wmt22-comet-da/checkpoints/model.ckpt
base_path=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory
flores_data_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/data/translationData/flores101_dataset/devtest
langs=('ces' 'dan' 'ukr' 'bul' 'fin' 'hun' 'nob' 'ind' 'jpn' 'kor' 'por' 'slv' 'vie' 'pol')

for lang in ${langs[*]}
do
echo ======
# to_english
dataset=${lang}_to_eng
output_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/flores_alpaca_en_llama2-7b/${dataset}
if [ -e ${output_dir}/labels.txt ];then
    echo ${dataset}
    cat ${output_dir}/result.txt
    # if [ -e ${output_dir}/result.txt ];then
    #     continue
    # fi
    # sacrebleu ${output_dir}/labels.txt -i ${output_dir}/predictions.txt --tokenize flores101 -m bleu -b -w 4 >> ${output_dir}/result.txt
    # comet-score --model ${comet_model_path} -s ${flores_data_dir}/${lang}.devtest -t ${output_dir}/predictions.txt -r ${output_dir}/labels.txt --quiet --only_system >> ${output_dir}/result.txt
fi
echo ------
# from_english
dataset=eng_to_${lang}
output_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/flores_alpaca_en_llama2-7b/${dataset}
if [ -e ${output_dir}/labels.txt ];then
    echo ${dataset}
    cat ${output_dir}/result.txt
    # if [ -e ${output_dir}/result.txt ];then
    #     continue
    # fi
    # sacrebleu ${output_dir}/labels.txt -i ${output_dir}/predictions.txt --tokenize flores101 -m bleu -b -w 4 >> ${output_dir}/result.txt
    # comet-score --model ${comet_model_path} -s ${flores_data_dir}/eng.devtest -t ${output_dir}/predictions.txt -r ${output_dir}/labels.txt --quiet --only_system >> ${output_dir}/result.txt
fi
done