#!/bin/bash

langs=('indonesian' 'japanese' 'korean' 'portuguese' 'vietnamese' 'ukrainian')
# langs=('indonesian')
for lang in ${langs[*]}
do

dataset=${lang}
output_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/xlsum_eng_instruct_alpaca_en_llama2-7b/${dataset}
if [ -e ${output_dir}/labels.txt ];then
    if [ -e ${output_dir}/result.txt ];then
        continue
    fi
    rouge -f ${output_dir}/predictions.txt ${output_dir}/labels.txt --avg > ${output_dir}/result.txt
fi

done