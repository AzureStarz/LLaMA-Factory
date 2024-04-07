import os
import json

def process_jsonl_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("generated_predictions.jsonl"):
                filepath = os.path.join(root, file)
                output_label_filepath = os.path.join(root, "labels.txt")
                output_predict_filepath = os.path.join(root, "predictions.txt")
                
                with open(filepath, 'r') as f, open(output_label_filepath, 'w') as label_file, open(output_predict_filepath, 'w') as predict_file:
                    for line in f:
                        data = json.loads(line)
                        label = data['label']
                        predict = data['predict'].replace('\n', '')
                        # if len(predict) == 0:
                        #     continue
                        label_file.write(label + '\n')
                        predict_file.write(predict + '\n')

# 调用函数并传入目录路径
process_jsonl_files('/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/xlsum_eng_instruct_alpaca_en_llama2-7b')
