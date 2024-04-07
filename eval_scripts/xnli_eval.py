import os
import json
import string

OPTIONS = ['Yes', 'False', 'Neither', 'No', 'True']
MAPPINGS = {'entailment': ['Yes', 'True'], 'contradiction': ['False', 'No'], 'neutral': 'Neither'}

def remove_punctuation(text):
    # 创建一个翻译表，将标点符号映射为 None
    translator = str.maketrans('', '', string.punctuation)
    # 使用 translate() 方法去除标点符号
    return text.translate(translator)


def process_jsonl_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("generated_predictions.jsonl"):
                filepath = os.path.join(root, file)
                output_label_filepath = os.path.join(root, "labels.txt")
                output_predict_filepath = os.path.join(root, "predictions.txt")
                
                prefix = root.split('/')[-1]

                correct = 0
                fail = 0

                with open(filepath, 'r') as f, open(output_label_filepath, 'w') as label_file, open(output_predict_filepath, 'w') as predict_file:
                    for i, line in enumerate(f):
                        data = json.loads(line)
                        label = data['label']
                        predict = remove_punctuation(data['predict'].split()[0])
                        if predict not in OPTIONS:
                            fail += 1

                        if predict in MAPPINGS[label]:
                            correct += 1
                
                acc = float(correct / i) * 100
                print(f'{prefix}: Accuracy: {acc}% Fails: {fail} Correct: {correct}')


# 调用函数并传入目录路径
process_jsonl_files('/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/xnli_eng_instruct_alpaca_en_llama2-7b')
