import os
import json
from sacrebleu.metrics import BLEU, CHRF, TER

bleu = BLEU(tokenize='flores101', effective_order=True)

OPTIONS = ['A', 'B', 'C', 'D']

def process_jsonl_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("generated_predictions.jsonl"):
                filepath = os.path.join(root, file)
                output_label_filepath = os.path.join(root, "labels.txt")
                output_predict_filepath = os.path.join(root, "predictions.txt")
                
                prefix = root.split('/')[-1]
                belebele_correct_answer = f'/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/belebele/Belebele/{prefix}.jsonl'
                with open(belebele_correct_answer, 'r', encoding='utf-8') as input_file:
                    correct_answers = input_file.read().splitlines()

                correct = 0
                fail = 0

                with open(filepath, 'r') as f, open(output_label_filepath, 'w') as label_file, open(output_predict_filepath, 'w') as predict_file:
                    for i, line in enumerate(f):
                        data = json.loads(line)
                        label = data['label']
                        predict = data['predict'].replace('(','').replace(')','').replace('[','').replace(']','')
                        if predict not in OPTIONS:
                            fail += 1
                            correct_answer_item = json.loads(correct_answers[i])
                            choices = [correct_answer_item.get(f'mc_answer{i}', '') for i in range(1, 5)]
                            predict_option = 'A'
                            max_score = -1
                            for opt, choice in enumerate(choices):
                                score = bleu.sentence_score(predict, [choice]).score
                                if score > max_score:
                                    predict_option = opt
                            predict = OPTIONS[predict_option]

                        if predict == label:
                            correct += 1
                
                acc = float(correct / len(correct_answers)) * 100
                print(f'{prefix}: Accuracy: {acc}% Fails: {fail} Correct: {correct}')


# 调用函数并传入目录路径
process_jsonl_files('/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLM-SFT_pred_output/belebele_eng_instruct_alpaca_en_llama2-7b')
