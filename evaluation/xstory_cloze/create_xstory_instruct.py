import json
import os

dataset_info = {}

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as input_file, open(output_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            data = json.loads(line.strip())
            passage = data.get('flores_passage', '')
            query = data.get('question', '')
            choices = [data.get(f'mc_answer{i}', '') for i in range(1, 5)]
            correct_answer_num = int(data.get('correct_answer_num', '0'))
            
            if correct_answer_num == 1:
                correct_answer = 'A'
            elif correct_answer_num == 2:
                correct_answer = 'B'
            elif correct_answer_num == 3:
                correct_answer = 'C'
            elif correct_answer_num == 4:
                correct_answer = 'D'
            else:
                correct_answer = ''
            
            output_data = {
                'instruction': "Given the following passage, query, and answer choices, output the letter corresponding to the correct answer.",
                'input': f"###\nPassage:\n{passage}\n###\nQuery:\n{query}\n###\nChoices:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n###\nAnswer:\n",
                'output': f"{correct_answer}"
            }
            
            output_line = json.dumps(output_data, ensure_ascii=False)
            output_file.write(output_line + '\n')

def process_jsonl_files_in_directory(directory):
    raw_dir = os.path.join(directory, 'Belebele')
    output_dir = os.path.join(directory, 'eng_instruct_data')
    for filename in os.listdir(raw_dir):
        if filename.endswith('.jsonl'):
            input_path = os.path.join(raw_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_jsonl(input_path, output_path)
            dataset_info[filename.split('.')[0]] = {'file_name': filename}

# 使用示例
directory_path = '/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/belebele'
process_jsonl_files_in_directory(directory_path)


with open(f'{directory_path}/eng_instruct_data/dataset_info.json', "w", encoding="utf-8") as outfile:
    json.dump(dataset_info, outfile, ensure_ascii=False, indent=4)