import json
import os

def convert_to_jsonl(file_path, output_path, lang):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip().split('\n\n')

    data = []
    for example in content:
        lines = example.split('\n')
        tokens = []
        tags = []
        for line in lines:
            token, tag = line.split('	')
            tokens.append(token.replace(f"{lang}:", ""))
            tags.append(tag)
        data.append({"tokens": tokens, "tags": tags})
        if len(data) > 100:
            break

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

dataset_dir = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/pan-x/panx_dataset"

for lang in os.listdir(dataset_dir):
    for split in ['train', 'dev', 'test']:
        convert_to_jsonl(f'{dataset_dir}/{lang}/{split}', f'{dataset_dir}/{lang}/{split}_sample100.jsonl', lang)

# 使用函数
# convert_to_jsonl('/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/pan-x/panx_dataset/en/test', 'output.jsonl')
