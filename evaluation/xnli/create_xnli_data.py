import os
import json

filename = '/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/xnli/XNLI-1.0/xnli.dev.jsonl'
target_dir = '/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/xnli/data/dev'
os.makedirs(target_dir, exist_ok=True)

with open(filename, 'r') as f:
    for line in f:
        # 将每一行解析为一个 JSON 对象
        json_obj = json.loads(line)
        
        # 获取该 JSON 对象的 "language" 键的值
        language = json_obj.get('language')
        if not language:
            continue  # 如果这个 JSON 对象没有 "language" 键，跳过它
        
        # 创建一个以 "language" 为名的子目录（如果它还不存在的话）
        # subdir_path = os.path.join(target_dir, language)
        # os.makedirs(subdir_path, exist_ok=True)
        
        # 将这个 JSON 对象写入到相应的文件中
        new_filename = os.path.join(target_dir, f'{language}.jsonl')
        with open(new_filename, 'a') as f:
            f.write(json.dumps(json_obj) + '\n')