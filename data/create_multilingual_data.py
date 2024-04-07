import os
import json

def merge_json_files(folder_path, output_path):
    total_list = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and filename != 'en.json':
            file_path = os.path.join(folder_path, filename)
            # 读取.json文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data_list = json.load(file)

            for data in data_list:
                # 创建新的.json文件，其中的instruction取自当前文件，input和output取自en.json
                new_data = {
                    "instruction": data["instruction"],
                    "input": data["input"],
                    "output": data["output"],
                }
                total_list.append(new_data)

    # 将新的.json文件写入到output_path中
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in total_list:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
# 使用示例
raw_data_path = '/home/export/base/ycsc_chenkh/hitici_02/online1/data/InstructData/Bactrian-X/data'
output_data_path = '/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/data/Bactrian-X.jsonl'
merge_json_files(raw_data_path, output_data_path)
