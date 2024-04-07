import os
import json

def merge_json_files(folder_path, en_file_path, output_path):
    # 读取en.json文件
    with open(en_file_path, 'r', encoding='utf-8') as en_file:
        en_data_list = json.load(en_file)

    total_list = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and filename != 'en.json':
            file_path = os.path.join(folder_path, filename)
            # 读取.json文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data_list = json.load(file)

            for data, en_data in zip(data_list, en_data_list):
                # 创建新的.json文件，其中的instruction取自当前文件，input和output取自en.json
                to_en_new_data = {
                    "instruction": data["instruction"],
                    "input": data["input"],
                    "output": en_data["output"],
                }
                total_list.append(to_en_new_data)

                from_en_new_data = {
                    "instruction": en_data["instruction"],
                    "input": en_data["input"],
                    "output": data["output"],
                }
                total_list.append(from_en_new_data)

    # 将新的.json文件写入到output_path中
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in total_list:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
# 使用示例
merge_json_files('/home/export/base/ycsc_chenkh/hitici_02/online1/data/InstructData/Bactrian-X/data', '/home/export/base/ycsc_chenkh/hitici_02/online1/data/InstructData/Bactrian-X/data/en.json', '/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/data/Bactrian-Crosslingual.jsonl')
