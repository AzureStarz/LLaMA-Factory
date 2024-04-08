import os
import json
import glob

# 指定源目录和目标目录
source_dir = '/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/xlsum/XLSum_complete_v2.0'
target_dir = '/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/xlsum/data'

# 如果目标目录不存在，则创建它
os.makedirs(target_dir, exist_ok=True)

# 遍历源目录下的所有.jsonl文件
for filename in glob.glob(os.path.join(source_dir, '*.jsonl')):
    # 根据文件名确定子目录
    if '_test.jsonl' in filename:
        subdir = 'test'
    elif '_val.jsonl' in filename:
        subdir = 'val'
    elif '_train.jsonl' in filename:
        subdir = 'train'
    else:
        continue  # 如果文件名不符合预期，跳过这个文件

    # 创建子目录（如果它还不存在的话）
    subdir_path = os.path.join(target_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)

    # 读取每个文件的前1000行
    with open(filename, 'r') as f:
        lines = []
        for i, line in enumerate(f):
            if i == 1000:
                break
            lines.append(line)
    
    # 将这1000行写入到新的文件中
    new_filename = os.path.join(subdir_path, os.path.basename(filename))
    with open(new_filename, 'w') as f:
        f.writelines(lines)
