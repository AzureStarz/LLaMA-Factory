import json
import os

prefixes = ['ces_Latn', 'dan_Latn', 'ukr_Cyrl', 'bul_Cyrl', 'fin_Latn', 'hun_Latn', 'nob_Latn', 'ind_Latn', 'jpn_Jpan', 'kor_Hang', 'por_Latn', 'slv_Latn', 'vie_Latn', 'pol_Latn']

lang_mapping = {'ces_Latn': 'Czech', 'dan_Latn': 'Danish', 'ukr_Cyrl': 'Ukrainian', 'bul_Cyrl': 'Bulgarian', 'fin_Latn': 'Finnish', 'hun_Latn': 'Hungarian', 
                'nob_Latn': 'Norwegian', 'ind_Latn': 'Indonesian', 'jpn_Jpan': 'Japanese', 'kor_Hang': 'Korean', 'por_Latn': 'Portuguese', 'slv_Latn': 'Slovenian', 
                'vie_Latn': 'Vietnamese', 'pol_Latn': 'Polish'}


base_dir = '/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation'
flores_dir = f'{base_dir}/flores'
raw_data_dir = f'{flores_dir}/flores200_dataset'

devtest_output_dir = f'{flores_dir}/data/test'
dev_output_dir = f'{flores_dir}/data/dev'

if not os.path.exists(devtest_output_dir):
    os.makedirs(devtest_output_dir)
if not os.path.exists(dev_output_dir):
    os.makedirs(dev_output_dir)

pivot_language = 'eng_Latn'

with open(f'{raw_data_dir}/devtest/{pivot_language}.devtest', 'r', encoding='utf-8') as f:
    devtest_eng_sents = f.read().splitlines()

with open(f'{raw_data_dir}/dev/{pivot_language}.dev', 'r', encoding='utf-8') as f:
    dev_eng_sents = f.read().splitlines()

dataset_info_dict = {}

for prefix in prefixes:
    to_eng_data = []
    from_eng_data = []
    file_name = f'{raw_data_dir}/devtest/{prefix}.devtest'
    with open(file_name, 'r', encoding='utf-8') as f:
        input_sents = f.read().splitlines()

    for idx, input_sent in enumerate(input_sents):
        to_english_instance = dict()
        to_english_instance['instruction'] = ""
        to_english_instance['input'] = input_sent
        to_english_instance['output'] = devtest_eng_sents[idx]
        to_eng_data.append(to_english_instance)
    
        from_english_instance = dict()
        from_english_instance['instruction'] = ""
        from_english_instance['input'] = devtest_eng_sents[idx]
        from_english_instance['output'] = input_sent
        from_eng_data.append(from_english_instance)
        
    with open(f'{devtest_output_dir}/{prefix}_to_{pivot_language}.json', "w", encoding="utf-8") as outfile:
        json.dump(to_eng_data, outfile, ensure_ascii=False, indent=4)
    
    with open(f'{devtest_output_dir}/{pivot_language}_to_{prefix}.json', "w", encoding="utf-8") as outfile:
        json.dump(from_eng_data, outfile, ensure_ascii=False, indent=4)
    
    to_eng_data = []
    from_eng_data = []
    file_name = f'{raw_data_dir}/dev/{prefix}.dev'
    with open(file_name, 'r', encoding='utf-8') as f:
        input_sents = f.read().splitlines()

    for idx, input_sent in enumerate(input_sents):
        to_english_instance = dict()
        to_english_instance['instruction'] = ""
        to_english_instance['input'] = input_sent
        to_english_instance['output'] = dev_eng_sents[idx]
        to_eng_data.append(to_english_instance)
    
        from_english_instance = dict()
        from_english_instance['instruction'] = ""
        from_english_instance['input'] = dev_eng_sents[idx]
        from_english_instance['output'] = input_sent
        from_eng_data.append(from_english_instance)
        
    with open(f'{dev_output_dir}/{prefix}_to_{pivot_language}.json', "w", encoding="utf-8") as outfile:
        json.dump(to_eng_data, outfile, ensure_ascii=False, indent=4)
    
    with open(f'{dev_output_dir}/{pivot_language}_to_{prefix}.json', "w", encoding="utf-8") as outfile:
        json.dump(from_eng_data, outfile, ensure_ascii=False, indent=4)