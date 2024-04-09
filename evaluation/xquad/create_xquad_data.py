import json

dataset_dir = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/xquad/data"
LANGS = """
ar  el  es  ro  th  vi
de  en  hi  ru  tr  zh
""".split()
# LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar', 'zh']
SPLITS = ['test']

for lang in LANGS:
    for split in SPLITS:
        filename = f"{dataset_dir}/{split}/xquad.{lang}.json"
        new_data = []
        with open(filename, 'r') as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
            for article in dataset:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        q_id = qa['id']
                        question = qa['question']
                        answers = [answer['text'] for answer in qa['answers']]
                        data = {
                            'id': q_id,
                            'context': context,
                            'question': question,
                            'answers': answers,
                        }
                        new_data.append(data)
                if len(new_data) > 100:
                    with open(f"{dataset_dir}/{split}/{lang}-sample100.json", "w") as f:
                        json.dump(new_data, f, indent=4, ensure_ascii=False)
                    break

dev_filename = f"{dataset_dir}/dev/dev-v2.0.json"
new_data = []
with open(dev_filename, 'r') as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']
    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                q_id = qa['id']
                question = qa['question']
                answers = [answer['text'] for answer in qa['answers']]
                if len(answers) == 0:
                    continue
                data = {
                    'id': q_id,
                    'context': context,
                    'question': question,
                    'answers': answers,
                }
                new_data.append(data)
with open(f"{dataset_dir}/dev/xquad_dev.json", "w") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)