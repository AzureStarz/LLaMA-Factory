import json

dataset_dir = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/mlqa/data"
LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar', 'zh']
SPLITS = ['test', 'dev']

for lang in LANGS:
    for split in SPLITS:
        filename = f"{dataset_dir}/{split}/{split}-context-{lang}-question-{lang}.json"
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
        #         if split == 'test' and  len(new_data) > 1000:
        #             with open(f"{dataset_dir}/{split}/{split}-context-{lang}-question-{lang}-preprocess.json", "w") as f:
        #                 json.dump(new_data, f, indent=4, ensure_ascii=False)
        #                 break
        # if split == "dev":
        #     with open(f"{dataset_dir}/{split}/{split}-context-{lang}-question-{lang}-preprocess.json", "w") as f:
        #         json.dump(new_data, f, indent=4, ensure_ascii=False)
        with open(f"{dataset_dir}/{split}/{split}-context-{lang}-question-{lang}-preprocess.json", "w") as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)
