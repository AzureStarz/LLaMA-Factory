import json

SKIP_TYPES = ['long_answer', 'unanswerable']

def split_data_by_language(data):
    for language in data['queries'].keys():
        with open(f'/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/mkqa/data/test/{language}.jsonl', 'a') as file:
            json.dump({
                'example_id': data['example_id'],
                'query': data['queries'][language],
                'answers': data['answers'][language]
            }, file)
            file.write('\n')

def process_file(filename):
    test_len = 0
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            if data['answers']['en'][0]['type'] in SKIP_TYPES:
                continue
            split_data_by_language(data)
            test_len += 1
            if test_len > 1000:
                break

process_file('mkqa.jsonl')
