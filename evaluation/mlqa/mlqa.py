# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import datasets
import json

_CITATION = """\
tmp
"""

_DESCRIPTION = """\
MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance. MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between 4 different languages on average.
"""

_HOMEPAGE = "https://github.com/facebookresearch/MLQA/tree/main"

_LICENSE = "MIT"

_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"
_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/mlqa"

language_list = ['en', 'de', 'es', 'ar', 'zh', 'vi', 'hi']


class MLQAConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class MLQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MLQAConfig(
            name=language,
        )
        for language in language_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "passage": datasets.Value("string"),
                "summary": datasets.Value("string"),
                "question": datasets.Value("string"),
                "id": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # data_dir = dl_manager.download_and_extract(_URL)
        data_dir = _DATASET_DIR
        language = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "test", f"test-context-{language}-question-{language}-preprocess.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "dev", f"dev-context-{language}-question-{language}-preprocess.json"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for i, item in enumerate(data):
                instance = {
                    "passage": item["context"],
                    "id": item["id"],
                    "summary": item["answers"][0],
                    "question": item["question"],
                }
                yield i, instance
