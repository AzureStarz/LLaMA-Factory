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
import pandas as pd


_CITATION = """\
@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md. Saiful  and
      Mubasshir, Kazi  and
      Li, Yuan-Fang  and
      Kang, Yong-Bin  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.413",
    pages = "4693--4703",
}
"""

_DESCRIPTION = """\
This repository contains the code, data, and models of the paper titled "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages" published in Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021.
"""

_HOMEPAGE = "https://github.com/csebuetnlp/xl-sum?tab=readme-ov-file#xl-sum"

_LICENSE = "MIT"

_URL = "https://github.com/csebuetnlp/xl-sum?tab=readme-ov-file#xl-sum"
_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/xlsum"

language_list = [
    "amharic",
    "arabic",
    "azerbaijani",
    "bengali",
    "burmese",
    "chinese_simplified",
    "chinese_traditional",
    "english",
    "french",
    "gujarati",
    "hausa",
    "hindi",
    "igbo",
    "indonesian",
    "japanese",
    "kirundi",
    "korean",
    "kyrgyz",
    "marathi",
    "nepali",
    "oromo",
    "pashto",
    "persian",
    "pidgin",
    "portuguese",
    "punjabi",
    "russian",
    "scottish_gaelic",
    "serbian_cyrillic",
    "serbian_latin",
    "sinhala",
    "somali",
    "spanish",
    "swahili",
    "tamil",
    "telugu",
    "thai",
    "tigrinya",
    "turkish",
    "ukrainian",
    "urdu",
    "uzbek",
    "vietnamese",
    "welsh",
    "yoruba"
]


class XlsumConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class Xlsum(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        XlsumConfig(
            name=language,
        )
        for language in language_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "passage": datasets.Value("string"),
                "summary": datasets.Value("string"),
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
                        data_dir, "data", "test", f"{language}_test.jsonl"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "val", f"{language}_val.jsonl"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "train", f"{language}_train.jsonl"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                instance = {
                    "passage": data["text"],
                    "summary": data["summary"],
                }
                yield i, instance
