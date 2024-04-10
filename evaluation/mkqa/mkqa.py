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
@misc{mkqa,
    title = {MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering},
    author = {Shayne Longpre and Yi Lu and Joachim Daiber},
    year = {2020},
    URL = {https://arxiv.org/pdf/2007.15207.pdf}
}
"""

_DESCRIPTION = """\
Given a question ql in language l, the task is to produce a prediction pl in {No Answer, Yes, No, Text Answer}, where a Text Answer is a span of tokens in the corresponding language. pl can be obtained by any method, extracted from a document, generated, or derived from a knowledge graph. Wherever possible, textual answers are accompanied by Wikidata QIDs, for entity linking and evaluating knowledge graph approaches. These QIDs also enable automatic translations for most answers into any Wikipedia language through the Wikidata knowledge graph.
"""

_HOMEPAGE = "https://github.com/apple/ml-mkqa/tree/main"

_LICENSE = "MIT"

_URL = "https://github.com/apple/ml-mkqa/blob/main/dataset/mkqa.gz"
_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/mkqa"

language_list = """
ar  de  es  fr  hu  ja  ko  nl  pl  ru  th  vi     zh_hk
da  en  fi  he  it  km  ms  no  pt  sv  tr  zh_cn  zh_tw
""".split()

class MkqaConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class Mkqa(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MkqaConfig(
            name=language,
        )
        for language in language_list
    ]

    def _info(self):
        features = datasets.Features(
            {
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
                        data_dir, "data", "test", f"{language}.jsonl"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "dev", "NQ-open.dev.jsonl"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                instance = {
                    "id": data["example_id"] if "example_id" in data else None,
                    "question": data["question"] if "question" in data else data["query"],
                    "summary": data["answer"][0] if "answer" in data else data["answers"][0]['text'],
                }
                yield i, instance
