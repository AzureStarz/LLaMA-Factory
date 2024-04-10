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
@misc{chen2023breaking,
      title={Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations}, 
      author={Nuo Chen and Zinan Zheng and Ning Wu and Linjun Shou and Ming Gong and Yangqiu Song and Dongmei Zhang and Jia Li},
      year={2023},
      eprint={2310.20246},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
MSVAMP, an out-of-domain xMR test dataset, to conduct a more exhaustive and comprehensive evaluation of the model’s multilingual mathematical capabilities.
"""

_HOMEPAGE = "https://github.com/microsoft/MathOctopus/tree/main"

_LICENSE = "MIT"

_URL = "https://huggingface.co/datasets/Mathoctopus/MSVAMP"
_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/msvamp"

language_list = """
Bengali  English  German    Russian  Swahili
Chinese  French   Japanese  Spanish  Thai
""".split()


class MSVAMPConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class MSVAMP(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MSVAMPConfig(
            name=language,
        )
        for language in language_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
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
                        data_dir, "data", "test", f"test_{language}.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "train", f"train.jsonl"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        is_train = "train.jsonl" in filepath
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                instance = {
                    "question": data["question"] if is_train else data["m_query"],
                    "summary": data["answer"] if is_train else data["response"],
                }
                yield i, instance