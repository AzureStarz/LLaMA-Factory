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
@InProceedings{pawsx2019emnlp,
  title = {{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification}},
  author = {Yang, Yinfei and Zhang, Yuan and Tar, Chris and Baldridge, Jason},
  booktitle = {Proc. of EMNLP},
  year = {2019}
}
"""

_DESCRIPTION = """\
This dataset contains 23,659 human translated PAWS evaluation pairs and 296,406 machine translated training pairs in six typologically distinct languages: French, Spanish, German, Chinese, Japanese, and Korean. All translated pairs are sourced from examples in PAWS-Wiki.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/paws/tree/master/pawsx"

_LICENSE = "MIT"

_URL = "https://storage.googleapis.com/paws/pawsx/x-final.tar.gz"
_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/paws-x"

language_list = """
de  en  es  fr  ja  ko  zh
""".split()

class Paws_xConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class Paws_x(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        Paws_xConfig(
            name=language,
        )
        for language in language_list
    ]
    MAPPING = {'0': 'False', '1': 'True'}

    def _info(self):
        features = datasets.Features(
            {
                "premise": datasets.Value("string"),
                "hypothesis": datasets.Value("string"),
                "label": datasets.Value("string"),
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
                        data_dir, "x-final", language, f"test_2k.tsv"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "x-final", language, f"dev_2k.tsv"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        # Training set is in tsv format
        df = pd.read_csv(filepath, sep="\t")
        df.columns = ["id", "premise", "hypothesis", "label"]
        # 利用apply方法将df["answer"]从0,1映射成False, True
        df["label"] = df["label"].apply(lambda x: self.MAPPING[str(x)])
        selected_cols = ["premise", "hypothesis", "label"]

        for i, instance in enumerate(df[selected_cols].to_dict(orient="records")):
            yield i, instance