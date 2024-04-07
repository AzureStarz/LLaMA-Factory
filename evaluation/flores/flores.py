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
import json
import datasets


_CITATION = """\
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}

@inproceedings{,
  title={The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},
  author={Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm\'{a}n, Francisco and Fan, Angela},
  year={2021}
}

@inproceedings{,
  title={Two New Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English},
  author={Guzm\'{a}n, Francisco and Chen, Peng-Jen and Ott, Myle and Pino, Juan and Lample, Guillaume and Koehn, Philipp and Chaudhary, Vishrav and Ranzato, Marc'Aurelio},
  journal={arXiv preprint arXiv:1902.01382},
  year={2019}
}
"""

_DESCRIPTION = """\
The creation of FLORES-200 doubles the existing language coverage of FLORES-101. Given the nature of the new languages, which have less standardization and require more specialized professional translations, the verification process became more complex. This required modifications to the translation workflow. FLORES-200 has several languages which were not translated from English. Specifically, several languages were translated from Spanish, French, Russian and Modern Standard Arabic. Moreover, FLORES-200 also includes two script alternatives for four languages.
"""

_HOMEPAGE = "https://github.com/facebookresearch/flores/blob/main/flores200/README.md"

_LICENSE = "CC-BY-SA-4.0 license"

_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/flores"

_URL = "https://tinyurl.com/flores200dataset"

language_pair_list = """
bul_Cyrl_to_eng_Latn  eng_Latn_to_dan_Latn  eng_Latn_to_kor_Hang  eng_Latn_to_ukr_Cyrl  jpn_Jpan_to_eng_Latn  slv_Latn_to_eng_Latn
ces_Latn_to_eng_Latn  eng_Latn_to_fin_Latn  eng_Latn_to_nob_Latn  eng_Latn_to_vie_Latn  kor_Hang_to_eng_Latn  ukr_Cyrl_to_eng_Latn
dan_Latn_to_eng_Latn  eng_Latn_to_hun_Latn  eng_Latn_to_pol_Latn  fin_Latn_to_eng_Latn  nob_Latn_to_eng_Latn  vie_Latn_to_eng_Latn
eng_Latn_to_bul_Cyrl  eng_Latn_to_ind_Latn  eng_Latn_to_por_Latn  hun_Latn_to_eng_Latn  pol_Latn_to_eng_Latn
eng_Latn_to_ces_Latn  eng_Latn_to_jpn_Jpan  eng_Latn_to_slv_Latn  ind_Latn_to_eng_Latn  por_Latn_to_eng_Latn
""".split()


class FloresConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class Flores(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        FloresConfig(
            name=language_pair,
        )
        for language_pair in language_pair_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "instruction": datasets.Value("string"),
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
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
        language_pair = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, 'data', "test", f"{language_pair}.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, 'data', "dev", f"{language_pair}.json"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for i, old_instance in enumerate(data):
                instance = old_instance
                if "source" not in old_instance:
                    instance = {
                        "instruction": old_instance["instruction"],
                        "source": old_instance["input"],
                        "reference": old_instance["output"]
                    }
                yield i, instance
