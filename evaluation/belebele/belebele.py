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
@article{bandarkar2023belebele,
      title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants}, 
      author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      year={2023},
      journal={arXiv preprint arXiv:2308.16884}
}
"""

_DESCRIPTION = """\
Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. This dataset enables the evaluation of mono- and multi-lingual models in high-, medium-, and low-resource languages. Each question has four multiple-choice answers and is linked to a short passage from the FLORES-200 dataset. The human annotation procedure was carefully curated to create questions that discriminate between different levels of generalizable language comprehension and is reinforced by extensive quality checks. While all questions directly relate to the passage, the English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. Belebele opens up new avenues for evaluating and analyzing the multilingual abilities of language models and NLP systems.
"""

_HOMEPAGE = "https://github.com/facebookresearch/belebele/tree/main"

_LICENSE = "MIT"

_URL = "https://dl.fbaipublicfiles.com/belebele/Belebele.zip"
_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/belebele"

language_list = """
acm_Arab  dan_Latn  ilo_Latn  lvs_Latn  rus_Cyrl  tir_Ethi
afr_Latn  deu_Latn  ind_Latn  mal_Mlym  shn_Mymr  tsn_Latn
als_Latn  ell_Grek  isl_Latn  mar_Deva  sin_Latn  tso_Latn
amh_Ethi  eng_Latn  ita_Latn  mkd_Cyrl  sin_Sinh  tur_Latn
apc_Arab  est_Latn  jav_Latn  mlt_Latn  slk_Latn  ukr_Cyrl
arb_Arab  eus_Latn  jpn_Jpan  mri_Latn  slv_Latn  urd_Arab
arb_Latn  fin_Latn  kac_Latn  mya_Mymr  sna_Latn  urd_Latn
ars_Arab  fra_Latn  kan_Knda  nld_Latn  snd_Arab  uzn_Latn
ary_Arab  fuv_Latn  kat_Geor  nob_Latn  som_Latn  vie_Latn
arz_Arab  gaz_Latn  kaz_Cyrl  npi_Deva  sot_Latn  war_Latn
asm_Beng  grn_Latn  kea_Latn  npi_Latn  spa_Latn  wol_Latn
azj_Latn  guj_Gujr  khk_Cyrl  nso_Latn  srp_Cyrl  xho_Latn
bam_Latn  hat_Latn  khm_Khmr  nya_Latn  ssw_Latn  yor_Latn
ben_Beng  hau_Latn  kin_Latn  ory_Orya  sun_Latn  zho_Hans
ben_Latn  heb_Hebr  kir_Cyrl  pan_Guru  swe_Latn  zho_Hant
bod_Tibt  hin_Deva  kor_Hang  pbt_Arab  swh_Latn  zsm_Latn
bul_Cyrl  hin_Latn  lao_Laoo  pes_Arab  tam_Taml  zul_Latn
cat_Latn  hrv_Latn  lin_Latn  plt_Latn  tel_Telu
ceb_Latn  hun_Latn  lit_Latn  pol_Latn  tgk_Cyrl
ces_Latn  hye_Armn  lug_Latn  por_Latn  tgl_Latn
ckb_Arab  ibo_Latn  luo_Latn  ron_Latn  tha_Thai
""".split()

class BelebeleConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class Belebele(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BelebeleConfig(
            name=language,
        )
        for language in language_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "passage": datasets.Value("string"),
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "answer": datasets.Value("string"),
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
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "train", f"belebele_training_set.tsv"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        if "train" in filepath:
            # Training set is in tsv format
            df = pd.read_csv(filepath, sep="\t")
            df.columns = ["index", "dataset", "split", "passage_id", "question_id", "passage", "question", "A", "B", "C", "D", "answer_content", "answer"]
            # 利用apply方法将df["answer"]从1,2,3,4映射成A,B,C,D
            df["answer"] = df["answer"].apply(lambda x: chr(ord('A') + int(x) - 1))
            selected_cols = ["passage", "question", "A", "B", "C", "D", "answer"]

            for i, instance in enumerate(df[selected_cols].to_dict(orient="records")):
                yield i, instance
        else:
            # Test set is in jsonl format
            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    instance = {
                        "passage": data["flores_passage"],
                        "question": data["question"],
                        "A": data["mc_answer1"],
                        "B": data["mc_answer2"],
                        "C": data["mc_answer3"],
                        "D": data["mc_answer4"],
                        "answer": chr(ord('A') + int(data["correct_answer_num"]) - 1)
                    }
                    yield i, instance
