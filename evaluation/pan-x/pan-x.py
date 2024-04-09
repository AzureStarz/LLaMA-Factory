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
@inproceedings{pan-etal-2017-cross,
    title = "Cross-lingual Name Tagging and Linking for 282 Languages",
    author = "Pan, Xiaoman  and
      Zhang, Boliang  and
      May, Jonathan  and
      Nothman, Joel  and
      Knight, Kevin  and
      Ji, Heng",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-1178",
    doi = "10.18653/v1/P17-1178",
    pages = "1946--1958",
    abstract = "The ambitious goal of this work is to develop a cross-lingual name tagging and linking framework for 282 languages that exist in Wikipedia. Given a document in any of these languages, our framework is able to identify name mentions, assign a coarse-grained or fine-grained type to each mention, and link it to an English Knowledge Base (KB) if it is linkable. We achieve this goal by performing a series of new KB mining methods: generating {``}silver-standard{''} annotations by transferring annotations from English to other languages through cross-lingual links and KB properties, refining annotations through self-training and topic selection, deriving language-specific morphology features from anchor links, and mining word translation pairs from cross-lingual links. Both name tagging and linking results for 282 languages are promising on Wikipedia data and on-Wikipedia data.",
}
@inproceedings{rahimi-etal-2019-massively,
    title = "Massively Multilingual Transfer for {NER}",
    author = "Rahimi, Afshin  and
      Li, Yuan  and
      Cohn, Trevor",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1015",
    pages = "151--164",
}
"""

_DESCRIPTION = """\
WikiANN (sometimes called PAN-X) is a multilingual named entity recognition dataset consisting of Wikipedia articles annotated with LOC (location), PER (person), and ORG (organisation) tags in the IOB2 format. This version corresponds to the balanced train, dev, and test splits of Rahimi et al. (2019), which supports 176 of the 282 languages from the original WikiANN corpus.
"""

_HOMEPAGE = "https://huggingface.co/datasets/wikiann"

_LICENSE = "MIT"

_URL = "https://www.kaggle.com/datasets/shivanshuman/wikiann-or-panx?select=panx_dataset"
_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/pan-x"

language_list = """
ace  arz      be        ca       cs   el   fa       ga   hi   ilo  kk   lb       mg   ms   nl   pdc  ro   sh      su   tk  vec  xmf
af   as       be-x-old  cbk-zam  csb  eml  fi       gan  hr   io   km   li       mhr  mt   nn   pl   ru   si      sv   tl  vep  yi
als  ast      bg        cdo      cv   en   fiu-vro  gd   hsb  is   kn   lij      mi   mwl  no   pms  rw   simple  sw   tr  vi   yo
am   ay       bh        ce       cy   eo   fo       gl   hu   it   ko   lmo      min  my   nov  pnb  sa   sk      szl  tt  vls  zea
an   az       bn        ceb      da   es   fr       gn   hy   ja   ksh  ln       mk   mzn  oc   ps   sah  sl      ta   ug  vo   zh
ang  ba       bo        ckb      de   et   frr      gu   ia   jbo  ku   lt       ml   nap  or   pt   scn  so      te   uk  wa   zh-classical
ar   bar      br        co       diq  eu   fur      hak  id   jv   ky   lv       mn   nds  os   qu   sco  sq      tg   ur  war  zh-min-nan
arc  bat-smg  bs        crh      dv   ext  fy       he   ig   ka   la   map-bms  mr   ne   pa   rm   sd   sr      th   uz  wuu  zh-yue
""".split()


class PAN_XConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class PAN_X(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        PAN_XConfig(
            name=language,
        )
        for language in language_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "tags": datasets.Sequence(datasets.Value("string")),
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
                        data_dir, "data", language, f"test_sample100.jsonl"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", language, f"dev_sample100.jsonl"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", language, f"train_sample100.jsonl"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                instance = {
                    "tokens": data["tokens"],
                    "tags": data["tags"],
                }
                yield i, instance
