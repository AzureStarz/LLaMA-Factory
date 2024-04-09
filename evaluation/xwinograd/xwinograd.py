"""XWinograd"""

import json
import os
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@misc{muennighoff2022crosslingual,
      title={Crosslingual Generalization through Multitask Finetuning}, 
      author={Niklas Muennighoff and Thomas Wang and Lintang Sutawika and Adam Roberts and Stella Biderman and Teven Le Scao and M Saiful Bari and Sheng Shen and Zheng-Xin Yong and Hailey Schoelkopf and Xiangru Tang and Dragomir Radev and Alham Fikri Aji and Khalid Almubarak and Samuel Albanie and Zaid Alyafeai and Albert Webson and Edward Raff and Colin Raffel},
      year={2022},
      eprint={2211.01786},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{tikhonov2021heads,
    title={It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning},
    author={Alexey Tikhonov and Max Ryabinin},
    year={2021},
    eprint={2106.12066},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
A multilingual collection of Winograd Schemas in six languages \
that can be used for evaluation of cross-lingual commonsense reasoning capabilities.
"""
_LANG = ["en", "fr", "jp", "pt", "ru", "zh"]
_URL = "https://huggingface.co/datasets/Muennighoff/xwinograd/raw/main/test/{lang}.jsonl"
_VERSION = datasets.Version("1.1.0", "")
_DATASET_DIR = "/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/xwinograd"

class XWinograd(datasets.GeneratorBasedBuilder):
    """XWinograd"""


    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=lang,
            description=f"XWinograd in {lang}",
            version=_VERSION,
        )
        for lang in _LANG
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "passage": datasets.Value("string"),
                    "A": datasets.Value("string"),
                    "B": datasets.Value("string"),
                    "answer": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        # downloaded_files = dl_manager.download(_URL.format(lang=self.config.name))
        testfile_path = os.path.join(_DATASET_DIR, "data", "test", f"test_{self.config.name}.jsonl")
        trainfile_path = os.path.join(_DATASET_DIR, "data", "dev", "winograde_dev.jsonl")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': testfile_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'filepath': trainfile_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("Generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)

                yield id_, {
                "passage": data["sentence"],
                "A": data["option1"],
                "B": data["option2"],
                "answer": chr(ord('A') + int(data["answer"]) - 1),
            }
