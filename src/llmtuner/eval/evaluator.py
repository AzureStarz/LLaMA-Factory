# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import inspect
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file
from transformers import GenerationConfig

# metrics related
from comet import load_from_checkpoint
from sacrebleu.metrics import BLEU
# from bleurt import score
COMET_DIR="/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/wmt22-comet-da/checkpoints/model.ckpt"
# BLEURT_CKPT="/home/export/base/ycsc_chenkh/hitici_02/online1/LLM_for_mt/LLaMA/evaluation/bleurt/BLEURT-20"

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template


class Evaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)
        #TODO use left padding or right padding?
        self.tokenizer.padding_side = "left"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args.template)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.eval_template = get_eval_template(self.eval_args.eval_template)
    
    def _save_results(self, results: List[Dict[str, str]], metric_results: Dict[str, float], results_prefix: str) -> None:
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            # 将结果列表转换为 JSONL 格式
            with open(os.path.join(self.eval_args.save_dir, f"{results_prefix}_generation_results.jsonl"), "w", encoding="utf-8") as f:
                for entry in results:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            # with open(os.path.join(self.eval_args.save_dir, f"{results_prefix}_generation_results.json"), "w", encoding="utf-8", newline="\n") as f:
            #     json.dump(results, f, indent=2)
            with open(os.path.join(self.eval_args.save_dir, f"{results_prefix}_evaluation_results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(metric_results, f, indent=4)

class MultipleChoiceEvaluator(Evaluator):
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args)
        self.choice_inputs = [
            self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in CHOICES
        ]

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        logits = self.model(**batch_input).logits
        #TODO accoding to the padding side
        # right padding?
        # lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        # left padding?
        # lengths = [len(batch_input.input_ids[k]) for k in range(len(batch_input.input_ids))]
        # word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        word_probs = torch.stack([logits[i, -1, :] for i in range(len(batch_input.input_ids))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]

class MMLUEvaluator(MultipleChoiceEvaluator):
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args)

    def eval(self) -> None:
        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, self.eval_args.task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
                kwargs = {"trust_remote_code": True}
            else:
                kwargs = {}

            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                **kwargs,
            )

            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
                support_set = (
                    dataset["train"].shuffle(seed=self.eval_args.seed).select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset[self.data_args.split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                )
                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])

            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds = self.batch_inference(batch_input)
                outputs += preds

            corrects = np.array(outputs) == np.array(labels)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)

    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join(
            [
                "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)

class MRCEvaluator(MultipleChoiceEvaluator):
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args)

    def _calculate_metrics(predictions: List[str], labels: List[str]) -> Dict[str, float]:
        corrects = np.array(predictions) == np.array(labels)
        # 计算准确率
        accuracy = np.mean(corrects)
        # 返回准确率
        return {'accuracy': accuracy}
    
    def eval(self) -> None:        
        if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
            kwargs = {"trust_remote_code": True}
        else:
            kwargs = {}

        dataset = load_dataset(
            path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
            name=self.eval_args.lang,
            cache_dir=self.model_args.cache_dir,
            download_mode=self.eval_args.download_mode,
            token=self.model_args.hf_hub_token,
            **kwargs,
        )

        inputs, outputs, labels = [], [], []
        for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
            support_set = (
                dataset["train"].shuffle(seed=self.eval_args.seed).select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
            )
            messages = self.eval_template.format_example(
                target_data=dataset[self.data_args.split][i],
                support_set=support_set,
            )
            input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
            inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
            labels.append(messages[-1]["content"])

        for i in trange(
            0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
        ):
            batch_input = self.tokenizer.pad(
                inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
            ).to(self.model.device)
            preds = self.batch_inference(batch_input)
            outputs += preds

        # 确保 outputs 和 labels 的长度相同
        assert len(outputs) == len(labels)
        # 创建一个空的结果列表
        results = []
        # 遍历 outputs 和 labels，将每一对 prediction 和 reference 打包为一个字典，然后添加到结果列表中
        for output, label in zip(outputs, labels):
            results.append({"prediction": output, "reference": label})

        result_prefix = self.eval_args.eval_template + '_' + self.eval_args.lang
        # metrics_results = self._calculate_metrics(predictions=outputs, labels=labels)
        metrics_results = {'accuracy': np.mean(np.array(outputs) == np.array(labels))}
        self._save_results(results=results, metric_results=metrics_results, results_prefix=result_prefix)

class GenerationEvaluator(Evaluator):

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args)
        self.eval_template = get_eval_template(self.eval_args.eval_template if self.eval_args.eval_template is not None else self.eval_args.lang)
        # default generation config from model directory
        if self.eval_args.generation_config is not None:
            self.genration_config = GenerationConfig.from_dict(self.eval_args.generation_config)
    
    @torch.inference_mode()
    def batch_generation(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        generate_params = {
            "input_ids": batch_input.input_ids,
            "attention_mask": batch_input.attention_mask,
            "eos_token_id": [self.tokenizer.eos_token_id],
            # "eos_token_id": [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids(["<0x0A>"])[0]],
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            # "output_scores": True,
            # "max_length": max_length,
            # "max_new_tokens": 512,
        }
        if self.eval_args.generation_config is not None:
            generate_params["generation_config"] = self.genration_config
        
        generation_output = self.model.generate(**generate_params).sequences
        # lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        input_lengths = [len(batch_input.input_ids[k]) for k in range(len(batch_input.input_ids))]
        
        res = []
        for k in range(len(generation_output)):
            # res.append(_del_endswith_none(self.tokenizer.decode(generation_output[k][lengths[k]:], skip_special_tokens=True).strip()))
            prediction = self.tokenizer.decode(generation_output[k][input_lengths[k]:], skip_special_tokens=True).strip()
            if self.eval_template.force_decoder_prefix is not None:
                prediction = prediction.replace(self.eval_template.force_decoder_prefix, "")
            res.append(prediction)
        return res

class MMTEvaluator(GenerationEvaluator):

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args)
        self.bleu = BLEU(tokenize='flores101')
        self.comet_model = load_from_checkpoint(COMET_DIR)
        # self.bleurt_model = score.BleurtScorer(BLEURT_CKPT)
    
    def eval(self) -> None:

        if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
            kwargs = {"trust_remote_code": True}
        else:
            kwargs = {}

        dataset = load_dataset(
            path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
            name=self.eval_args.lang_pair,
            cache_dir=self.model_args.cache_dir,
            download_mode=self.eval_args.download_mode,
            token=self.model_args.hf_hub_token,
            **kwargs,
        )

        inputs, outputs, labels, src_sents = [], [], [], []
        for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
            support_set = (
                dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
            )
            
            messages = self.eval_template.format_example(
                target_data=dataset[self.data_args.split][i],
                support_set=support_set,
                lang_pair=self.eval_args.lang_pair
            )

            input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
            inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
            # labels.append(messages[-1]["content"])
            labels.append(dataset[self.data_args.split][i]['output'])
            src_sents.append(dataset[self.data_args.split][i]['input'])

        for i in trange(
            0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
        ):
            batch_input = self.tokenizer.pad(
                inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
            ).to(self.model.device)
            preds = self.batch_generation(batch_input)
            outputs += preds

        # 确保 outputs 和 labels 的长度相同
        assert len(outputs) == len(labels)
        # 创建一个空的结果列表
        results = []
        # 遍历 outputs 和 labels，将每一对 prediction 和 reference 打包为一个字典，然后添加到结果列表中
        for output, label in zip(outputs, labels):
            results.append({"prediction": output, "reference": label})

        result_prefix = self.eval_args.eval_template + '_' + self.eval_args.lang_pair
        metrics_results = self._calculate_metrics(hypotheses=outputs, labels=labels, source_sentences=src_sents)
        self._save_results(results=results, metric_results=metrics_results, results_prefix=result_prefix)

    def _calculate_metrics(self, hypotheses: List[str], labels: List[str], source_sentences: List[str]) -> Dict[str, float]:
        bleu_score = self.bleu.corpus_score(hypotheses, [labels]).score
        
        # Data must be in the following format:
        def _wrap_as_list_of_dict(src, ref, mt):
            """将src、ref和mt三个list包装成list of dict形式。

            Args:
                src: 一个包含源语言句子的list。
                ref: 一个包含参考译文的list。
                mt: 一个包含机器翻译结果的list。

            Returns:
                一个list of dict，其中每个dict里键值对是src以及src对应的值、ref以及ref对应的值、mt以及mt对应的值。
            """

            # 检查三个list的长度是否相等
            if len(src) != len(ref) or len(src) != len(mt):
                raise ValueError("三个list的长度必须相等。")

            # 创建一个list of dict
            list_of_dict = []

            # 将三个list中的数据一一对应地添加到dict中
            for i in range(len(src)):
                dict = {
                "src": src[i],
                "ref": ref[i],
                "mt": mt[i]
                }
                list_of_dict.append(dict)

            return list_of_dict
        
        wrapped_data = _wrap_as_list_of_dict(source_sentences, labels, hypotheses)
        # Call predict method:
        comet_score = self.comet_model.predict(wrapped_data, batch_size=64, gpus=1).system_score
        # total bleurt score
        # bleurt_score_total = self.bleurt_model.score(references=labels, candidates=hypotheses)
        # system bleurt score
        # bleurt_score = np.mean(bleurt_score_total)
        # score dict
        score_dict = {
            'bleu_score': bleu_score,
            'comet_score': comet_score,
            # 'bleurt_score': bleurt_score
        }
        return score_dict


# if __name__ == "__main__":
#     evaluator = Evaluator()
#     evaluator.eval()
