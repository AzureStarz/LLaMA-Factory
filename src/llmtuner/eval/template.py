from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..data import Role
from ..extras.constants import CHOICES

@dataclass
class EvalTemplate:
    system: str
    answer: str
    
    def _format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        for k in range(len(support_set)):
            prompt, response = self._parse_example(support_set[k])
            messages.append({"role": Role.USER.value, "content": prompt})
            messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data)
        messages.append({"role": Role.USER.value, "content": prompt})
        messages.append({"role": Role.ASSISTANT.value, "content": response})
        messages[0]["content"] = self.system + messages[0]["content"]
        return messages

@dataclass
class MultipleChoiceTemplate(EvalTemplate):
    choice: str
    prefix: str

@dataclass
class MMLUEvalTemplate(MultipleChoiceTemplate):

    def _parse_example(self, example: Dict[str, str]) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
        output: a tuple of (prompt, response)
        """
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], subject_name: str
    ) -> List[Dict[str, str]]:
        r"""
        here template attribute denotes the subject
        Converts dataset examples to messages.
        """
        self.system = self.system.format(subject=subject_name)
        return self._format_example(target_data, support_set)

@dataclass
class MRCEvalTemplate(MultipleChoiceTemplate):
    passage: str
    question: str
    
    def _parse_example(self, example: Dict[str, str]) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"passage", "question", "A", "B", "C", "D", "answer"}
        output: a tuple of (prompt, response)
        """
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        passage = self.passage.format(passage=example["passage"])
        question = self.question.format(question=example["question"])
        candidates = ["###\nChoices:"] + candidates
        return "".join([passage] + [question] + candidates + [self.answer]), example["answer"]
    
    def format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        r"""
        here template attribute denotes the subject
        Converts dataset examples to messages.
        """
        return self._format_example(target_data, support_set)

@dataclass
class MMTEvalTemplate(EvalTemplate):
    force_decoder_prefix: str
    flores_to_language_mapping = {'ces_Latn': 'Czech', 'dan_Latn': 'Danish', 'ukr_Cyrl': 'Ukrainian', 'bul_Cyrl': 'Bulgarian', 'fin_Latn': 'Finnish', 'hun_Latn': 'Hungarian', 
                'nob_Latn': 'Norwegian', 'ind_Latn': 'Indonesian', 'jpn_Jpan': 'Japanese', 'kor_Hang': 'Korean', 'por_Latn': 'Portuguese', 'slv_Latn': 'Slovenian', 
                'vie_Latn': 'Vietnamese', 'pol_Latn': 'Polish', 'eng_Latn': "English"}
    
    def _parse_example(self, example: Dict[str, str]) -> Tuple[str]:
        r"""
        input: a dict with keys {"source", "reference"}
        output: a tuple of (prompt, response)
        """
        return example["input"] + self.answer, self.force_decoder_prefix + example["output"]
    
    def format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], lang_pair: str
    ) -> List[Dict[str, str]]:
        r"""
        here template attribute denotes the language pair
        Converts dataset examples to messages.
        """
        src, tgt = self.flores_to_language_mapping[lang_pair.split("_to_")[0]], self.flores_to_language_mapping[lang_pair.split("_to_")[1]]
        self.answer = self.answer.format(tgt=tgt)
        self.system = self.system.format(src=src, tgt=tgt)
        return self._format_example(target_data, support_set)

@dataclass
class ATSEvalTemplate(EvalTemplate):
    passage: str
    
    def _parse_example(self, example: Dict[str, str]) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"passage", "summary"}
        output: a tuple of (prompt, response)
        """
        passage = self.passage.format(passage=example["passage"])
        return passage + self.answer, example["summary"]
    
    def format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        r"""
        here template attribute denotes the language pair
        Converts dataset examples to messages.
        """
        return self._format_example(target_data, support_set)

@dataclass
class NLIEvalTemplate(EvalTemplate):
    question: str
    mapping = {'entailment': 'True', 'contradiction': 'False', 'neutral': 'Neither'}
    
    def _parse_example(self, example: Dict[str, str]) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"premise", "hypothesis", "label"}
        output: a tuple of (prompt, response)
        """
        question = self.question.format(question=example["hypothesis"])
        return example["premise"] + question + self.answer, self.mapping[example["label"]]
    
    def format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        r"""
        here template attribute denotes the language pair
        Converts dataset examples to messages.
        """
        return self._format_example(target_data, support_set)

eval_templates: Dict[str, "EvalTemplate"] = {}

def _register_mmlu_eval_template(name: str, system: str, choice: str, answer: str, prefix: str) -> None:
    eval_templates[name] = MMLUEvalTemplate(system=system, choice=choice, answer=answer, prefix=prefix)

def _register_multilingual_machine_translation_eval_template(name: str, system: str, answer: str, force_decoder_prefix: str) -> None:
    eval_templates[name] = MMTEvalTemplate(system=system, answer=answer, force_decoder_prefix=force_decoder_prefix)

def _register_machine_reading_comprehension_eval_template(name: str, system: str, choice: str, answer: str, passage: str, question: str, prefix: str) -> None:
    eval_templates[name] = MRCEvalTemplate(system=system, choice=choice, answer=answer, passage=passage, question=question, prefix=prefix)

def _register_abstractive_text_summarization_eval_template(name: str, system: str, answer: str, passage: str) -> None:
    eval_templates[name] = ATSEvalTemplate(system=system, answer=answer, passage=passage)

def _register_natural_language_inference_eval_template(name: str, system: str, answer: str, question: str) -> None:
    eval_templates[name] = NLIEvalTemplate(system=system, answer=answer, question=question)

def get_eval_template(name: str) -> "EvalTemplate":
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


_register_mmlu_eval_template(
    name="en_mmlu",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)


_register_mmlu_eval_template(
    name="zh_mmlu",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix=" ",
)

_register_multilingual_machine_translation_eval_template(
    name="flores",
    system="You are a machine translation system that translates sentences from {src} to {tgt}. You just respond with the translation, without any additional comments.\n\n",
    answer="\nTranslate to {tgt} ",
    force_decoder_prefix="Suer, here's the translation: ",
)

_register_machine_reading_comprehension_eval_template(
    name="belebele",
    system="Given the following passage, query, and answer choices, output the letter corresponding to the correct answer.\n\n",
    passage="###\nPassage:\n{passage}\n",
    question="###\nQuestion:\n{question}\n",
    choice="\n{choice}. {content}",
    answer="\n###\nAnswer: ",
    prefix=" ",
)

_register_abstractive_text_summarization_eval_template(
    name="xlsum",
    system="You are an NLP assistant whose purpose is to summarize any given article. You should summarize all important information concisely in the same language in which you have been provided the document.\n\n",
    passage="###\nPassage:\n{passage}\n",
    answer="###\nSummary: ",
)

_register_natural_language_inference_eval_template(
    name="xnli",
    system="You are an NLP assistant whose purpose is to solve Natural Language Inference(NLI) problems. \
        NLI is the task of determining the inference relation between two (short, ordered) texts: entailment, contradiction, or neutral. \
            Answer as concisely as possible in the same format as the examples below:",
    question="\nQuestion:\n{question}\n",
    answer="True, False, or Neither?\nAnswer: "
)