from typing import Union, Optional, Dict, Any, Callable, Mapping, Tuple
from functools import partial
from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer

from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.modules.transforms import Transform
import logging

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger("WT")
SPACE = " "
STAR = "*"
BLANK = ""


class __InceptionToMessages(Transform):

    def __init__(self, train_on_input: bool = True, column_map: Optional[Dict[str, str]] = None) -> None:
        self.train_on_input = train_on_input
        self.column_map = column_map
        self.template = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }

    @staticmethod
    def __get_instruction_and_input(_source: str) -> Tuple[Optional[str], Optional[str]]:
        instruction_text = None
        input_text = None
        is_input = False

        source = str(_source)
        ins_start_text = None
        if '### Instruction :' in source:
            ins_start_text = '### Instruction :'
        elif '### Instruction:' in source:
            ins_start_text = '### Instruction:'

        ins_end_text = None
        if '### Input:' in source:
            ins_end_text = '### Input:'
            is_input = True
        elif '### Input :' in source:
            ins_end_text = '### Input :'
            is_input = True
        elif '### Response :' in source:
            ins_end_text = '### Response :'
        elif '### Response:' in source:
            ins_end_text = '### Response:'

        if ins_start_text is not None and ins_end_text is not None:
            instruction_text = str(
                source[(source.index(ins_start_text) + len(ins_start_text)): source.rindex(ins_end_text)])
            for i in range(3):
                instruction_text = instruction_text.replace(SPACE + SPACE, SPACE).strip("\n").strip()

        inp_end_text = None
        if is_input:
            inp_start_text = ins_end_text
            if '### Response :' in source:
                inp_end_text = '### Response :'
            elif '### Response:' in source:
                inp_end_text = '### Response:'

            if inp_start_text is not None and inp_end_text is not None:
                input_text = str(
                    source[(source.index(inp_start_text) + len(inp_start_text)): source.rindex(inp_end_text)])
                for i in range(3):
                    input_text = input_text.replace(SPACE + SPACE, SPACE).strip("\n").strip()

        return input_text, instruction_text

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:

        column_map = self.column_map or {}
        key_source = column_map.get("source", "source")
        key_target = column_map.get("target", "target")

        source = str(sample[key_source]).strip()
        target = str(sample[key_target]).strip().replace(STAR + STAR, BLANK).replace(SPACE + SPACE, SPACE)
        input_text, instruction_text = self.__get_instruction_and_input(source)

        # print(f"source: {source}")
        # print(f"target: {target}")
        # print(f"input_text: {input_text}")
        # print(f"instruction_text: {instruction_text}")

        if input_text:
            prompt = self.template["prompt_input"].format(instruction=instruction_text, input=input_text)
        else:
            prompt = self.template["prompt_no_input"].format(instruction=instruction_text)

        messages = [
            Message(
                role="user",
                content=prompt,
                masked=not self.train_on_input,
                eot=True,
            ),
            Message(
                role="assistant",
                content=target,
                masked=False,
                eot=True,
            ),
        ]

        # print(messages)

        return {"messages": messages}


def __inception_base(
        tokenizer: ModelTokenizer,
        *,
        source: str = "json",
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = True,
        packed: bool = False,
        filter_fn: Optional[Callable] = None,
        split: str = "train",
        **load_dataset_kwargs: Dict[str, Any]
) -> Union[SFTDataset, PackedDataset]:
    message_transform = __InceptionToMessages(
        train_on_input=train_on_input, column_map=column_map
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds


inc_ar_hc3 = partial(__inception_base, name='inc_ar_hc3',
                     data_files=['/project/audio/data/llm/jais/Ar_HC3.jsonl'])
inc_ar_botim_qa = partial(__inception_base, name='inc_ar_botim_qa',
                          data_files=['/project/audio/data/llm/jais/Ar_IIAI_Botim_QA.jsonl'])
inc_ar_alpaca_manual = partial(__inception_base, name='inc_ar_alpaca_manual',
                               data_files=['/project/audio/data/llm/jais/Ar_alpaca_manual.jsonl'])
inc_ar_baize_mult_turn = partial(__inception_base, name='inc_ar_baize_mult_turn',
                                 data_files=['/project/audio/data/llm/jais/Ar_baize_mult_turn.jsonl'])
inc_ar_dolly_15k = partial(__inception_base, name='inc_ar_dolly_15k',
                           data_files=['/project/audio/data/llm/jais/Ar_dolly_15k.jsonl'])
inc_ar_hh_rlhf_mult_turn = partial(__inception_base, name='inc_ar_hh_rlhf_mult_turn',
                                   data_files=['/project/audio/data/llm/jais/Ar_hh-rlhf_mult_turn.jsonl'])
inc_ar_internal = partial(__inception_base, name='inc_ar_internal',
                          data_files=['/project/audio/data/llm/jais/Ar_iiai_prepared.jsonl'])
inc_ar_natural_question = partial(__inception_base, name='inc_ar_natural_question',
                                  data_files=['/project/audio/data/llm/jais/Ar_natural_question.jsonl'])
inc_ar_supernatural = partial(__inception_base, name='inc_ar_supernatural',
                              data_files=['/project/audio/data/llm/jais/Ar_supernatural.jsonl'])
inc_ar_unnatural = partial(__inception_base, name='inc_ar_unnatural',
                           data_files=['/project/audio/data/llm/jais/Ar_unnatural.jsonl'])
inc_ar_cahya = partial(__inception_base, name='inc_ar_cahya',
                       data_files=['/project/audio/data/llm/jais/ar_cahya.jsonl'])
inc_ar_climate_chatgpt = partial(__inception_base, name='inc_ar_climate_chatgpt',
                                 data_files=['/project/audio/data/llm/jais/ar_climate_chatgpt.jsonl'])
inc_ar_instruct_wild = partial(__inception_base, name='inc_ar_instruct_wild',
                               data_files=['/project/audio/data/llm/jais/ar_instruct_wild.jsonl'])
inc_ar_lmsys = partial(__inception_base, name='inc_ar_lmsys',
                       data_files=['/project/audio/data/llm/jais/ar_lmsys.jsonl'])
inc_ar_bactrian = partial(__inception_base, name='inc_ar_bactrian',
                          data_files=['/project/audio/data/llm/jais/bactrian_ar.jsonl'])
inc_ar_raft = partial(__inception_base, name='inc_ar_raft',
                      data_files=['/project/audio/data/llm/jais/raft_ar.jsonl'])
inc_ar_school_hack = partial(__inception_base, name='inc_ar_school_hack',
                             data_files=['/project/audio/data/llm/jais/school_hack_ar.jsonl'])
inc_ar_en_school_hack = partial(__inception_base, name='inc_ar_en_school_hack',
                                data_files=['/project/audio/data/llm/jais/school_hack_en_ar.jsonl'])

# def main():
#     from torch.utils.data import DataLoader
#     from torchtune.models.llama3 import Llama3Tokenizer
#
#     m_tokenizer = Llama3Tokenizer(path="/workspace/models/Meta-Llama-3.1-70B-Instruct/original/tokenizer.model")
#     ds = inc_ar_hh_rlhf_mult_turn(tokenizer=m_tokenizer)
#
#     for batch in DataLoader(ds, batch_size=64, collate_fn=lambda x: x):
#         print(f"Batch Size: {len(batch)}")
#
#
# if __name__ == '__main__':
#     main()
