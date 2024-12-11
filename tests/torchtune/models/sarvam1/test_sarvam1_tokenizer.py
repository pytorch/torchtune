import sentencepiece as spm
from transformers import AutoTokenizer, LlamaTokenizer
from torchtune.models.sarvam1 import sarvam1_tokenizer
from tqdm import tqdm

sp_tokenizer = spm.SentencePieceProcessor(model_file='/home/rahul_sarvam_ai/nemo_models/original_tokenizer/tokenizer.model')
hf_tokenizer = LlamaTokenizer.from_pretrained('sarvam/sarvam-1-nemo-sft')
# hf_tokenizer = LlamaTokenizer.from_pretrained('sarvam/sarvam-2b-sft', legacy=False, split_special_tokens=True)
tt_tokenizer = sarvam1_tokenizer(path='/home/rahul_sarvam_ai/nemo_models/original_tokenizer/tokenizer.model')

import datasets
indic_ds = datasets.load_dataset('sarvam/indic-sft-dataset-sample-sep2024', split='train')
en_ds = datasets.load_dataset('sarvam/claude-generated-sft', 'indic-culture', split='train')
chat_ds = datasets.load_dataset('sarvam/norobots_sft', split='train')

indic_ds = indic_ds.map(lambda x: {'text': x['query'] + '\n' + x['response']}, remove_columns=indic_ds.column_names)
en_ds = en_ds.map(lambda x: {'text': x['question'] + '\n' + x['answer']}, remove_columns=en_ds.column_names)
chat_ds = chat_ds.map(lambda x: {'text': hf_tokenizer.apply_chat_template([x['messages']], tokenize=False)[0]}, remove_columns=chat_ds.column_names)

ds = datasets.concatenate_datasets([indic_ds, en_ds])
for d in tqdm(ds):
    sp_output = sp_tokenizer.encode(d['text'])
    hf_output = hf_tokenizer.encode(d['text'], add_special_tokens=False)
    tt_output = tt_tokenizer.encode(d['text'])
    assert sp_output == hf_output, (f"Mismatch for {d['text']}\n{sp_output}\n{hf_output}")
    assert sp_output == tt_output, (f"Mismatch for {d['text']}\n{sp_output}\n{tt_output}")

for d in tqdm(chat_ds):
    sp_output = sp_tokenizer.encode(d['text'])
    hf_output = hf_tokenizer.encode(d['text'], add_special_tokens=False)
    tt_output = tt_tokenizer.encode(d['text'])
    assert sp_output == hf_output, (f"Mismatch for {d['text']}\n{sp_output}\n{hf_output}")
    assert sp_output == tt_output, (f"Mismatch for {d['text']}\n{sp_output}\n{tt_output}")
