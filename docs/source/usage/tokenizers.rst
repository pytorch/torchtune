.. _tokenizers_usage_label:

==========
Tokenizers
==========

Snappy intro about tokenizers in general, how they're used in torchtune


Downloading tokenizers from Hugging Face
----------------------------------------

tune download


Loading tokenizers from file
----------------------------

Passing path into tokenizer builder function


Base tokenizers
---------------

SentencePiece, TikToken, and how this is distinct from model tokenizers


Model tokenizers
----------------

Different contract, must implement tokenize_messages, model specific special tokens, and more.


Special tokens
--------------

What are special tokens, how is it different from a template, how can I customize special tokens


Prompt templates
----------------

What are prompt templates, when should I use them, how can I configure them in tokenizer, how can I easily carry this over to inference


Setting max sequence length
---------------------------

Passing this in tokenizer builder
