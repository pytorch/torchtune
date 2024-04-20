=================================
Fine-tuning LLaMA3 with chat data
=================================

In this tutorial, we'll demystify what prompt templates are and when you'll need them
and talk about the differences between prompt templating for LLaMA2 and LLaMA3. Then,
we'll wrap up with a LLaMA3 finetuning example on a custom chat dataset.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn:

      * What is a prompt template and how is it different from special tokens
      * When to use and not to use a prompt template
      * How to use your own chat dataset to fine-tune LLaMA3

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with :ref:`configuring datasets<dataset_tutorial_label>`
      * Know how to :ref:`download LLaMA3 weights <llama3_label>`


Prompt templates & special tokens
---------------------------------

Prompt templates and tokenization schemes are often conflated, and it's sometimes
unclear what you need to do to format your data to optimize the training performance
of your model. Let's walk through the LLaMA2/LLaMA3 templates to better understand
the distinction.

In torchtune, a prompt template is simply a structured way to format your prompt
into the model. The prompt template adds flavor text to prime the model to expect
a certain structure when it is used for inference. From the `official LLaMA2 prompt
template guide <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2>`_
for the LLaMA2 chat model, we can see that special tags are added:

.. code-block:: text

    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
    <s>[INST] {{ user_message_2 }} [/INST]

Since the chat model was pretrained with this prompt template, if you want to run
inference on the model, you'll need to use the same template for optimal performance
on chat data. Otherwise, the model will just perform standard text completion, which
may or may not align with your intended use case.

If you look at our :class:`~torchtune.data.Llama2ChatFormat` class, you'll notice that
we don't include the :code:`<s>` and :code:`</s>` tokens. These are the beginning-of-sequence
(BOS) and end-of-sequence (EOS) tokens that are represented differently in the tokenizer
than the rest of the prompt template. Let's tokenize some examples with the
:class:`~torchtune.modules.tokenizers.SentencePieceTokenizer` used by LLaMA2 to see
why.

.. code-block:: python

    from torchtune.modules.tokenizers import SentencePieceTokenizer

    tokenizer = SentencePieceTokenizer("/tmp/Llama-2-7b-hf/tokenizer.model")
    text = "[INST] <<SYS>> Hello"
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    print(tokens)
    # [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 15043, 2]

We've added the BOS and EOS tokens when encoding our example text. This shows up
as IDs 1 and 2. We can verify that these are our BOS and EOS tokens.

.. code-block:: python

    print(tokenizer.spm_model.piece_to_id("<s>"))
    # 1
    print(tokenizer.spm_model.piece_to_id("</s>"))
    # 2

The BOS and EOS tokens are what we call special tokens, because they have their own
reserved token IDs. This means that they will index to their own individual vectors in
the model's learnt embedding table. The rest of the prompt template tags, :code:`[INST]`
and :code:`<<SYS>>` are tokenized as normal text and not their own IDs.

.. code-block:: python

    print(tokenizer.decode(518))
    # '['
    print(tokenizer.decode(25580))
    # 'INST'
    print(tokenizer.decode(29962))
    # ']'

It's important to note that you should not place the special reserved tokens in your
input prompts manually, as it will be treated as normal text and not as a special
token.

.. code-block:: python

    print(tokenizer.encode("<s>", add_bos=False, add_eos=False))
    # [529, 29879, 29958]

When should I use a prompt template?
------------------------------------

When to use or not to use a prompt template is governed by what your desired inference
behavior is. You should use a prompt template in the following scenarios:

1. You are running inference on the base model and it was pre-trained with a prompt
template
2. You want to prime a fine-tuned model to expect a certain prompt structure on inference
for a specific task

It is not strictly necessary to fine-tune with a prompt template, but generally you
want the model to perform some sort of task, which will require some formatting of
the prompt.

For example, the :class:`~torchtune.data.SummarizeTemplate` provides a lightweight
structure to prime your fine-tuned model for prompts asking to summarize text.

.. code-block:: python

    f"Summarize this dialogue:\n{dialogue}\n---\nSummary:\n"

You can fine-tune LLaMA2 with this template even though the model was originally pre-trained
with the :class:`~torchtune.data.Llama2ChatFormat`, as long as this is what the model
sees during inference. The model should be robust enough to adapt to a new template.

Special tokens in LLaMA3
------------------------

LLaMA3 `overhauled <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3>`_
the special tokens and prompt templating from LLaMA2 to better support multiturn conversations.

.. code-block:: text

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {{ user_message_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {{ model_answer_1 }}<|eot_id|><|start_header_id|>user<|end_header_id|>

Now, all the additional tags are special tokens, and no prompt template is required.
You can see how they get encoded as their own token IDs below:

.. code-block:: python

    from torchtune.modules.tokenizers import TikTokenTokenizer

    tokenizer = TikTokenTokenizer("/tmp/Meta-Llama-3-8B/original/tokenizer.model")
    print(tokenizer._encode_special_token("<|begin_of_text|>"))
    # 128000
    print(tokenizer._encode_special_token("<|eot_id|>"))
    # 128009

The best part is - all these special tokens are handled purely by the tokenizer.
That means you won't have to worry about messing up any required prompt templates!

Fine-tuning on a custom chat dataset
------------------------------------

Let's test our understanding by trying to fine-tune the LLaMA3-8B model with a custom
chat dataset. We'll walk through how to set up our data so that it can be tokenized
correctly and fed into our model.
