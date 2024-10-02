.. _chat_tutorial_label:

=================================
Fine-Tuning Llama3 with Chat Data
=================================

Llama3 Instruct introduced a new prompt template for fine-tuning with chat data. In this tutorial,
we'll cover what you need to know to get you quickly started on preparing your own
custom chat dataset for fine-tuning Llama3 Instruct.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn:

      * How the Llama3 Instruct format differs from Llama2
      * All about prompt templates and special tokens
      * How to use your own chat dataset to fine-tune Llama3 Instruct

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with :ref:`configuring datasets<dataset_tutorial_label>`
      * Know how to :ref:`download Llama3 Instruct weights <llama3_label>`


Template changes from Llama2 to Llama3
--------------------------------------

The Llama2 chat model requires a specific template when prompting the pre-trained
model. Since the chat model was pretrained with this prompt template, if you want to run
inference on the model, you'll need to use the same template for optimal performance
on chat data. Otherwise, the model will just perform standard text completion, which
may or may not align with your intended use case.

From the `official Llama2 prompt
template guide <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2>`_
for the Llama2 chat model, we can see that special tags are added:

.. code-block:: text

    <s>[INST] <<SYS>>
    You are a helpful, respectful, and honest assistant.
    <</SYS>>

    Hi! I am a human. [/INST] Hello there! Nice to meet you! I'm Meta AI, your friendly AI assistant </s>

Llama3 Instruct `overhauled <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3>`_
the template from Llama2 to better support multiturn conversations. The same text
in the Llama3 Instruct format would look like this:

.. code-block:: text

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful, respectful, and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

    Hi! I am a human.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Hello there! Nice to meet you! I'm Meta AI, your friendly AI assistant<|eot_id|>

The tags are entirely different, and they are actually encoded differently than in
Llama2. Let's walk through tokenizing an example with the Llama2 template and the
Llama3 template to understand how.

.. note::
    The Llama3 Base model uses a `different prompt template
    <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3>`_ than Llama3 Instruct
    because it has not yet been instruct tuned and the extra special tokens are untrained. If you
    are running inference on the Llama3 Base model without fine-tuning we recommend the base
    template for optimal performance. Generally, for instruct and chat data, we recommend using
    Llama3 Instruct with its prompt template. The rest of this tutorial assumes you are using
    Llama3 Instruct.

.. _prompt_template_vs_special_tokens:

Tokenizing prompt templates & special tokens
--------------------------------------------

Let's say I have a sample of a single user-assistant turn accompanied with a system
prompt:

.. code-block:: python

    sample = [
        {
            "role": "system",
            "content": "You are a helpful, respectful, and honest assistant.",
        },
        {
            "role": "user",
            "content": "Who are the most influential hip-hop artists of all time?",
        },
        {
            "role": "assistant",
            "content": "Here is a list of some of the most influential hip-hop "
            "artists of all time: 2Pac, Rakim, N.W.A., Run-D.M.C., and Nas.",
        },
    ]

Now, let's format this with the :class:`~torchtune.models.llama2.Llama2ChatTemplate` class and
see how it gets tokenized. The Llama2ChatTemplate is an example of a **prompt template**,
which simply structures a prompt with flavor text to indicate a certain task.

.. code-block:: python

    from torchtune.data import Llama2ChatTemplate, Message

    messages = [Message.from_dict(msg) for msg in sample]
    formatted_messages = Llama2ChatTemplate.format(messages)
    print(formatted_messages)
    # [
    #     Message(
    #         role='user',
    #         content='[INST] <<SYS>>\nYou are a helpful, respectful, and honest assistant.\n<</SYS>>\n\nWho are the most influential hip-hop artists of all time? [/INST] ',
    #         ...,
    #     ),
    #     Message(
    #         role='assistant',
    #         content='Here is a list of some of the most influential hip-hop artists of all time: 2Pac, Rakim, N.W.A., Run-D.M.C., and Nas.',
    #         ...,
    #     ),
    # ]

There are also special tokens used by Llama2, which are not in the prompt template.
If you look at our :class:`~torchtune.models.llama2.Llama2ChatTemplate` class, you'll notice that
we don't include the :code:`<s>` and :code:`</s>` tokens. These are the beginning-of-sequence
(BOS) and end-of-sequence (EOS) tokens that are represented differently in the tokenizer
than the rest of the prompt template. Let's tokenize this example with the
:func:`~torchtune.models.llama2.llama2_tokenizer` used by Llama2 to see
why.

.. code-block:: python

    from torchtune.models.llama2 import llama2_tokenizer

    tokenizer = llama2_tokenizer("/tmp/Llama-2-7b-hf/tokenizer.model")
    user_message = formatted_messages[0].text_content
    tokens = tokenizer.encode(user_message, add_bos=True, add_eos=True)
    print(tokens)
    # [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, ..., 2]

We've added the BOS and EOS tokens when encoding our example text. This shows up
as IDs 1 and 2. We can verify that these are our BOS and EOS tokens.

.. code-block:: python

    print(tokenizer._spm_model.spm_model.piece_to_id("<s>"))
    # 1
    print(tokenizer._spm_model.spm_model.piece_to_id("</s>"))
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
    print(tokenizer.decode([3532, 14816, 29903, 6778]))
    # '<<SYS>>'

It's important to note that you should not place the special reserved tokens in your
input prompts manually, as it will be treated as normal text and not as a special
token.

.. code-block:: python

    print(tokenizer.encode("<s>", add_bos=False, add_eos=False))
    # [529, 29879, 29958]

Now let's take a look at Llama3's formatting to see how it's tokenized differently
than Llama2.

.. code-block:: python

    from torchtune.models.llama3 import llama3_tokenizer

    tokenizer = llama3_tokenizer("/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model")
    messages = [Message.from_dict(msg) for msg in sample]
    tokens, mask = tokenizer.tokenize_messages(messages)
    print(tokenizer.decode(tokens))
    # '<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful,
    # and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho
    # are the most influential hip-hop artists of all time?<|eot_id|><|start_header_id|>
    # assistant<|end_header_id|>\n\nHere is a list of some of the most influential hip-hop
    # artists of all time: 2Pac, Rakim, N.W.A., Run-D.M.C., and Nas.<|eot_id|>'

.. note::
    We used the ``tokenize_messages`` API for Llama3, which is different than
    encode. It simply manages adding all the special tokens in the correct
    places after encoding the individual messages.

We can see that the tokenizer handled all the formatting without us specifying a prompt
template. It turns out that all of the additional tags are special tokens, and we don't require
a separate prompt template. We can verify this by checking if the tags get encoded
as their own token IDs.

.. code-block:: python

    print(tokenizer.special_tokens["<|begin_of_text|>"])
    # 128000
    print(tokenizer.special_tokens["<|eot_id|>"])
    # 128009

The best part is - all these special tokens are handled purely by the tokenizer.
That means you won't have to worry about messing up any required prompt templates!


When should I use a prompt template?
------------------------------------

Whether or not to use a prompt template is governed by what your desired inference
behavior is. You should use a prompt template if you are running inference on the
base model and it was pre-trained with a prompt template, or you want to prime a
fine-tuned model to expect a certain prompt structure on inference for a specific task.

It is not strictly necessary to fine-tune with a prompt template, but generally
specific tasks will require specific templates. For example, the :class:`~torchtune.data.SummarizeTemplate`
provides a lightweight structure to prime your fine-tuned model for prompts asking to summarize text.
This would wrap around the user message, with the assistant message untouched.

.. code-block:: python

    f"Summarize this dialogue:\n{dialogue}\n---\nSummary:\n"

You can fine-tune Llama2 with this template even though the model was originally pre-trained
with the :class:`~torchtune.models.llama2.Llama2ChatTemplate`, as long as this is what the model
sees during inference. The model should be robust enough to adapt to a new template.


Fine-tuning on a custom chat dataset
------------------------------------

Let's test our understanding by trying to fine-tune the Llama3-8B instruct model with a custom
chat dataset. We'll walk through how to set up our data so that it can be tokenized
correctly and fed into our model.

Let's say we have a local dataset saved as a JSON file that contains conversations
with an AI model. How can we get something like this into a format
Llama3 understands and tokenizes correctly?

.. code-block:: python

    # data/my_data.json
    [
        {
            "dialogue": [
                {
                    "from": "human",
                    "value": "What is your name?"
                },
                {
                    "from": "gpt",
                    "value": "I am an AI assistant, I don't have a name."
                },
                {
                    "from": "human",
                    "value": "Pretend you have a name."
                },
                {
                    "from": "gpt",
                    "value": "My name is Mark Zuckerberg."
                }
            ]
        },
    ]

Let's first take a look at the :ref:`dataset_builders` and see which fits our use case. Since we
have conversational data, :func:`~torchtune.datasets.chat_dataset` seems to be a good fit. For any
custom local dataset we always need to specify ``source``, ``data_files``, and ``split`` for any dataset
builder in torchtune. For :func:`~torchtune.datasets.chat_dataset`, we additionally need to specify
``conversation_column`` and ``conversation_style``. Our data follows the ``"sharegpt"`` format, so
we can specify that here. Altogether, our :func:`~torchtune.datasets.chat_dataset` call should
look like so:

.. code-block:: python

    from torchtune.datasets import chat_dataset
    from torchtune.models.llama3 import llama3_tokenizer

    tokenizer = llama3_tokenizer("/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model")
    ds = chat_dataset(
        tokenizer=tokenizer,
        source="json",
        data_files="data/my_data.json",
        split="train",
        conversation_column="dialogue",
        conversation_style="sharegpt",
    )

.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.llama3.llama3_tokenizer
      path: /tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model

    dataset:
      _component_: torchtune.datasets.chat_dataset
      source: json
      data_files: data/my_data.json
      split: train
      conversation_column: dialogue
      conversation_style: sharegpt

.. note::
    You can pass in any keyword argument for `load_dataset <https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/loading_methods#datasets.load_dataset>`_ into all our
    Dataset classes and they will honor them. This is useful for common parameters
    such as specifying the data split with :code:`split` or configuration with
    :code:`name`

If you needed to add a prompt template, you would simply pass it into the tokenizer.
Since we're fine-tuning Llama3, the tokenizer will handle all formatting for
us and prompt templates are optional. Other models such as Mistral's :class:`~torchtune.models.mistral._tokenizer.MistralTokenizer`,
use a chat template by default (:class:`~torchtune.models.mistral.MistralChatTemplate`) to format
all messages according to their `recommendations <https://docs.mistral.ai/getting-started/open_weight_models/#chat-template>`_.

Now we're ready to start fine-tuning! We'll use the built-in LoRA single device recipe.
Use the :ref:`tune cp <tune_cp_cli_label>` command to get a copy of the :code:`8B_lora_single_device.yaml`
config and update it with your dataset configuration.

Launch the fine-tune!

.. code-block:: bash

    $ tune run lora_finetune_single_device --config custom_8B_lora_single_device.yaml epochs=15
