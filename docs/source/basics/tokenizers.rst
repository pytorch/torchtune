.. _tokenizers_usage_label:

==========
Tokenizers
==========

Tokenizers are a key component of any LLM. They convert raw text into token IDs, which index into embedding vectors that are
understood by the model.

In torchtune, tokenizers play the role of converting :class:`~torchtune.data.Message` objects into token IDs and any necessary model-specific special tokens.

.. code-block:: python

    from torchtune.data import Message
    from torchtune.models.phi3 import phi3_mini_tokenizer

    sample = {
        "input": "user prompt",
        "output": "model response",
    }

    msgs = [
        Message(role="user", content=sample["input"]),
        Message(role="assistant", content=sample["output"])
    ]

    p_tokenizer = phi3_mini_tokenizer("/tmp/Phi-3-mini-4k-instruct/tokenizer.model")
    tokens, mask = p_tokenizer.tokenize_messages(msgs)
    print(tokens)
    # [1, 32010, 29871, 13, 1792, 9508, 32007, 29871, 13, 32001, 29871, 13, 4299, 2933, 32007, 29871, 13]
    print(p_tokenizer.decode(tokens))
    # '\nuser prompt \n \nmodel response \n'

Model tokenizers are usually based on an underlying byte-pair encoding algorithm, such as SentencePiece or TikToken, which are both
supported in torchtune.

Downloading tokenizers from Hugging Face
----------------------------------------

Models hosted on Hugging Face are also distributed with the tokenizers they were trained with. These are automatically downloaded alongside
model weights when using ``tune download``. For example, this command downloads the Mistral-7B model weights and tokenizer:

.. code-block:: bash

    tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1 --hf-token <HF_TOKEN>
    cd /tmp/Mistral-7B-v0.1/
    ls tokenizer.model
    # tokenizer.model

Loading tokenizers from file
----------------------------

Once you've downloaded the tokenizer file, you can load it into the corresponding tokenizer class by pointing
to the file path of the tokenizer model in your config or in the constructor. You can also pass in a custom file path if you've already
downloaded it to a different location.

.. code-block:: python

    # In code
    from torchtune.models.mistral import mistral_tokenizer

    m_tokenizer = mistral_tokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model")
    type(m_tokenizer)
    # <class 'torchtune.models.mistral._tokenizer.MistralTokenizer'>

.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model

Setting max sequence length
---------------------------

Setting max sequence length can give you control over memory usage and adhere to model specifications.

.. code-block:: python

    # In code
    from torchtune.models.mistral import mistral_tokenizer

    m_tokenizer = mistral_tokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model", max_seq_len=8192)

    # Set an arbitrarily small seq len for demonstration
    from torchtune.data import Message

    m_tokenizer = mistral_tokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model", max_seq_len=7)
    msg = Message(role="user", content="hello world")
    tokens, mask = m_tokenizer.tokenize_messages([msg])
    print(len(tokens))
    # 7
    print(tokens)
    # [1, 733, 16289, 28793, 6312, 28709, 2]
    print(m_tokenizer.decode(tokens))
    # '[INST] hello'


.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model
      max_seq_len: 8192


Prompt templates
----------------

Prompt templates are enabled by passing it into any model tokenizer. See :ref:`prompt_templates_usage_label` for more details.

Special tokens
--------------

Special tokens are model-specific tags that are required to prompt the model. They are different from prompt templates
because they are assigned their own unique token IDs. For an extended discussion on the difference between special tokens
and prompt templates, see :ref:`prompt_templates_usage_label`.

Special tokens are automatically added to your data by the model tokenizer and do not require any additional configuration
from you. You also have the ability to customize the special tokens for experimentation by passing in a file path to
the new special tokens mapping in a JSON file. This will NOT modify the underlying ``tokenizer.model`` to support the new
special token ids - it is your responsibility to ensure that the tokenizer file encodes it correctly. Note also that
some models require the presence of certain special tokens for proper usage, such as the ``"<|eot_id|>"`` in Llama3 Instruct.

For example, here we change the ``"<|begin_of_text|>"`` and ``"<|end_of_text|>"`` token IDs in Llama3 Instruct:

.. code-block:: python

    # tokenizer/special_tokens.json
    {
        "added_tokens": [
            {
                "id": 128257,
                "content": "<|begin_of_text|>",
            },
            {
                "id": 128258,
                "content": "<|end_of_text|>",
            },
            # Remaining required special tokens
            ...
        ]
    }

.. code-block:: python

    # In code
    from torchtune.models.llama3 import llama3_tokenizer

    tokenizer = llama3_tokenizer(
        path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model",
        special_tokens_path="tokenizer/special_tokens.json",
    )
    print(tokenizer.special_tokens)
    # {'<|begin_of_text|>': 128257, '<|end_of_text|>': 128258, ...}

.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.llama3.llama3_tokenizer
      path: /tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model
      special_tokens_path: tokenizer/special_tokens.json

.. _base_tokenizers:

Base tokenizers
---------------

:class:`~torchtune.modules.tokenizers.BaseTokenizer` are the underlying byte-pair encoding modules that perform the actual raw string to token ID conversion and back.
In torchtune, they are required to implement ``encode`` and ``decode`` methods, which are called by the :ref:`model_tokenizers` to convert
between raw text and token IDs.

.. code-block:: python

    class BaseTokenizer(Protocol):

        def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
            """
            Given a string, return the encoded list of token ids.

            Args:
                text (str): The text to encode.
                **kwargs (Dict[str, Any]): kwargs.

            Returns:
                List[int]: The encoded list of token ids.
            """
            pass

        def decode(self, token_ids: List[int], **kwargs: Dict[str, Any]) -> str:
            """
            Given a list of token ids, return the decoded text, optionally including special tokens.

            Args:
                token_ids (List[int]): The list of token ids to decode.
                **kwargs (Dict[str, Any]): kwargs.

            Returns:
                str: The decoded text.
            """
            pass

If you load any :ref:`model_tokenizers`, you can see that it calls its underlying :class:`~torchtune.modules.tokenizers.BaseTokenizer`
to do the actual encoding and decoding.

.. code-block:: python

    from torchtune.models.mistral import mistral_tokenizer
    from torchtune.modules.tokenizers import SentencePieceBaseTokenizer

    m_tokenizer = mistral_tokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model")
    # Mistral uses SentencePiece for its underlying BPE
    sp_tokenizer = SentencePieceBaseTokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model")

    text = "hello world"

    print(m_tokenizer.encode(text))
    # [1, 6312, 28709, 1526, 2]

    print(sp_tokenizer.encode(text))
    # [1, 6312, 28709, 1526, 2]

.. _model_tokenizers:

Model tokenizers
----------------

:class:`~torchtune.modules.tokenizers.ModelTokenizer` are specific to a particular model. They are required to implement the ``tokenize_messages`` method,
which converts a list of Messages into a list of token IDs.

.. code-block:: python

    class ModelTokenizer(Protocol):

        special_tokens: Dict[str, int]
        max_seq_len: Optional[int]

        def tokenize_messages(
            self, messages: List[Message], **kwargs: Dict[str, Any]
        ) -> Tuple[List[int], List[bool]]:
            """
            Given a list of messages, return a list of tokens and list of masks for
            the concatenated and formatted messages.

            Args:
                messages (List[Message]): The list of messages to tokenize.
                **kwargs (Dict[str, Any]): kwargs.

            Returns:
                Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
            """
            pass

The reason they are model specific and different from :ref:`base_tokenizers`
is because they add all the necessary special tokens or prompt templates required to prompt the model.

.. code-block:: python

    from torchtune.models.mistral import mistral_tokenizer
    from torchtune.modules.tokenizers import SentencePieceBaseTokenizer
    from torchtune.data import Message

    m_tokenizer = mistral_tokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model")
    # Mistral uses SentencePiece for its underlying BPE
    sp_tokenizer = SentencePieceBaseTokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model")

    text = "hello world"
    msg = Message(role="user", content=text)

    tokens, mask = m_tokenizer.tokenize_messages([msg])
    print(tokens)
    # [1, 733, 16289, 28793, 6312, 28709, 1526, 28705, 733, 28748, 16289, 28793]
    print(sp_tokenizer.encode(text))
    # [1, 6312, 28709, 1526, 2]
    print(m_tokenizer.decode(tokens))
    # [INST] hello world  [/INST]
    print(sp_tokenizer.decode(sp_tokenizer.encode(text)))
    # hello world
