.. _tokenizers_usage_label:

==========
Tokenizers
==========

Tokenizers are a key component of any LLM. They convert raw text into token IDs, which index into embedding vectors that are
understood by the model.

In torchtune, tokenizers play the role of converting Messages object into token IDs and any necessary model-specific special tokens.
They use byte-pair encoding algorithms to convert raw strings into integer IDs. These are usually based on SentencePiece or TikToken.

Downloading tokenizers from Hugging Face
----------------------------------------

Use the ``tune download`` command to download tokenizers from Hugging Face. They are found on the model page and are downloaded with
the model weights.

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

    from torchtune.models.mistral import mistral_tokenizer

    m_tokenizer = mistral_tokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model")
    type(m_tokenizer)
    # <class 'torchtune.models.mistral._tokenizer.MistralTokenizer'>

.. code-block:: yaml

    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model

Setting max sequence length
---------------------------

.. code-block:: python

    from torchtune.models.mistral import mistral_tokenizer

    m_tokenizer = mistral_tokenizer("/tmp/Mistral-7B-v0.1/tokenizer.model", max_seq_len=8192)

.. code-block:: yaml

    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model
      max_seq_len: 8192


Prompt templates
----------------

For more details on what prompt templates are and when you should use them, see :ref:`prompt_templates_usage_label`. Prompt templates
are passed into the tokenizer and will be automatically applied for the dataset you are fine-tuning on. You can pass it in two ways:
- A string dotpath to a prompt template class, i.e., "torchtune.models.mistral.MistralChatTemplate" or "path.to.my.CustomPromptTemplate"
- A dictionary that maps role to a tuple of strings indicating the text to add before and after the message content

Defining via dotpath string
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torchtune.models.mistral import mistral_tokenizer

    m_tokenizer = mistral_tokenizer(
        path="/tmp/Mistral-7B-v0.1/tokenizer.model"
        prompt_template="torchtune.models.mistral.MistralChatTemplate"
    )

.. code-block:: yaml

    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model
      prompt_template: torchtune.models.mistral.MistralChatTemplate

Defining via dictionary
^^^^^^^^^^^^^^^^^^^^^^^

For example to achieve the following prompt template:

.. code-block:: text

    System: {content}\\n
    User: {content}\\n
    Assistant: {content}\\n
    Tool: {content}\\n

You need to pass in a tuple for each role, where ``PREPEND_TAG`` is the string
added before the text content and ``APPEND_TAG`` is the string added after.

.. code-block:: python
        
    template = {role: (PREPEND_TAG, APPEND_TAG)}

Thus, the template would be defined as follows:

.. code-block:: python

    template = {
        "system": ("System: ", "\\n"),
        "user": ("User: ", "\\n"),
        "assistant": ("Assistant: ", "\\n"),
        "ipython": ("Tool: ", "\\n"),
    }

Now we can pass it into the tokenizer as a dictionary:

.. code-block:: python

    from torchtune.models.mistral import mistral_tokenizer

    template = {
        "system": ("System: ", "\\n"),
        "user": ("User: ", "\\n"),
        "assistant": ("Assistant: ", "\\n"),
        "ipython": ("Tool: ", "\\n"),
    }
    m_tokenizer = mistral_tokenizer(
        path="/tmp/Mistral-7B-v0.1/tokenizer.model"
        prompt_template=template,
    )

.. code-block:: yaml

    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model
      prompt_template:
        system: 
          - "System: "
          - "\\n"
        user: 
          - "User: "
          - "\\n"
        assistant: 
          - "Assistant: "
          - "\\n"
        ipython:
          - "Tool: "
          - "\\n"

If you don't want to add a prepend/append tag to a role, you can just pass in an empty string "" where needed.

For more advanced customization of prompt templates, see :ref:`prompt_templates_usage_label`.

.. TODO (RdoubleA) add a section on how to define prompt templates for inference once generate scsript is finalized

Special tokens
--------------

Special tokens are model-specific tags that are required to prompt the model. They are different from prompt templates
because they are assigned their own unique token IDs. For an extended discussion on the difference between special tokens
and prompt templates, see :ref:`prompt_templates_usage_label`.

Special tokens are automatically added to your data by the model tokenizer and do not require any additional configuration
by the user. You also have the ability to customize the special tokens for experimentation by passing in a file path to
the new special tokens mapping in a JSON file. This will NOT modify the underlying ``tokenizer.model`` to support the new
special token ids - it is the user's responsibility to ensure that the tokenizer file encodes it correctly. Note also that 
some models require the presence of certain special tokens for proper usage.

For example, here we change the ``"<|begin_of_text|>"`` and ``"<|end_of_text|>"`` token IDs in Llama3:

.. code-block:: python

    # special_tokens.json
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

    from torchtune.models.llama3 import llama3_tokenizer

    tokenizer = llama3_tokenizer(
        path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model",
        special_tokens_path="special_tokens.json",
    )
    print(tokenizer.special_tokens)
    # {'<|begin_of_text|>': 128257, '<|end_of_text|>': 128258, ...}

Base tokenizers
---------------

:class:`~torchtune.modules.tokenizers.BaseTokenizer` are the underlying byte-pair encoding modules that perform the actual raw string to token ID conversion and back.
In torchtune, they are required to implement ``encode`` and ``decode`` methods, which are called by the :class:`~torchtune.modules.tokenizers.ModelTokenizer` to convert
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

If you load any :class:`~torchtune.modules.tokenizers.ModelTokenizer`, you can see that it calls its underlying :class:`~torchtune.modules.tokenizers.BaseTokenizer` 
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

The reason they are model specific and different from :class:`~torchtune.modules.tokenizers.BaseTokenizer`
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



