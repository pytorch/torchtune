.. _prompt_templates_usage_label:

================
Prompt Templates
================

Prompt templates are structured text templates which are used to format user prompts
to optimize model performance on specific tasks. They can serve many purposes:

1. Model-specific templates that are required whenever the model is prompted, such as the [INST]
   tags in the instruct-tuned Llama2 and Mistral models. These models were pre-trained with these tags and using them
   in inference can help ensure optimal performance.
2. Task-specific templates to gear models for a particular task that it will expect after training.
   Example include grammar correction (:class:`~torchtune.data.GrammarErrorCorrectionTemplate`),
   summarization (:class:`~torchtune.data.SummarizeTemplate`), question answering (:class:`~torchtune.data.QuestionAnswerTemplate`),
   and more.
3. Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

For example, if I wanted to fine-tune a model to perform a grammar correction task, I could use the :class:`~torchtune.data.GrammarErrorCorrectionTemplate`
to add the text "Correct this to standard English: {prompt} --- Corrected: {response}" to all my data samples.

.. code-block:: python

    from torchtune.data import GrammarErrorCorrectionTemplate, Message

    sample = {
        "incorrect": "This are a cat",
        "correct": "This is a cat.",
    }
    msgs = [
        Message(role="user", content=sample["incorrect"]),
        Message(role="assistant", content=sample["correct"]),
    ]

    gec_template = GrammarErrorCorrectionTemplate()
    templated_msgs = gec_template(msgs)
    for msg in templated_msgs:
        print(msg.text_content)
    # Correct this to standard English: This are a cat
    # ---
    # Corrected:
    # This is a cat.


The added text is different from special tokens that are added by the model tokenizer. For an extended
discussion on the different between prompt templates and special tokens, see :ref:`prompt_template_vs_special_tokens`.

.. _using_prompt_templates:

Using prompt templates
----------------------
Prompt templates are passed into the tokenizer and will be automatically applied for the dataset you are fine-tuning on. You can pass it in two ways:

- A string dotpath to a prompt template class, i.e., "torchtune.models.mistral.MistralChatTemplate" or "path.to.my.CustomPromptTemplate"
- A dictionary that maps role to a tuple of strings indicating the text to add before and after the message content


Defining via dotpath string
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # In code
    from torchtune.models.mistral import mistral_tokenizer

    m_tokenizer = mistral_tokenizer(
        path="/tmp/Mistral-7B-v0.1/tokenizer.model"
        prompt_template="torchtune.models.mistral.MistralChatTemplate"
    )

.. code-block:: yaml

    # In config
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

    # In code
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

    # In config
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

Using the :class:`~torchtune.data.PromptTemplate` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A template dictionary can also be passed into :class:`~torchtune.data.PromptTemplate` so you can use it as a standalone custom
prompt template class.

.. code-block:: python

    from torchtune.data import PromptTemplate

    def my_custom_template() -> PromptTemplate:
        return PromptTemplate(
            template={
                "user": ("User: ", "\\n"),
                "assistant": ("Assistant: ", "\\n"),
            },
        )

    template = my_custom_template()
    msgs = [
        Message(role="user", content="Hello world!"),
        Message(role="assistant", content="Is AI overhyped?"),
    ]
    templated_msgs = template(msgs)
    for msg in templated_msgs:
        print(msg.role, msg.text_content)
    # user, User: Hello world!
    #
    # assistant, Assistant: Is AI overhyped?
    #

.. TODO (RdoubleA) add a section on how to define prompt templates for inference once generate script is finalized

Custom prompt templates
-----------------------

For more advanced configuration that doesn't neatly fall into the ``PREPEND_TAG content APPEND_TAG``
pattern, you can create a new class that inherits from :class:`~torchtune.data.PromptTemplateInterface`
and implements the ``__call__`` method.

.. code-block:: python

    from torchtune.data import Message

    class PromptTemplateInterface(Protocol):
        def __call__(
            self,
            messages: List[Message],
            inference: bool = False,
        ) -> List[Message]:
            """
            Format each role's message(s) according to the prompt template

            Args:
                messages (List[Message]): a single conversation, structured as a list
                    of :class:`~torchtune.data.Message` objects
                inference (bool): Whether the template is being used for inference or not.

            Returns:
                The formatted list of messages
            """
            pass

    # Contrived example - make all assistant prompts say "Eureka!"
    class EurekaTemplate(PromptTemplateInterface):
        def __call__(
            self,
            messages: List[Message],
            inference: bool = False,
        ) -> List[Message]:
            formatted_dialogue = []
            for message in messages:
                if message.role == "assistant":
                    content = "Eureka!"
                else:
                    content = message.content
                formatted_dialogue.append(
                    Message(
                        role=message.role,
                        content=content,
                        masked=message.masked,
                        ipython=message.ipython,
                        eot=message.eot,
                    ),
                )
            return formatted_dialogue

    template = EurekaTemplate()
    msgs = [
        Message(role="user", content="Hello world!"),
        Message(role="assistant", content="Is AI overhyped?"),
    ]
    templated_msgs = template(msgs)
    for msg in templated_msgs:
        print(msg.role, msg.text_content)
    # user, Hello world!
    # assistant, Eureka!

For more examples, you can look at :class:`~torchtune.models.mistral.MistralChatTemplate` or
:class:`~torchtune.models.llama2.Llama2ChatTemplate`.

To use this custom template in the tokenizer, you can pass it in via dotpath string:

.. code-block:: python

    # In code
    from torchtune.models.mistral import mistral_tokenizer

    m_tokenizer = mistral_tokenizer(
        path="/tmp/Mistral-7B-v0.1/tokenizer.model",
        prompt_template="path.to.template.EurekaTemplate",
    )

.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model
      prompt_template: path.to.template.EurekaTemplate

Built-in prompt templates
-------------------------
- :class:`torchtune.data.GrammarErrorCorrectionTemplate`
- :class:`torchtune.data.SummarizeTemplate`
- :class:`torchtune.data.QuestionAnswerTemplate`
- :class:`torchtune.data.ChatMLTemplate`
