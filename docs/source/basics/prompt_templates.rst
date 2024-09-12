.. _prompt_templates_usage_label:

================
Prompt Templates
================

Prompt templates are structured text templates that user prompts are formatted with
for optimal model performance on a task. They can serve many purposes:

1. Model-specific templates that are required whenever the model is prompted, such as the [INST]
   tags in Llama2 and in Mistral. These models were pre-trained with these tags and using them
   in inference can help ensure optimal performance.
2. Task-specific templates to gear models for a particular task that it will expect after training.
   Example include grammar correction (:class:`~torchtune.data.GrammarErrorCorrectionTemplate`),
   summarization (:class:`~torchtune.data.SummarizeTemplate`), question answering (:class:`~torchtune.data.QuestionAnswerTemplate`),
   and more.
3. Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

The added text is different from special tokens that are added by the model tokenizer. For an extended
discussion on the different between prompt templates and special tokens, see :ref:`prompt_template_vs_special_tokens`.

Custom prompt templates
-----------------------
In most cases, you can define your prompt template as a dictionary and pass it directly into the tokenizer.
The expected format is:

.. code-block:: python

    template = {role: (PREPEND_TAG, APPEND_TAG)}

See :ref:`custom_dictionary_template` for an example of how to pass it in the tokenizer.

For more advanced configuration that doesn't neatly fall into the ``PREPEND_TAG content APPEND_TAG``
pattern, you can create a new class that inherits from :class:`~torchtune.data.PromptTemplateInterface`
and implements the ``__call__`` method.

.. code-block:: python

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

For more examples, you can look at :class:`~torchtune.models.mistral.MistralChatTemplate` or
:class:`~torchtune.models.llama2.Llama2ChatTemplate`.

To use this custom template in the tokenizer, you can pass it in via dotpath string:

.. code-block:: python

    from torchtune.models.mistral import mistral_tokenizer

    m_tokenizer = mistral_tokenizer(
        path="/tmp/Mistral-7B-v0.1/tokenizer.model",
        prompt_template="path.to.template.EurekaTemplate",
    )

.. code-block:: yaml

    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model
      prompt_template: path.to.template.EurekaTemplate
