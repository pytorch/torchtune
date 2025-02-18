from typing import List
from torchtune.data import Message, PromptTemplateInterface

class Llama3ChatTemplate(PromptTemplateInterface):
    """
    Prompt template that formats chat data for Llama3 models using the new special tokens.
    The template formats messages as follows:

        System message:
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system message content}
            <|eot_id|>

        User message:
            <|start_header_id|>user<|end_header_id|>
            {user message content}
            <|eot_id|>

        Assistant message:
            <|start_header_id|>assistant<|end_header_id|>
            {assistant message content}

        Ipython (tool) message:
            <|start_header_id|>ipython<|end_header_id|>
            {ipython message content}
    
    This template is intended for Llama3 instruct-style conversations.
    """

    template = {
        "system": ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n", "<|eot_id|>\n"),
        "user": ("<|start_header_id|>user<|end_header_id|>\n", "<|eot_id|>\n"),
        "assistant": ("<|start_header_id|>assistant<|end_header_id|>\n", ""),
        "ipython": ("<|start_header_id|>ipython<|end_header_id|>\n", ""),
    }

    def __call__(self, messages: List[Message]) -> List[Message]:
        """
        Format conversation messages for Llama3.
        
        Args:
            messages (List[Message]): A conversation as a list of Message objects.
        
        Returns:
            List[Message]: The list of messages with content formatted using Llama3 tokens.
        """
        formatted_dialogue = []
        for message in messages:
            if message.role == "system":
                content = (
                    [{"type": "text", "content": self.template["system"][0]}] +
                    message.content +
                    [{"type": "text", "content": self.template["system"][1]}]
                )
            elif message.role == "user":
                content = (
                    [{"type": "text", "content": self.template["user"][0]}] +
                    message.content +
                    [{"type": "text", "content": self.template["user"][1]}]
                )
            elif message.role == "assistant":
                content = (
                    [{"type": "text", "content": self.template["assistant"][0]}] +
                    message.content +
                    [{"type": "text", "content": self.template["assistant"][1]}]
                )
            elif message.role == "ipython":
                content = (
                    [{"type": "text", "content": self.template["ipython"][0]}] +
                    message.content +
                    [{"type": "text", "content": self.template["ipython"][1]}]
                )
            else:
                # If an unknown role is encountered, use the message content as is.
                content = message.content
            formatted_dialogue.append(
                Message(
                    role=message.role,
                    content=content,
                    masked=message.masked,
                    ipython=message.ipython,
                    eot=message.eot,
                )
            )
        return formatted_dialogue
