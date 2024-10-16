def render_template(tools, messages, add_generation_prompt=False):
    result = []

    if tools:
        result.append("<|im_start|>system\n")
        if messages[0]["role"] == "system":
            result.append(messages[0]["content"])
        else:
            result.append(
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            )

        result.append(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )

        for tool in tools:
            result.append("\n" + json.dumps(tool))

        result.append(
            '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        )
    else:
        if messages[0]["role"] == "system":
            result.append(f'<|im_start|>system\n{messages[0]["content"]}<|im_end|>\n')
        else:
            result.append(
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            )

    for i, message in enumerate(messages):
        if (
            (message["role"] == "user")
            or (message["role"] == "system" and i != 0)
            or (message["role"] == "assistant" and "tool_calls" not in message)
        ):
            result.append(
                f'<|im_start|>{message["role"]}\n{message["content"]}<|im_end|>\n'
            )
        elif message["role"] == "assistant":
            result.append(f'<|im_start|>{message["role"]}')
            if "content" in message and message["content"]:
                result.append(f'\n{message["content"]}')
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    if "function" in tool_call:
                        tool_call = tool_call["function"]
                    result.append(
                        f'\n<tool_call>\n{{"name": "{tool_call["name"]}", "arguments": {json.dumps(tool_call["arguments"])}}}\n</tool_call>'
                    )
            result.append("<|im_end|>\n")
        elif message["role"] == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                result.append("<|im_start|>user")
            result.append(f'\n<tool_response>\n{message["content"]}\n</tool_response>')
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                result.append("<|im_end|>\n")

    if add_generation_prompt:
        result.append("<|im_start|>assistant\n")

    return "".join(result)


# Usage example:
# tools = [...]  # List of tool dictionaries
# messages = [...]  # List of message dictionaries
# add_generation_prompt = True  # or False
# output = render_template(tools, messages, add_generation_prompt)
# print(output)
