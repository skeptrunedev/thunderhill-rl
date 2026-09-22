"""FunctionGemma native declarations and role=tool feedback via its own template.

https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma
"""
from native_tools import NativeBikeTools


class FunctionGemmaBikeTools(NativeBikeTools):
    tool_start = "<start_function_call>"
    tool_end = "<end_function_call>"
    tool_stop = "<start_function_response>"
    action_version = "functiongemma-native-bike-tools-v1"

    def messages(self, features):
        messages = super().messages(features)
        # Google documents this instruction as required for function calling.
        messages[0]["content"] = (
            "You are a model that can do function calling with the following functions. "
            + messages[0]["content"]
        )
        return messages

    def _render(self, messages, *, add_generation_prompt):
        return self.tokenizer.apply_chat_template(
            messages, tools=self.tools, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def assistant_message(self, controls, response=None):
        if response is not None:
            raise ValueError("FunctionGemma responses require a separate tool role message")
        return super().assistant_message(controls)

    def feedback_messages(self, controls, response):
        return [self.assistant_message(controls),
                {"role": "tool", "content": {"name": "control_bike", "response": response}}]
