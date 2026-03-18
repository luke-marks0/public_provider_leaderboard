from types import SimpleNamespace

from openai.types.chat.chat_completion_message import ChatCompletionMessage

from token_difr.openrouter_api import tokenize_openrouter_responses


class _TokenizerStub:
    def __init__(self) -> None:
        self.name_or_path = "deepseek-ai/DeepSeek-V3.2"
        self.chat_template = None
        self.encoded_texts: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        self.encoded_texts.append(text)
        return [ord(ch) for ch in text]


def _completion_with_message(message_payload: dict) -> SimpleNamespace:
    message = ChatCompletionMessage.model_validate(message_payload)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_tokenize_openrouter_responses_uses_reasoning_content_fallback() -> None:
    tokenizer = _TokenizerStub()
    conversations = [[{"role": "user", "content": "hello"}]]
    responses = [
        _completion_with_message(
            {
                "role": "assistant",
                "content": "final answer",
                "reasoning_content": "step by step",
            }
        )
    ]

    tokenize_openrouter_responses(conversations, responses, tokenizer, max_tokens=None)

    assert tokenizer.encoded_texts[1] == "<think>step by step</think>final answer"


def test_tokenize_openrouter_responses_uses_reasoning_details_fallback() -> None:
    tokenizer = _TokenizerStub()
    conversations = [[{"role": "user", "content": "hello"}]]
    responses = [
        _completion_with_message(
            {
                "role": "assistant",
                "content": "done",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "alpha "},
                    {"type": "reasoning.text", "text": {"content": "beta"}},
                ],
            }
        )
    ]

    tokenize_openrouter_responses(conversations, responses, tokenizer, max_tokens=None)

    assert tokenizer.encoded_texts[1] == "<think>alpha beta</think>done"


def test_tokenize_openrouter_responses_avoids_double_inline_thinking() -> None:
    tokenizer = _TokenizerStub()
    conversations = [[{"role": "user", "content": "hello"}]]
    responses = [
        _completion_with_message(
            {
                "role": "assistant",
                "content": "<think>private chain</think>final answer",
                "reasoning": "private chain",
            }
        )
    ]

    tokenize_openrouter_responses(conversations, responses, tokenizer, max_tokens=None)

    assert tokenizer.encoded_texts[1] == "<think>private chain</think>final answer"
