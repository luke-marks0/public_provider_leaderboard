from token_difr.common import encode_thinking_response


class _RecordingTokenizer:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path
        self.last_encoded_text = ""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.last_encoded_text = text
        return list(range(len(text)))


def test_encode_thinking_response_uses_think_tags_for_non_harmony_models() -> None:
    tokenizer = _RecordingTokenizer("Qwen/Qwen3-8B")

    token_ids = encode_thinking_response(
        content="final answer",
        reasoning="chain of thought",
        tokenizer=tokenizer,
    )

    assert tokenizer.last_encoded_text == "<think>chain of thought</think>final answer"
    assert len(token_ids) == len(tokenizer.last_encoded_text)


def test_encode_thinking_response_uses_harmony_format_for_gpt_oss() -> None:
    tokenizer = _RecordingTokenizer("openai/gpt-oss-20b")

    token_ids = encode_thinking_response(
        content="final answer",
        reasoning="analysis text",
        tokenizer=tokenizer,
    )

    assert tokenizer.last_encoded_text == (
        "<|channel|>analysis<|message|>analysis text"
        "<|end|><|start|>assistant<|channel|>final<|message|>final answer"
    )
    assert len(token_ids) == len(tokenizer.last_encoded_text)


def test_encode_thinking_response_uses_harmony_final_channel_without_reasoning() -> None:
    tokenizer = _RecordingTokenizer("openai/gpt-oss-120b")

    token_ids = encode_thinking_response(
        content="final answer",
        reasoning="",
        tokenizer=tokenizer,
    )

    assert tokenizer.last_encoded_text == "<|channel|>final<|message|>final answer"
    assert len(token_ids) == len(tokenizer.last_encoded_text)


def test_encode_thinking_response_applies_max_tokens_after_encoding() -> None:
    tokenizer = _RecordingTokenizer("openai/gpt-oss-20b")

    token_ids = encode_thinking_response(
        content="x" * 50,
        reasoning="y" * 50,
        tokenizer=tokenizer,
        max_tokens=10,
    )

    assert len(token_ids) == 10
