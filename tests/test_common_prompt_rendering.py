from token_difr.common import render_conversation_for_tokenization


class _TokenizerWithTemplate:
    def __init__(self) -> None:
        self.chat_template = "{{ fake }}"
        self.called = False

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):  # noqa: ANN001
        self.called = True
        assert tokenize is False
        assert add_generation_prompt is True
        return "template-rendered"


class _TokenizerWithoutTemplate:
    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path
        self.chat_template = None


def test_render_conversation_uses_native_chat_template_when_available() -> None:
    tokenizer = _TokenizerWithTemplate()
    conversation = [{"role": "user", "content": "Hello"}]

    rendered = render_conversation_for_tokenization(
        tokenizer,
        conversation,
        add_generation_prompt=True,
    )

    assert rendered == "template-rendered"
    assert tokenizer.called is True


def test_render_conversation_uses_deepseek_v32_fallback_format() -> None:
    tokenizer = _TokenizerWithoutTemplate("deepseek-ai/DeepSeek-V3.2")
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "1+1?"},
    ]

    rendered = render_conversation_for_tokenization(
        tokenizer,
        conversation,
        add_generation_prompt=True,
    )

    assert rendered == (
        "<｜begin▁of▁sentence｜>"
        "You are a helpful assistant."
        "<｜User｜>Hello<｜Assistant｜></think>"
        "Hi!<｜end▁of▁sentence｜>"
        "<｜User｜>1+1?<｜Assistant｜></think>"
    )


def test_render_conversation_uses_generic_fallback_for_other_models() -> None:
    tokenizer = _TokenizerWithoutTemplate("Qwen/Qwen3-8B")
    conversation = [{"role": "user", "content": "Hello"}]

    rendered = render_conversation_for_tokenization(
        tokenizer,
        conversation,
        add_generation_prompt=True,
    )

    assert rendered == "user: Hello\n\nassistant:"
