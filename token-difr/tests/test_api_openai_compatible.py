"""Unit tests for OpenAI-compatible logprobs parsing."""

from token_difr.api import _openai_logprobs_to_sparse_logprobs


def test_openai_logprobs_parses_fireworks_content_rows() -> None:
    payload = {
        "content": [
            {
                "token_id": 101,
                "top_logprobs": [
                    {"token_id": 101, "logprob": -0.1},
                    {"token_id": 205, "logprob": -1.5},
                ],
            },
            {
                "token_id": 102,
                "top_logprobs": [
                    {"token_id": 102, "logprob": -0.2},
                ],
            },
        ]
    }

    parsed = _openai_logprobs_to_sparse_logprobs(payload, start_idx=0, n_tokens=2)
    assert parsed == [[(101, -0.1), (205, -1.5)], [(102, -0.2)]]


def test_openai_logprobs_parses_token_id_string_maps() -> None:
    payload = {
        "top_logprobs": [
            {"token_id:10": -0.01, "token_id:11": -1.25, "plain_text_token": -3.0},
            {"token_id:12": -0.02},
        ]
    }

    parsed = _openai_logprobs_to_sparse_logprobs(payload, start_idx=0, n_tokens=2)
    assert parsed == [[(10, -0.01), (11, -1.25)], [(12, -0.02)]]
