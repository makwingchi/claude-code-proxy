"""Unit tests for OpenAI client compatibility behavior."""

import json
import logging

import httpx
import pytest
from fastapi import HTTPException
from openai._exceptions import BadRequestError

from src.conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest


def _build_bad_request_error() -> BadRequestError:
    """Create a BadRequestError matching the max_tokens compatibility failure."""
    request = httpx.Request("POST", "https://example.test/openai/deployments/test/chat/completions")
    response = httpx.Response(status_code=400, request=request)
    body = {
        "error": {
            "message": (
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead."
            ),
            "type": "invalid_request_error",
            "param": "max_tokens",
            "code": "unsupported_parameter",
        }
    }
    return BadRequestError(
        message=f"Error code: 400 - {body}",
        response=response,
        body=body,
    )


class _DummyModelResponse:
    """Simple object that mimics OpenAI SDK model_dump behavior."""

    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        """Return the mocked SDK payload."""
        return self.payload


class _AsyncIterator:
    """Minimal async iterator for streaming tests."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _NeverDisconnectedRequest:
    """Test request object that always reports a connected client."""

    async def is_disconnected(self) -> bool:
        """Return False to simulate an active connection."""
        return False


class _DummyCancellationClient:
    """Stub client for cancellation callbacks."""

    def cancel_request(self, request_id: str) -> bool:
        """No-op cancellation hook for converter tests."""
        return False


def _build_client(monkeypatch: pytest.MonkeyPatch) -> OpenAIClient:
    """Create a client without inheriting proxy settings from the shell."""
    for proxy_var in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(proxy_var, raising=False)

    return OpenAIClient("test-key", "https://example.test/v1")


@pytest.mark.asyncio
async def test_create_chat_completion_retries_with_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    """Non-streaming requests should retry with max_completion_tokens when required."""
    client = _build_client(monkeypatch)
    calls = []

    async def fake_create(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise _build_bad_request_error()

        return _DummyModelResponse(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    client.client.chat.completions.create = fake_create

    result = await client.create_chat_completion(
        {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
        request_id="req-non-stream",
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert calls[0]["max_tokens"] == 32
    assert "max_completion_tokens" not in calls[0]
    assert calls[1]["max_completion_tokens"] == 32
    assert "max_tokens" not in calls[1]
    assert "req-non-stream" not in client.active_requests


@pytest.mark.asyncio
async def test_create_chat_completion_stream_retries_with_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Streaming setup should retry before the HTTP response starts."""
    client = _build_client(monkeypatch)
    calls = []
    caplog.set_level(logging.INFO)

    async def fake_create(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise _build_bad_request_error()

        return _AsyncIterator(
            [
                _DummyModelResponse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "hello"},
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
                _DummyModelResponse(
                    {
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ]
                    }
                ),
            ]
        )

    client.client.chat.completions.create = fake_create

    stream = await client.create_chat_completion_stream(
        {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
        request_id="req-stream",
    )
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 3
    assert json.loads(chunks[0][6:])["choices"][0]["delta"]["content"] == "hello"
    assert chunks[-1] == "data: [DONE]"
    assert calls[0]["max_tokens"] == 32
    assert "max_completion_tokens" not in calls[0]
    assert calls[1]["max_completion_tokens"] == 32
    assert "max_tokens" not in calls[1]
    assert "req-stream" not in client.active_requests
    assert any("LLM stream response: hello" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_streaming_converter_turns_http_exception_into_error_event():
    """Streaming conversion should emit an SSE error instead of re-raising HTTP errors."""

    class _FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise HTTPException(status_code=400, detail="bad request")

    claude_request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    events = [
        event
        async for event in convert_openai_streaming_to_claude_with_cancellation(
            _FailingStream(),
            claude_request,
            logging.getLogger(__name__),
            _NeverDisconnectedRequest(),
            _DummyCancellationClient(),
            "req-converter",
        )
    ]

    assert any(event.startswith("event: error") for event in events)
    assert any("bad request" in event for event in events)
