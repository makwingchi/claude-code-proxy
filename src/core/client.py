import asyncio
import json
import uuid
import logging
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any, List
from openai import AsyncOpenAI, AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError

logger = logging.getLogger(__name__)


def _extract_readable_output(response_dict: Dict[str, Any]) -> str:
    """Extract the human-readable text from a Chat Completions-format response."""
    parts = []
    choices = response_dict.get("choices", [])
    for choice in choices:
        msg = choice.get("message") or choice.get("delta") or {}
        content = msg.get("content")
        if content:
            parts.append(content)
        for tc in (msg.get("tool_calls") or []):
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "")
            parts.append(f"[tool_call: {name}({args[:200]})]")
    return " ".join(parts) if parts else "(empty response)"


def _convert_messages_to_responses_input(messages: List[Dict[str, Any]]) -> tuple:
    """Convert Chat Completions messages to Responses API input format.

    Returns (instructions, input_items) where instructions is the system prompt
    and input_items is the Responses API input array.
    """
    instructions = None
    input_items = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            # System messages become the instructions parameter
            if isinstance(content, str):
                instructions = content if not instructions else f"{instructions}\n\n{content}"
            continue

        if role == "user":
            if isinstance(content, str):
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}]
                })
            elif isinstance(content, list):
                resp_content = []
                for block in content:
                    if block.get("type") == "text":
                        resp_content.append({"type": "input_text", "text": block.get("text", "")})
                    elif block.get("type") == "image_url":
                        image_url = block.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:"):
                            # Extract base64 data
                            parts = image_url.split(",", 1)
                            media_type = parts[0].replace("data:", "").replace(";base64", "")
                            data = parts[1] if len(parts) > 1 else ""
                            resp_content.append({
                                "type": "input_image",
                                "image_url": image_url,
                            })
                        else:
                            resp_content.append({
                                "type": "input_image",
                                "image_url": image_url,
                            })
                if resp_content:
                    input_items.append({
                        "type": "message",
                        "role": "user",
                        "content": resp_content
                    })
            continue

        if role == "assistant":
            # Build assistant message content
            assistant_content = []
            if content:
                assistant_content.append({"type": "output_text", "text": content})

            if assistant_content:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": assistant_content
                })

            # Handle tool calls — these are top-level items after the assistant message
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                func = tc.get("function", {})
                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
                input_items.append({
                    "type": "function_call",
                    "id": tc_id,
                    "call_id": tc_id,
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                })
            continue

        if role == "tool":
            # Tool results become function_call_output
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": msg.get("content", ""),
            })
            continue

    return instructions, input_items


def _convert_tools_for_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Chat Completions tools format to Responses API tools format."""
    resp_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            resp_tools.append({
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })
    return resp_tools


def _normalize_responses_to_chat_completion(response_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Responses API output to Chat Completions format for downstream converters."""
    output = response_dict.get("output", [])

    text_parts = []
    tool_calls = []
    tc_index = 0

    for item in output:
        item_type = item.get("type", "")

        if item_type == "message":
            for content_block in item.get("content", []):
                if content_block.get("type") == "output_text":
                    text_parts.append(content_block.get("text", ""))

        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:24]}")),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                },
                "index": tc_index,
            })
            tc_index += 1

    # Determine finish reason
    status = response_dict.get("status", "completed")
    if tool_calls:
        finish_reason = "tool_calls"
    elif status == "incomplete" and response_dict.get("incomplete_details", {}).get("reason") == "max_output_tokens":
        finish_reason = "length"
    else:
        finish_reason = "stop"

    message = {}
    message["role"] = "assistant"
    message["content"] = "".join(text_parts) if text_parts else None
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = response_dict.get("usage", {})

    return {
        "id": response_dict.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
        },
    }


class OpenAIClient:
    """Async OpenAI client with cancellation support."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 90, api_version: Optional[str] = None, custom_headers: Optional[Dict[str, str]] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        
        # Prepare default headers
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "claude-proxy/1.0.0"
        }
        
        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}
        
        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            logger.info(f"api_key={api_key[:10]}")
            logger.info(f"azure_endpoint={base_url}")
            logger.info(f"api_version={api_version}")
            logger.info(f"timeout={timeout}")
            logger.info(f"default_headers={default_headers}")

            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=all_headers
            )
        self.active_requests: Dict[str, asyncio.Event] = {}

    def _build_responses_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Chat Completions-format request dict into a Responses API request dict."""
        messages = request.get("messages", [])
        instructions, input_items = _convert_messages_to_responses_input(messages)

        resp_request = {
            "model": request["model"],
            "input": input_items,
            "max_output_tokens": request.get("max_tokens", 16384),
        }

        if instructions:
            resp_request["instructions"] = instructions

        if request.get("temperature") is not None:
            resp_request["temperature"] = request["temperature"]

        if request.get("top_p") is not None:
            resp_request["top_p"] = request["top_p"]

        # Convert tools
        tools = request.get("tools")
        if tools:
            resp_request["tools"] = _convert_tools_for_responses(tools)

        # Convert tool_choice
        tool_choice = request.get("tool_choice")
        if tool_choice is not None:
            resp_request["tool_choice"] = tool_choice

        return resp_request

    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            deployment = request.get("model", "")
            is_codex = "codex" in deployment

            if is_codex:
                request = self._build_responses_request(request)
                completion_task = asyncio.create_task(
                    self.client.responses.create(**request)
                )
            else:
                completion_task = asyncio.create_task(
                    self.client.chat.completions.create(**request)
                )

            if request_id:
                # Wait for either completion or cancellation
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check if request was cancelled
                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(status_code=499, detail="Request cancelled by client")

                completion = await completion_task
            else:
                completion = await completion_task

            result = completion.model_dump()

            # Normalize Responses API output to Chat Completions format
            if is_codex:
                result = _normalize_responses_to_chat_completion(result)

            logger.info(f"[DEBUG] LLM response: {_extract_readable_output(result)}")
            return result
        
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def create_chat_completion_stream(self, request: Dict[str, Any], request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            deployment = request.get("model", "")
            is_codex = "codex" in deployment

            if is_codex:
                request = self._build_responses_request(request)
                request["stream"] = True
                streaming_completion = await self.client.responses.create(**request)
            else:
                request["stream"] = True
                streaming_completion = await self.client.chat.completions.create(**request)

            if is_codex:
                # Normalize Responses API streaming events to Chat Completions SSE format
                async for event in streaming_completion:
                    if request_id and request_id in self.active_requests:
                        if self.active_requests[request_id].is_set():
                            raise HTTPException(status_code=499, detail="Request cancelled by client")

                    event_dict = event.model_dump() if hasattr(event, 'model_dump') else event
                    event_type = event_dict.get("type", "")

                    if event_type == "response.output_text.delta":
                        chunk = {
                            "choices": [{"index": 0, "delta": {"content": event_dict.get("delta", "")}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}"

                    elif event_type == "response.function_call_arguments.delta":
                        item_id = event_dict.get("item_id", "")
                        call_id = event_dict.get("call_id", item_id)
                        output_index = event_dict.get("output_index", 0)
                        chunk = {
                            "choices": [{"index": 0, "delta": {
                                "tool_calls": [{
                                    "index": output_index,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"arguments": event_dict.get("delta", "")}
                                }]
                            }, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}"

                    elif event_type == "response.output_item.added":
                        item = event_dict.get("item", {})
                        if item.get("type") == "function_call":
                            output_index = event_dict.get("output_index", 0)
                            chunk = {
                                "choices": [{"index": 0, "delta": {
                                    "tool_calls": [{
                                        "index": output_index,
                                        "id": item.get("call_id", item.get("id", "")),
                                        "type": "function",
                                        "function": {"name": item.get("name", ""), "arguments": ""}
                                    }]
                                }, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}"

                    elif event_type == "response.completed":
                        resp = event_dict.get("response", {})
                        normalized = _normalize_responses_to_chat_completion(resp)
                        logger.info(f"[DEBUG] LLM stream response: {_extract_readable_output(normalized)}")
                        usage = resp.get("usage", {})
                        output = resp.get("output", [])
                        has_tool_calls = any(item.get("type") == "function_call" for item in output)

                        if has_tool_calls:
                            finish_reason = "tool_calls"
                        else:
                            finish_reason = "stop"

                        # Send finish chunk
                        chunk = {
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                            "usage": {
                                "prompt_tokens": usage.get("input_tokens", 0),
                                "completion_tokens": usage.get("output_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            }
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}"

                yield "data: [DONE]"
            else:
                # Standard Chat Completions streaming
                async for chunk in streaming_completion:
                    if request_id and request_id in self.active_requests:
                        if self.active_requests[request_id].is_set():
                            raise HTTPException(status_code=499, detail="Request cancelled by client")

                    chunk_dict = chunk.model_dump()
                    chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                    if chunk_dict.get("usage"):
                        logger.info(f"[DEBUG] LLM stream response: {_extract_readable_output(chunk_dict)}")
                    yield f"data: {chunk_json}"

                yield "data: [DONE]"
                
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."
        
        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False