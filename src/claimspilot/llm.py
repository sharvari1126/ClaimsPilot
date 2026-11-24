"""Thin LLM abstraction so everything can be tested with a fake client.

Messages use the Anthropic Messages API shape internally. Two real clients exist:
  * GeminiClient    - set GEMINI_API_KEY (free tier available); converts the messages to Gemini format
  * AnthropicClient - set ANTHROPIC_API_KEY
get_llm() picks one (LLM_PROVIDER=gemini|anthropic to force a choice); None means offline mode.
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # override with CLAIMSPILOT_MODEL
DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL  # kept for backwards compatibility


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider-native message, replayed verbatim on the next turn (Gemini needs this)


class LLMClient(Protocol):
    def generate(self, system: str, messages: list[dict], tools: Optional[list[dict]] = None,
                 max_tokens: int = 1500) -> LLMResponse: ...


# ------------------------------------------------------------------ Anthropic
class AnthropicClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model or os.environ.get("CLAIMSPILOT_MODEL", DEFAULT_ANTHROPIC_MODEL)

    def generate(self, system, messages, tools=None, max_tokens=1500) -> LLMResponse:
        clean = [{k: v for k, v in m.items() if k != "raw"} for m in messages]  # "raw" is Gemini-only
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=max_tokens, system=system, messages=clean)
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=calls)


# --------------------------------------------------------------------- Gemini
def _gemini_schema(d: dict) -> dict:
    """JSON-schema dict -> dict the Gemini SDK accepts (upper-case type names, recursive)."""
    out: dict[str, Any] = {"type": str(d.get("type", "object")).upper()}
    if "description" in d:
        out["description"] = d["description"]
    if d.get("properties"):
        out["properties"] = {k: _gemini_schema(v) for k, v in d["properties"].items()}
    if d.get("required"):
        out["required"] = list(d["required"])
    return out


def to_gemini_declarations(tools: list[dict]):
    from google.genai import types

    decls = []
    for t in tools:
        schema = t.get("input_schema") or {}
        kwargs: dict[str, Any] = {"name": t["name"], "description": t.get("description", "")}
        if schema.get("properties"):  # Gemini rejects an empty parameter object: omit it for no-arg tools
            kwargs["parameters"] = types.Schema(**_gemini_schema(schema))
        decls.append(types.FunctionDeclaration(**kwargs))
    return decls


def to_gemini_contents(messages: list[dict]):
    """Anthropic-shaped messages -> Gemini Content list."""
    from google.genai import types

    names: dict[str, str] = {}  # tool_use id -> tool name, needed to label function responses
    contents = []
    for m in messages:
        if m.get("raw") is not None:  # the model's own earlier turn, with thought signatures intact
            contents.append(m["raw"])
            for b in m["content"] if isinstance(m["content"], list) else []:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
            continue
        role = "user" if m["role"] == "user" else "model"
        content = m["content"]
        parts = []
        if isinstance(content, str):
            parts.append(types.Part(text=content))
        else:
            for b in content:
                kind = b.get("type")
                if kind == "text" and b.get("text"):
                    parts.append(types.Part(text=b["text"]))
                elif kind == "tool_use":
                    names[b["id"]] = b["name"]
                    parts.append(types.Part(function_call=types.FunctionCall(name=b["name"], args=b["input"])))
                elif kind == "tool_result":
                    parts.append(types.Part.from_function_response(
                        name=names.get(b["tool_use_id"], "tool"), response={"result": b["content"]}))
        if parts:
            contents.append(types.Content(role=role, parts=parts))
    return contents


class GeminiClient:
    def __init__(self, model: str | None = None, api_key: str | None = None, client=None,
                 retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        if client is None:
            from google import genai

            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            client = genai.Client(api_key=key)
        self._client = client
        self.model = model or os.environ.get("CLAIMSPILOT_MODEL", DEFAULT_GEMINI_MODEL)
        self._retries, self._sleep = retries, sleep
        self._n = 0

    def generate(self, system, messages, tools=None, max_tokens=1500) -> LLMResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            # thinking tokens count against this limit on some models; too small = empty replies
            max_output_tokens=max(max_tokens, 4096),
            tools=[types.Tool(function_declarations=to_gemini_declarations(tools))] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = to_gemini_contents(messages)
        resp = None
        for attempt in range(self._retries):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except Exception as e:  # free tier rate limits (429) and overload (503) are common
                transient = getattr(e, "code", None) in (429, 500, 503) or "429" in str(e) or "503" in str(e)
                if not transient or attempt == self._retries - 1:
                    raise
                self._sleep(4 * (attempt + 1))

        cands = getattr(resp, "candidates", None) or []
        if not cands or cands[0].content is None or not cands[0].content.parts:
            return LLMResponse(text="")  # blocked or empty; callers treat this as an invalid reply
        content = cands[0].content
        text, calls = [], []
        for part in content.parts:
            if getattr(part, "function_call", None):
                self._n += 1
                fc = part.function_call
                calls.append(ToolCall(fc.id or f"g{self._n}", fc.name, dict(fc.args or {})))
            elif getattr(part, "text", None) and not getattr(part, "thought", False):
                text.append(part.text)
        return LLMResponse(text="".join(text), tool_calls=calls, raw=content)


# ------------------------------------------------------------------- selection
def get_llm() -> LLMClient | None:
    """A real client if a key is configured, else None (the app then runs in offline mode)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "gemini" or (not provider and has_gemini):
        return GeminiClient() if has_gemini else None
    if provider == "anthropic" or (not provider and has_claude):
        return AnthropicClient() if has_claude else None
    return None


class FakeLLM:
    """Scripted client for tests: pops one response (or a callable returning one) per call."""

    def __init__(self, responses: list[LLMResponse | Callable]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, system, messages, tools=None, max_tokens=1500) -> LLMResponse:
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        r = self.responses.pop(0)
        return r(system, messages, tools) if callable(r) else r
