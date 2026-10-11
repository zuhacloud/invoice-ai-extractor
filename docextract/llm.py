"""One client for both approaches: any OpenAI-compatible chat API (Ollama by default).

We use plain JSON mode (`response_format={"type": "json_object"}`) and describe the schema in
the prompt, then validate with Pydantic. Grammar-constrained `json_schema` mode would be
stricter, but Ollama's implementation garbles string values that start with a digit
("12.50" comes back as "bytes"), which is most of what an invoice contains.

Ollama's runner for vision models can also get stuck after many requests: every reply then
comes back empty (or as "token repeat limit reached"), even for documents that worked before.
`chat_json` detects that, unloads the model so Ollama restarts it, and asks once more.

Token log-probabilities, when the server returns them, become per-field confidence scores:
see `field_confidences`.
"""

import base64
import io
import json
import math
import urllib.request
from dataclasses import dataclass, field

from openai import APIError, OpenAI
from PIL import Image

from .config import Settings


@dataclass
class LLMReply:
    text: str
    tokens: list[tuple[str, float]] = field(default_factory=list)  # (token, logprob)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def image_to_data_url(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class LLM:
    def __init__(self, settings: Settings, model: str):
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                             timeout=settings.request_timeout)  # fmt: skip
        self.model = model
        self.temperature = settings.temperature

    def chat_json(self, system: str, user: str, images: list[Image.Image] | None = None,
                  history: list[dict] | None = None) -> LLMReply:  # fmt: skip
        content: list[dict] | str = user
        if images:
            content = [{"type": "image_url", "image_url": {"url": image_to_data_url(i)}} for i in images]
            content.append({"type": "text", "text": user})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}, *(history or [])]
        try:
            reply = self._complete(messages)
        except APIError:
            reply = None
        if reply is None or not reply.text.strip():
            self._restart_model()
            reply = self._complete(messages)  # errors from here on are real: let the caller handle them
        return reply

    def _complete(self, messages: list[dict]) -> LLMReply:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            logprobs=True,
        )
        choice = resp.choices[0]
        tokens = []
        if choice.logprobs and choice.logprobs.content:
            tokens = [(t.token, t.logprob) for t in choice.logprobs.content]
        usage = resp.usage
        return LLMReply(choice.message.content or "", tokens,
                        usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)  # fmt: skip

    def _restart_model(self) -> None:
        """Ask Ollama to unload the model; the next request loads a fresh copy. No-op for other APIs."""
        base = str(self.client.base_url).rstrip("/").removesuffix("/v1")
        body = json.dumps({"model": self.model, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{base}/api/generate", body, {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30).read()
        except OSError:
            pass  # not Ollama, or not reachable: nothing to restart


def extract_json(text: str) -> tuple[str, int]:
    """The JSON object inside a reply and its offset: tolerates ```json fences and text around it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return text, 0
    return text[start : end + 1], start


# ---------------------------------------------------------------------------------------------
# Confidence from token probabilities
# ---------------------------------------------------------------------------------------------


def _leaf_spans(text: str) -> dict[str, tuple[int, int]]:
    """Character span of every leaf value in a JSON string, keyed by path like 'line_items[0].amount'.

    A tiny scanner rather than json.loads, because we need *where* each value sits in the text.
    """
    spans: dict[str, tuple[int, int]] = {}
    i = 0
    n = len(text)

    def skip_ws():
        nonlocal i
        while i < n and text[i] in " \t\r\n":
            i += 1

    def read_string() -> str:
        nonlocal i
        start = i
        i += 1
        while i < n and text[i] != '"':
            i += 2 if text[i] == "\\" else 1
        i += 1
        return json.loads(text[start:i])

    def value(path: str):
        nonlocal i
        skip_ws()
        if i >= n:
            return
        ch = text[i]
        if ch == "{":
            i += 1
            while True:
                skip_ws()
                if i >= n or text[i] == "}":
                    i += 1
                    return
                if text[i] == ",":
                    i += 1
                    continue
                key = read_string()
                skip_ws()
                i += 1  # ':'
                value(f"{path}.{key}" if path else key)
        elif ch == "[":
            i += 1
            idx = 0
            while True:
                skip_ws()
                if i >= n or text[i] == "]":
                    i += 1
                    return
                if text[i] == ",":
                    i += 1
                    continue
                value(f"{path}[{idx}]")
                idx += 1
        else:
            start = i
            if ch == '"':
                read_string()
            else:
                while i < n and text[i] not in ",}] \t\r\n":
                    i += 1
            spans[path] = (start, i)

    i = max(text.find("{"), 0)  # skip anything before the object (e.g. a ```json fence)
    try:
        value("")
    except (json.JSONDecodeError, IndexError):
        pass  # malformed output: return whatever was found
    return spans


def field_confidences(text: str, tokens: list[tuple[str, float]]) -> dict[str, float]:
    """Per-field confidence = probability of the *least* certain token inside that field's value.

    The minimum (not the average) is used because one doubtful digit makes the whole amount doubtful.
    """
    if not tokens or "".join(t for t, _ in tokens) != text:
        return {}  # no logprobs, or tokens don't line up with the text
    offsets = []
    pos = 0
    for tok, lp in tokens:
        offsets.append((pos, pos + len(tok), lp))
        pos += len(tok)
    result = {}
    for path, (start, end) in _leaf_spans(text).items():
        lps = [lp for s, e, lp in offsets if s < end and e > start]
        if lps:
            result[path] = round(math.exp(min(lps)), 4)
    return result
