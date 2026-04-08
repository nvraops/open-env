import json
from dataclasses import dataclass
from types import SimpleNamespace
from urllib import request


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


class _ChatCompletions:
    def __init__(self, client):
        self._client = client

    def create(self, *, model, messages, response_format=None, temperature=0):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self._client.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._client.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        choices = []
        for item in raw.get("choices", []):
            message = item.get("message", {})
            choices.append(_Choice(message=_Message(content=message.get("content"))))
        return SimpleNamespace(choices=choices)


class _Chat:
    def __init__(self, client):
        self.completions = _ChatCompletions(client)


class OpenAI:
    def __init__(self, *, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _Chat(self)
