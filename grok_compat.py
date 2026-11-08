"""Grok Voice Agent API ↔ OpenAI Agents SDK 호환 어댑터.

Grok은 OpenAI Realtime 스펙 호환이지만 일부 서버 이벤트 형태가 OpenAI GA 스키마와 달라
SDK의 pydantic 검증에서 떨어진다 (→ RealtimeAgentStart/End 이벤트 누락, 트레이스 턴 경계 깨짐).
SDK가 검증하기 전에 이벤트를 OpenAI 형태로 정규화한다. 계측기는 그 위 레이어
(RealtimeSession)를 패치하므로 트레이싱 코드는 그대로 동작한다.
"""
from typing import Any

from agents.realtime.openai_realtime import OpenAIRealtimeWebSocketModel

# Grok 전용(=OpenAI에 없는) 이벤트: 그냥 버린다
_DROP_TYPES = {"ping", "conversation.item.input_audio_transcription.updated"}

_ASSISTANT_PART = {"audio": "output_audio", "text": "output_text"}
_USER_PART = {"audio": "input_audio", "text": "input_text"}


def _fix_item(item: dict) -> dict:
    item = dict(item)
    item.pop("replayed", None)
    item_type = item.get("type")
    if item_type == "message":
        role = item.get("role")
        mapping = _ASSISTANT_PART if role == "assistant" else _USER_PART
        item["content"] = [
            {**p, "type": mapping.get(p.get("type"), p.get("type"))} for p in item.get("content") or []
        ]
    elif item_type in ("function_call", "function_call_output"):
        item.pop("role", None)  # Grok은 function_call에 role:"user"를 붙임
    return item


def _fix_response(resp: dict) -> dict:
    resp = dict(resp)
    if isinstance(resp.get("status_details"), str):
        resp["status_details"] = None
    if not resp.get("usage"):
        resp.pop("usage", None)
    resp["output"] = [_fix_item(i) for i in resp.get("output") or []]
    return resp


def normalize_grok_event(event: Any) -> Any | None:
    if not isinstance(event, dict):
        return event
    t = event.get("type")
    if t in _DROP_TYPES:
        return None
    e = dict(event)
    if t in ("session.created", "session.updated") and isinstance(e.get("session"), dict):
        e["session"] = {**e["session"], "type": "realtime"}
        e["session"].pop("turn_detection", None)  # Grok: {"type": null} 등 비표준 값
    if t in ("response.created", "response.done") and isinstance(e.get("response"), dict):
        e["response"] = _fix_response(e["response"])
    if "item" in e and isinstance(e["item"], dict):
        e["item"] = _fix_item(e["item"])
        e["item"].setdefault("content", []) if e["item"].get("type") == "message" else None
    if t in ("response.content_part.added", "response.content_part.done"):
        e.setdefault("part", {"type": "audio"})  # Grok은 part를 생략함 (이 이벤트는 audio/text 그대로)
    if t == "conversation.item.input_audio_transcription.completed":
        e.setdefault("usage", {"type": "duration", "seconds": 0})
        e.pop("status", None)
    if t.startswith("response.") and "response_id" in e:
        e.setdefault("output_index", 0)
        e.setdefault("content_index", 0) if "item_id" in e and "function_call" not in t else None
    return e


def normalize_status_details(event: Any) -> Any:
    """비표준 서버가 response.status_details를 문자열(예: 'unimplemented')로 보내면 None으로 바꾼다."""
    if not isinstance(event, dict) or event.get("type") not in ("response.created", "response.done"):
        return event
    resp = event.get("response")
    if isinstance(resp, dict) and isinstance(resp.get("status_details"), str):
        return {**event, "response": {**resp, "status_details": None}}
    return event


class TolerantRealtimeModel(OpenAIRealtimeWebSocketModel):
    """OpenAI 프로바이더용: status_details 문자열만 허용하도록 최소 정규화."""

    async def _handle_ws_event(self, event: dict[str, Any]):
        await super()._handle_ws_event(normalize_status_details(event))


class GrokRealtimeModel(OpenAIRealtimeWebSocketModel):
    async def _handle_ws_event(self, event: dict[str, Any]):
        fixed = normalize_grok_event(event)
        if fixed is None:
            return
        await super()._handle_ws_event(fixed)
