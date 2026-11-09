"""ElevenLabs TTS 연동 + 트레이싱.

구조:  마이크 → Realtime 모델(OpenAI/Grok, 텍스트 출력) → ElevenLabs 스트리밍 TTS → 스피커

트레이스:
  AUDIO "conversation.turn"
  ├─ USER "user"              ← 사용자 음성 + 전사 (계측기 자동)
  ├─ LLM  "assistant"         ← 텍스트 응답 (계측기 + 아래 텍스트 패치)
  │  └─ TOOL ...
  └─ LLM  "elevenlabs.tts"    ← ElevenLabs가 만든 음성(재생 가능), voice/model, 지연시간
"""
import os
import time
from typing import AsyncIterator

import httpx
from openinference.instrumentation.openai_agents import _realtime as oi_rt
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
DEFAULT_MODEL_ID = "eleven_flash_v2_5"  # 저지연 + 한국어 지원
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # ELEVENLABS_VOICE_ID 로 변경 가능
SAMPLE_RATE = 24_000

_tracer = trace.get_tracer("voice-app.elevenlabs")


# ── 계측기 패치: 텍스트 출력도 assistant 스팬의 출력으로 기록 ─────────────────
# 계측기는 오디오 전사(response.output_audio_transcript.*)만 기록하므로,
# 텍스트 전용 응답(response.output_text.*)을 같은 핸들러로 연결한다.
_TEXT_EVENTS = {
    "response.output_text.delta": "delta", "response.text.delta": "delta",
    "response.output_text.done": "done", "response.text.done": "done",
}


def patch_instrumentor_for_text_output() -> None:
    if getattr(oi_rt, "_text_patch_applied", False):
        return
    original = oi_rt._dispatch_raw

    def _dispatch_raw(state, event):
        original(state, event)
        raw = getattr(getattr(event, "data", None), "data", None)
        if not isinstance(raw, dict):
            return
        kind = _TEXT_EVENTS.get(raw.get("type") or "")
        if kind == "delta":
            state.on_asst_transcript_delta(raw.get("response_id") or "", raw.get("delta") or "")
        elif kind == "done":
            state.on_asst_transcript_done(raw.get("response_id") or "", raw.get("text") or "")

    oi_rt._dispatch_raw = _dispatch_raw
    oi_rt._text_patch_applied = True


def raw_reply_done(event) -> str | None:
    """RealtimeRawModelEvent 가 응답 텍스트 완료면 그 텍스트를 반환.

    OpenAI(텍스트 출력 모드): response.output_text.done 의 text
    Grok(텍스트 전용 모드 미지원 → 항상 오디오 생성): response.output_audio_transcript.done 의 transcript
    """
    raw = getattr(getattr(event, "data", None), "data", None)
    if not isinstance(raw, dict):
        return None
    t = raw.get("type") or ""
    if _TEXT_EVENTS.get(t) == "done":
        return raw.get("text") or ""
    if t in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
        return raw.get("transcript") or ""
    return None


def raw_type(event) -> str:
    raw = getattr(getattr(event, "data", None), "data", None)
    return raw.get("type", "") if isinstance(raw, dict) else ""


def _current_turn(session):
    state = oi_rt._session_states.get(session)
    return getattr(state, "_latest_turn", None) if state else None


class ElevenLabsTTS:
    def __init__(self):
        self.api_key = os.environ["ELEVENLABS_API_KEY"]
        self.voice_id = os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)
        self.model_id = os.environ.get("ELEVENLABS_MODEL_ID", DEFAULT_MODEL_ID)
        self._client = httpx.AsyncClient(timeout=30)

    async def _stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        async with self._client.stream(
            "POST",
            ELEVEN_URL.format(voice_id=self.voice_id),
            params={"output_format": f"pcm_{SAMPLE_RATE}"},
            headers={"xi-api-key": self.api_key},
            json={"text": text, "model_id": self.model_id},
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                raise RuntimeError(f"ElevenLabs {resp.status_code}: {body[:300]}")
            leftover = b""
            async for chunk in resp.aiter_bytes():
                chunk = leftover + chunk
                cut = len(chunk) - (len(chunk) % 2)  # PCM16 샘플 경계 유지
                leftover = chunk[cut:]
                if cut:
                    yield chunk[:cut]

    async def speak(self, text: str, session, play) -> None:
        """text를 합성해 play(pcm_bytes) 로 재생하고, 현재 턴 아래에 TTS 스팬을 남긴다."""
        turn = _current_turn(session)
        ctx = trace.set_span_in_context(turn.turn_span) if turn and not turn.closed else None
        span = _tracer.start_span(
            "elevenlabs.tts",
            context=ctx,
            attributes={
                # Arize UI는 output.audio.url을 LLM 스팬에서만 플레이어로 렌더링
                "openinference.span.kind": "LLM",
                "llm.provider": "elevenlabs",
                "llm.model_name": self.model_id,
                "tts.provider": "elevenlabs",
                "tts.model_id": self.model_id,
                "tts.voice_id": self.voice_id,
            },
        )
        t0 = time.monotonic_ns()  # 계측기와 같은 시계(monotonic) 사용
        first_ns = None
        pcm = bytearray()
        try:
            async for chunk in self._stream_pcm(text):
                if first_ns is None:
                    first_ns = time.monotonic_ns()
                    span.set_attribute("tts.time_to_first_byte_ms", (first_ns - t0) / 1e6)
                    if turn and turn.user_query_ended_ns:
                        # 사용자 발화 종료 → 첫 음성까지: 체감 응답 지연
                        span.set_attribute(
                            "time_to_first_audio_ms", (first_ns - turn.user_query_ended_ns) / 1e6
                        )
                pcm.extend(chunk)
                await play(chunk)
            span.set_attribute("tts.audio_duration_s", len(pcm) / 2 / SAMPLE_RATE)
            if not oi_rt._env_bool(oi_rt._ENV_HIDE_OUTPUT_AUDIO) and pcm:
                uri = oi_rt.truncate_audio_data_uri(
                    oi_rt.pcm16_to_wav_data_uri(bytes(pcm)), oi_rt._base64_audio_max_length()
                )
                span.set_attribute("output.audio.url", uri)
                span.set_attribute("output.audio.mime_type", "audio/wav")
                span.set_attribute("output.audio.transcript", text)
            # input/output.value는 넣지 않음: 있으면 Arize가 오디오 플레이어 대신 텍스트 뷰를 보여줌
            span.set_status(Status(StatusCode.OK))
        except BaseException as e:  # 취소(끼어들기) 포함
            span.set_status(Status(StatusCode.ERROR, description=type(e).__name__))
            if not isinstance(e, Exception):
                span.set_attribute("tts.interrupted", True)
            raise
        finally:
            span.end()
