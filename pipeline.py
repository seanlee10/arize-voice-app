"""STT → LLM → TTS 파이프라인 음성 에이전트 + Arize AX 트레이싱.

Realtime API 대신 단계별 API를 조합하는 구성. LLM 호출은 OpenInference 계측기가 자동으로
트레이싱하고, 오디오(USER 음성, TTS 음성)는 같은 시맨틱 컨벤션(input/output.audio.*)으로
직접 스팬을 만들어 붙인다 → Arize에서 Realtime 버전과 같은 오디오 UI로 보인다.

트레이스 (턴마다):
  AUDIO "conversation.turn"     ← 입력/출력 텍스트, 단계별 지연시간
  ├─ USER  "user"               ← 사용자 음성(재생 가능) + 전사   (발화 시작~끝 시간)
  ├─ CHAIN "stt.<provider>"     ← STT 모델, 지연시간
  ├─ LLM   (자동 계측)           ← 프롬프트/응답/토큰
  ├─ TOOL  "<tool_name>"        ← 툴 인자/결과
  └─ LLM   "tts.<provider>"     ← 생성된 음성(재생 가능), 보이스/모델, 첫 바이트 지연
                                  (Arize는 출력 오디오를 LLM 스팬에서만 재생 UI로 표시)

실행:
  .venv/bin/python pipeline.py                                   # xAI STT → Grok → xAI TTS
  .venv/bin/python pipeline.py --llm claude                      # LLM만 Claude로
  .venv/bin/python pipeline.py --stt elevenlabs --tts elevenlabs --llm claude
  .venv/bin/python pipeline.py --turns 3
"""
import argparse
import asyncio
import io
import json
import os
import pickle
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
os.environ.setdefault("OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH", "2000000")

import httpx
import numpy as np
import sounddevice as sd
from arize.otel import register
from openinference.instrumentation import using_session
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openai_agents._realtime import (
    pcm16_to_wav_data_uri,
    truncate_audio_data_uri,
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

PROJECT_NAME = os.environ.get("ARIZE_PROJECT_NAME", "openai-realtime-voice")
SAMPLE_RATE = 24_000
CHUNK_FRAMES = 1_200            # 50 ms
SILENCE_END_MS = 700            # 이만큼 조용하면 발화 종료로 판단
MIN_SPEECH_MS = 300
PREROLL_MS = 300
RECORDINGS_DIR = Path(__file__).parent / "recordings"
AUDIO_MAX = int(os.environ["OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH"])

# ── 트레이싱: LLM은 자동 계측, 오디오 단계는 아래에서 직접 스팬 생성 ───────────
tracer_provider = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name=PROJECT_NAME,
)
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)  # Grok(OpenAI 호환 SDK)용
tracer = trace.get_tracer("voice-app.pipeline")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Reply in the same language the user speaks "
    "(Korean if the user speaks Korean). Use the tools for weather and time questions. "
    "Your reply is read aloud by a TTS engine: keep it to one or two short sentences, "
    "plain spoken text only — no markdown, lists, or emoji."
)


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def audio_uri(pcm: bytes) -> str:
    return truncate_audio_data_uri(pcm16_to_wav_data_uri(pcm), AUDIO_MAX)


# ── 툴 ────────────────────────────────────────────────────────────────
def get_weather(location: str, unit: str = "celsius") -> str:
    if unit == "fahrenheit":
        return f"The weather in {location} is 72 °F and sunny."
    return f"The weather in {location} is 22 °C and sunny."


def get_current_time(timezone: str = "Asia/Seoul") -> str:
    return f"The current time in {timezone} is {datetime.now():%I:%M %p}."


TOOLS = {"get_weather": get_weather, "get_current_time": get_current_time}
TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current time for a timezone (IANA name, e.g. Asia/Tokyo).",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": [],
        },
    },
]


def run_tool(name: str, args: dict) -> str:
    with tracer.start_as_current_span(
        name,
        attributes={
            "openinference.span.kind": "TOOL",
            "tool.name": name,
            "input.value": json.dumps(args, ensure_ascii=False),
            "input.mime_type": "application/json",
        },
    ) as span:
        result = TOOLS[name](**args)
        span.set_attribute("output.value", result)
        return result


# ── STT ───────────────────────────────────────────────────────────────
class XaiSTT:
    provider, model = "xai", "grok-voice-transcribe-2.0"

    def __init__(self, http: httpx.AsyncClient, language: str | None):
        self.http, self.language = http, language
        self.headers = {"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"}

    async def transcribe(self, pcm: bytes) -> str:
        data = {"model": self.model}
        if self.language:
            data["language"] = self.language
        r = await self.http.post(
            "https://api.x.ai/v1/stt", headers=self.headers, data=data,
            files={"file": ("speech.wav", wav_bytes(pcm), "audio/wav")},  # file은 마지막 필드
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()


class ElevenLabsSTT:
    provider = "elevenlabs"

    def __init__(self, http: httpx.AsyncClient, language: str | None):
        self.http, self.language = http, language
        self.model = os.environ.get("ELEVENLABS_STT_MODEL_ID", "scribe_v2")
        self.headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}

    async def transcribe(self, pcm: bytes) -> str:
        data = {"model_id": self.model}
        if self.language:
            data["language_code"] = self.language
        r = await self.http.post(
            "https://api.elevenlabs.io/v1/speech-to-text", headers=self.headers, data=data,
            files={"file": ("speech.wav", wav_bytes(pcm), "audio/wav")},
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()


# ── TTS (스트리밍 PCM16 24 kHz) ─────────────────────────────────────────
class XaiTTS:
    provider, model = "xai", ""  # xAI TTS는 모델 선택 파라미터 없음

    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.voice = os.environ.get("XAI_TTS_VOICE", "eve")
        self.headers = {"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"}

    def stream(self, text: str):
        return self.http.stream(
            "POST", "https://api.x.ai/v1/tts", headers=self.headers,
            json={
                "text": text, "voice_id": self.voice, "language": "auto",
                "output_format": {"codec": "pcm", "sample_rate": SAMPLE_RATE},
                "optimize_streaming_latency": 1,
            },
        )


class ElevenLabsTTS:
    provider = "elevenlabs"

    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.voice = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
        self.model = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
        self.headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}

    def stream(self, text: str):
        return self.http.stream(
            "POST", f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}/stream",
            params={"output_format": f"pcm_{SAMPLE_RATE}"}, headers=self.headers,
            json={"text": text, "model_id": self.model},
        )


# ── LLM (툴 루프 포함, 대화 기록 유지) ─────────────────────────────────────
class GrokLLM:
    """xAI는 OpenAI 호환 API → openai SDK + OpenAIInstrumentor로 자동 트레이싱."""

    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
        self.model = os.environ.get("GROK_MODEL", "grok-4.20-0309-non-reasoning")
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tools = [{"type": "function", "function": t} for t in TOOL_SCHEMAS]

    async def reply(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        while True:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.tools,
            )
            msg = resp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                return (msg.content or "").strip()
            for call in msg.tool_calls:
                result = run_tool(call.function.name, json.loads(call.function.arguments or "{}"))
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})


class ClaudeLLM:
    def __init__(self):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()
        self.model = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
        self.messages: list[dict] = []
        self.tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOL_SCHEMAS
        ]

    async def reply(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        while True:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=512, system=SYSTEM_PROMPT,
                messages=self.messages, tools=self.tools,
            )
            self.messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c.id, "content": run_tool(c.name, c.input)}
                for c in calls
            ]})


# ── 한 턴 처리 (트레이싱 포함) ──────────────────────────────────────────
async def handle_turn(utt: dict, stt, llm, tts, play) -> None:
    pcm = utt["pcm"]
    turn = tracer.start_span(
        "conversation.turn", start_time=utt["start_ns"],
        attributes={"openinference.span.kind": "AUDIO", "voice.mode": "pipeline",
                    "stt.provider": stt.provider, "tts.provider": tts.provider,
                    "llm.provider": type(llm).__name__.removesuffix("LLM").lower()},
    )
    turn_ctx = trace.set_span_in_context(turn)
    status = Status(StatusCode.OK)
    try:
        with trace.use_span(turn, end_on_exit=False):
            # USER: 실제 발화 구간 [start, end] — evaluate.py가 이 구간으로 녹음을 자른다
            user = tracer.start_span("user", context=turn_ctx, start_time=utt["start_ns"], attributes={
                "openinference.span.kind": "USER",
                "input.audio.url": audio_uri(pcm), "input.audio.mime_type": "audio/wav",
            })

            t0 = time.monotonic_ns()
            with tracer.start_as_current_span(f"stt.{stt.provider}", attributes={
                "openinference.span.kind": "CHAIN", "stt.provider": stt.provider, "stt.model": stt.model,
                "stt.audio_duration_s": len(pcm) / 2 / SAMPLE_RATE,
            }) as s:
                text = await stt.transcribe(pcm)
                s.set_attribute("output.value", text)
            stt_ms = (time.monotonic_ns() - t0) / 1e6
            user.set_attribute("input.audio.transcript", text)
            user.end(end_time=utt["end_ns"])
            print(f"[user] {text}  (STT {stt_ms:.0f} ms)")
            turn.set_attribute("input.value", text)
            turn.set_attribute("latency.stt_ms", stt_ms)
            if not text:
                turn.set_attribute("voice.end_reason", "empty_transcript")
                return

            t1 = time.monotonic_ns()
            reply = await llm.reply(text)  # LLM/TOOL 스팬은 자동 계측 + run_tool
            llm_ms = (time.monotonic_ns() - t1) / 1e6
            print(f"[assistant] {reply}  (LLM {llm_ms:.0f} ms)")
            turn.set_attribute("output.value", reply)
            turn.set_attribute("latency.llm_ms", llm_ms)

            await speak(reply, tts, play, turn, utt["end_mono"])
    except Exception as e:
        status = Status(StatusCode.ERROR, description=str(e)[:200])
        print(f"[error] {e}")
    finally:
        turn.set_status(status)
        turn.end()


async def speak(text: str, tts, play, turn, speech_end_mono: int) -> None:
    # Arize UI는 output.audio.url을 LLM 스팬에서만 플레이어로 렌더링 → TTS 스팬도 LLM 종류로 기록
    with tracer.start_as_current_span(f"tts.{tts.provider}", attributes={
        "openinference.span.kind": "LLM", "input.value": text,
        "llm.provider": tts.provider, "llm.model_name": tts.model or f"{tts.provider}-tts",
        "tts.provider": tts.provider, "tts.voice": tts.voice,
    }) as span:
        if tts.model:
            span.set_attribute("tts.model", tts.model)
        t0 = time.monotonic_ns()
        out = bytearray()
        leftover = b""
        async with tts.stream(text) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                raise RuntimeError(f"TTS {tts.provider} {resp.status_code}: {body[:300]}")
            async for chunk in resp.aiter_bytes():
                chunk = leftover + chunk
                cut = len(chunk) - len(chunk) % 2
                leftover = chunk[cut:]
                if not cut:
                    continue
                if not out:
                    now = time.monotonic_ns()
                    span.set_attribute("tts.time_to_first_byte_ms", (now - t0) / 1e6)
                    # 체감 지연: 사용자 발화 종료 → 첫 음성
                    ttfa = (now - speech_end_mono) / 1e6
                    span.set_attribute("time_to_first_audio_ms", ttfa)
                    turn.set_attribute("time_to_first_audio_ms", ttfa)
                    turn.set_attribute("latency.tts_first_byte_ms", (now - t0) / 1e6)
                    print(f"[tts] first audio {ttfa:.0f} ms after speech end")
                out.extend(chunk[:cut])
                await play(chunk[:cut])
        pcm = bytes(out)
        span.set_attribute("tts.audio_duration_s", len(pcm) / 2 / SAMPLE_RATE)
        span.set_attribute("output.audio.url", audio_uri(pcm))
        span.set_attribute("output.audio.mime_type", "audio/wav")
        span.set_attribute("output.audio.transcript", text)
        span.set_attribute("output.value", text)


# ── 마이크 + 에너지 기반 VAD (반이중: 응답 중엔 마이크 무시 → 스피커 에코 방지) ───
async def run(args) -> list[tuple[int, bytes]]:
    loop = asyncio.get_running_loop()
    mic_q: asyncio.Queue = asyncio.Queue()
    recording: list[tuple[int, bytes]] = []
    http = httpx.AsyncClient(timeout=60)

    stt = {"xai": XaiSTT, "elevenlabs": ElevenLabsSTT}[args.stt](http, args.language)
    tts = {"xai": XaiTTS, "elevenlabs": ElevenLabsTTS}[args.tts](http)
    llm = {"grok": GrokLLM, "claude": ClaudeLLM}[args.llm]()
    session_id = str(uuid.uuid4())

    def cb(indata, frames, _t, status):
        chunk = indata.copy().tobytes()
        ts = time.time_ns()
        recording.append((ts, chunk))
        loop.call_soon_threadsafe(mic_q.put_nowait, (ts, time.monotonic_ns(), chunk))

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=CHUNK_FRAMES, callback=cb), \
         sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                         blocksize=CHUNK_FRAMES) as out:

        async def play(pcm: bytes):
            await loop.run_in_executor(None, out.write, np.frombuffer(pcm, dtype=np.int16))

        print(f"[pipeline] STT={args.stt}  LLM={args.llm}  TTS={args.tts}  — 말씀하세요 "
              f"({args.turns}턴 후 종료, Ctrl+C 중단)\n")
        chunk_ms = CHUNK_FRAMES * 1000 // SAMPLE_RATE
        preroll: list = []
        speech: list = []
        silence_ms = 0
        turns = 0
        deadline = time.monotonic() + args.max_seconds
        with using_session(session_id):
            while turns < args.turns and time.monotonic() < deadline:
                try:
                    ts, mono, chunk = await asyncio.wait_for(mic_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                loud = np.sqrt(np.mean(np.frombuffer(chunk, np.int16).astype(np.float32) ** 2)) > args.vad_threshold
                if not speech:
                    preroll = (preroll + [(ts, mono, chunk)])[-(PREROLL_MS // chunk_ms):]
                    if loud:
                        speech, silence_ms = list(preroll), 0
                        print("[listening…]")
                    continue
                speech.append((ts, mono, chunk))
                silence_ms = 0 if loud else silence_ms + chunk_ms
                if silence_ms < SILENCE_END_MS:
                    continue
                # 발화 종료
                voiced = speech[: len(speech) - silence_ms // chunk_ms]
                speech, preroll = [], []
                if len(voiced) * chunk_ms < MIN_SPEECH_MS:
                    continue
                utt = {
                    "pcm": b"".join(c for _, _, c in voiced),
                    "start_ns": voiced[0][0], "end_ns": voiced[-1][0] + chunk_ms * 1_000_000,
                    "end_mono": voiced[-1][1] + chunk_ms * 1_000_000,
                }
                await handle_turn(utt, stt, llm, tts, play)
                turns += 1
                print(f"[turn {turns}/{args.turns} 완료]\n")
                while not mic_q.empty():  # 응답 중 들어온 마이크 입력(에코) 버림
                    mic_q.get_nowait()

    await http.aclose()
    tracer_provider.force_flush()
    print("세션 종료. 트레이스를 Arize로 전송했습니다.")
    return recording


def main():
    p = argparse.ArgumentParser(description="STT → LLM → TTS 파이프라인 음성 에이전트")
    p.add_argument("--stt", choices=["xai", "elevenlabs"], default="xai")
    p.add_argument("--llm", choices=["grok", "claude"], default="grok")
    p.add_argument("--tts", choices=["xai", "elevenlabs"], default="xai")
    p.add_argument("--turns", type=int, default=1)
    p.add_argument("--language", default=None, help="STT 언어 힌트 (예: ko). 기본: 자동 감지")
    p.add_argument("--vad-threshold", type=float, default=500, help="발화 감지 RMS 임계값")
    p.add_argument("--max-seconds", type=int, default=180)
    args = p.parse_args()

    recording = asyncio.run(run(args))
    RECORDINGS_DIR.mkdir(exist_ok=True)
    path = RECORDINGS_DIR / f"mic_{datetime.now():%Y%m%d_%H%M%S}.pkl"
    with open(path, "wb") as f:
        pickle.dump(recording, f)
    print(f"마이크 녹음 저장: {path}  → 톤 평가: .venv/bin/python evaluate.py")


if __name__ == "__main__":
    main()
