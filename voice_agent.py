"""OpenAI Realtime 음성 에이전트 + Arize AX 자동 트레이싱 데모.

실행:  .venv/bin/python voice_agent.py            # 1턴 대화 후 종료 (문서 기본 동작)
       .venv/bin/python voice_agent.py --provider grok   # Grok Voice Agent API 사용
       .venv/bin/python voice_agent.py --tts elevenlabs  # 음성 생성을 ElevenLabs로
       .venv/bin/python voice_agent.py --turns 3  # 3턴 대화 (헤드폰 권장)

세션이 끝나면 마이크 녹음을 recordings/ 에 저장합니다 → evaluate.py 가 톤 평가에 사용.
"""
import argparse
import asyncio
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 인라인 base64 오디오 상한을 ~30초로 상향 (기본 ~0.5초).
# 반드시 .instrument() 호출 전에 설정해야 함.
os.environ.setdefault("OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH", "2000000")

import numpy as np
import sounddevice as sd
from agents import function_tool
from agents.realtime import RealtimeAgent, RealtimeRunner
from agents.realtime.events import (
    RealtimeAgentEndEvent,
    RealtimeAgentStartEvent,
    RealtimeAudio,
    RealtimeAudioInterrupted,
    RealtimeRawModelEvent,
    RealtimeToolStart,
)
from arize.otel import register
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

PROJECT_NAME = os.environ.get("ARIZE_PROJECT_NAME", "openai-realtime-voice")
SAMPLE_RATE = 24_000
CHANNELS = 1
MIC_CHUNK_FRAMES = 1_200
MAX_SESSION_SECONDS = 120
EXIT_GRACE_PERIOD = 1.5
RECORDINGS_DIR = Path(__file__).parent / "recordings"

# 프로바이더별 접속 설정. Grok Voice Agent API는 OpenAI Realtime 스펙 호환이라
# URL·API 키·모델/보이스 이름만 바꾸면 같은 SDK·같은 계측기로 트레이싱된다.
PROVIDERS = {
    "openai": {"model_config": {}, "settings": {}, "model": None},
    "grok": {
        "model_config": {
            "url": "wss://api.x.ai/v1/realtime?model=grok-voice-latest",
            "api_key_env": "XAI_API_KEY",
        },
        "settings": {"model_name": "grok-voice-latest", "voice": "eve"},
        "model": "grok",
    },
}


def build_runner(provider: str, tts: str = "native") -> tuple["RealtimeRunner", dict]:
    cfg = PROVIDERS[provider]
    settings = dict(cfg["settings"])
    if tts == "elevenlabs":
        # 모델은 텍스트로만 답하고, 음성은 ElevenLabs가 생성
        settings["output_modalities"] = ["text"]
    model_config = dict(cfg["model_config"])
    if env := model_config.pop("api_key_env", None):
        model_config["api_key"] = os.environ[env]
    model = None
    if cfg["model"] == "grok":
        from grok_compat import GrokRealtimeModel  # Grok 이벤트 형태를 OpenAI 스키마로 정규화
        model = GrokRealtimeModel()
    runner = RealtimeRunner(
        agent, model=model,
        config={"model_settings": settings} if settings else None,
    )
    return runner, model_config

# ── 트레이싱 설정 (이게 트레이싱 코드의 전부) ─────────────────────────────
tracer_provider = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name=PROJECT_NAME,
)
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


# ── 툴 ────────────────────────────────────────────────────────────────
@function_tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    if unit == "fahrenheit":
        return f"The weather in {location} is 72 °F and sunny."
    return f"The weather in {location} is 22 °C and sunny."


@function_tool
def get_current_time(timezone: str = "Asia/Seoul") -> str:
    """Get the current time for a timezone."""
    now = datetime.now().strftime("%I:%M %p")
    return f"The current time in {timezone} is {now}."


agent = RealtimeAgent(
    name="Assistant",
    instructions=(
        "You are a helpful voice assistant. Reply in the same language the user speaks "
        "(Korean if the user speaks Korean). "
        "You have tools to look up weather and the current time — use them when asked. "
        "Keep responses short and conversational."
    ),
    tools=[get_weather, get_current_time],
)


# ── 오디오 ─────────────────────────────────────────────────────────────
def make_mic_callback(mic_queue, loop, recording=None):
    def cb(indata, frames, _time, status):
        if status:
            print(f"Mic: {status}")
        chunk = indata.copy().tobytes()
        if recording is not None:
            recording.append((time.time_ns(), chunk))
        loop.call_soon_threadsafe(mic_queue.put_nowait, chunk)
    return cb


async def run_session(
    max_turns: int, provider: str = "openai", tts: str = "native"
) -> list[tuple[int, bytes]]:
    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    stop = asyncio.Event()
    mic_recording: list[tuple[int, bytes]] = []

    async def hard_timer():
        await asyncio.sleep(MAX_SESSION_SECONDS)
        print("\n[timeout]")
        stop.set()

    tts_engine = None
    if tts == "elevenlabs":
        import elevenlabs_tts
        elevenlabs_tts.patch_instrumentor_for_text_output()
        tts_engine = elevenlabs_tts.ElevenLabsTTS()

    runner, model_config = build_runner(provider, tts)
    async with await runner.run(model_config=model_config or None) as session:
        print(f"[{provider} / voice: {tts}] 연결됨. 말씀하세요 — {max_turns}턴 후 자동 종료 (Ctrl+C로 중단).\n")

        async def send_mic():
            while not stop.is_set():
                try:
                    chunk = await asyncio.wait_for(mic_queue.get(), timeout=0.1)
                    await session.send_audio(chunk)
                except asyncio.TimeoutError:
                    continue

        timer_task = asyncio.create_task(hard_timer())
        send_task = asyncio.create_task(send_mic())
        events_task = None
        tts_task = None

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                blocksize=MIC_CHUNK_FRAMES,
                callback=make_mic_callback(mic_queue, loop, recording=mic_recording),
            ), sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                blocksize=MIC_CHUNK_FRAMES,
            ) as out_stream:

                # ── ElevenLabs 모드: 응답 텍스트를 순서대로 합성·재생 ──
                tts_queue: asyncio.Queue[str] = asyncio.Queue()
                speaking: asyncio.Task | None = None

                async def play(pcm: bytes):
                    arr = np.frombuffer(pcm, dtype=np.int16)
                    await loop.run_in_executor(None, out_stream.write, arr)

                async def tts_worker():
                    nonlocal speaking
                    while True:
                        text = await tts_queue.get()
                        try:
                            speaking = asyncio.create_task(tts_engine.speak(text, session, play))
                            await speaking
                        except asyncio.CancelledError:
                            if speaking is None or not speaking.cancelled():
                                raise  # 워커 자체 종료
                        except Exception as e:
                            print(f"[tts error] {e}")
                        finally:
                            speaking = None
                            tts_queue.task_done()

                async def interrupt_tts():
                    while not tts_queue.empty():
                        tts_queue.get_nowait()
                        tts_queue.task_done()
                    if speaking is not None and not speaking.done():
                        speaking.cancel()
                        await loop.run_in_executor(None, out_stream.abort)
                        await loop.run_in_executor(None, out_stream.start)
                        print("[interrupted]")

                if tts_engine is not None:
                    tts_task = asyncio.create_task(tts_worker())

                async def handle_events():
                    exit_task = None
                    turns_done = 0

                    async def turn_finished():
                        nonlocal turns_done
                        try:
                            await asyncio.sleep(EXIT_GRACE_PERIOD)
                            await tts_queue.join()  # ElevenLabs 재생이 끝날 때까지 대기
                            turns_done += 1
                            print(f"[turn {turns_done}/{max_turns} 완료]")
                            if turns_done >= max_turns:
                                await loop.run_in_executor(None, out_stream.stop)
                                stop.set()
                        except asyncio.CancelledError:
                            pass

                    def cancel_pending():
                        nonlocal exit_task
                        if exit_task is not None and not exit_task.done():
                            exit_task.cancel()
                        exit_task = None

                    async for event in session:
                        if stop.is_set():
                            break
                        if tts_engine is not None and isinstance(event, RealtimeRawModelEvent):
                            if (text := elevenlabs_tts.raw_reply_done(event)) and text.strip():
                                print(f"[assistant] {text}")
                                tts_queue.put_nowait(text)
                            elif elevenlabs_tts.raw_type(event) == "input_audio_buffer.speech_started":
                                cancel_pending()
                                await interrupt_tts()
                        elif isinstance(event, RealtimeAudio):
                            if tts_engine is not None:
                                continue  # 모델 자체 음성은 재생하지 않음 (Grok은 항상 생성)
                            arr = np.frombuffer(event.audio.data, dtype=np.int16)
                            await loop.run_in_executor(None, out_stream.write, arr)
                        elif isinstance(event, RealtimeAudioInterrupted):
                            cancel_pending()
                            await loop.run_in_executor(None, out_stream.abort)
                            await loop.run_in_executor(None, out_stream.start)
                            print("[interrupted]")
                        elif isinstance(event, RealtimeToolStart):
                            cancel_pending()
                            print(f"[tool] {event.tool.name}")
                        elif isinstance(event, RealtimeAgentStartEvent):
                            cancel_pending()
                        elif isinstance(event, RealtimeAgentEndEvent):
                            cancel_pending()
                            exit_task = asyncio.create_task(turn_finished())

                events_task = asyncio.create_task(handle_events())
                await stop.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nInterrupted.")
            stop.set()
        finally:
            pending = [t for t in (timer_task, send_task, events_task, tts_task) if t is not None]
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    tracer_provider.force_flush()
    print("세션 종료. 트레이스를 Arize로 전송했습니다.")
    return mic_recording


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=1, help="종료 전까지 대화 턴 수")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="openai")
    parser.add_argument("--tts", choices=["native", "elevenlabs"], default="native",
                        help="native: 모델 자체 음성 / elevenlabs: ElevenLabs로 음성 생성")
    args = parser.parse_args()

    recording = asyncio.run(run_session(args.turns, args.provider, args.tts))

    RECORDINGS_DIR.mkdir(exist_ok=True)
    path = RECORDINGS_DIR / f"mic_{datetime.now():%Y%m%d_%H%M%S}.pkl"
    with open(path, "wb") as f:
        pickle.dump(recording, f)
    print(f"마이크 녹음 저장: {path}  ({len(recording)} chunks)")
    print(f"Arize 프로젝트 '{PROJECT_NAME}' 에서 conversation.turn 트레이스를 확인하세요.")
    print("톤 평가: .venv/bin/python evaluate.py  (Arize 수집 지연으로 1~2분 후 실행 권장)")


if __name__ == "__main__":
    main()
