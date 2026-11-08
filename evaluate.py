"""Arize에서 USER 스팬을 내려받아, 로컬 마이크 녹음을 스팬 구간별로 잘라
gpt-audio-mini로 톤(positive/neutral/negative)을 분류하고 결과를 Arize에 다시 기록.

실행:  .venv/bin/python evaluate.py                       # 가장 최근 녹음 사용
       .venv/bin/python evaluate.py recordings/mic_xxx.pkl
"""
import base64
import io
import json
import os
import pickle
import re
import sys
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from arize import ArizeClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_NAME = os.environ.get("ARIZE_PROJECT_NAME", "openai-realtime-voice")
SAMPLE_RATE = 24_000
CHANNELS = 1
RECORDINGS_DIR = Path(__file__).parent / "recordings"

TONE_RAILS = ["positive", "neutral", "negative"]
TONE_SCORES = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
TONE_SYSTEM_PROMPT = (
    "You are an evaluator that listens to a user's voice clip and classifies the tone of voice. "
    f"Reply with one of exactly these labels: {', '.join(TONE_RAILS)}. "
    "Respond with ONLY a JSON object on a single line, with two keys: "
    '"label" (one of the labels above) and "explanation" (a one-sentence reason). '
    "Do not wrap the JSON in markdown, do not include any preamble, do not add any other text."
)

openai_client = OpenAI()


def load_recording(path: str | None) -> list[tuple[int, bytes]]:
    if path is None:
        files = sorted(RECORDINGS_DIR.glob("mic_*.pkl"))
        if not files:
            sys.exit("recordings/ 에 녹음이 없습니다. 먼저 voice_agent.py 를 실행하세요.")
        path = files[-1]
    print(f"녹음 파일: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def wrap_pcm_as_wav_b64(pcm: bytes) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def extract_audio_b64(span, recording) -> str | None:
    start_ns = int(pd.Timestamp(span["start_time"]).value)
    end_ns = int(pd.Timestamp(span["end_time"]).value)
    pcm = b"".join(c for ts, c in recording if start_ns <= ts <= end_ns)
    return wrap_pcm_as_wav_b64(pcm) if pcm else None


def classify_tone(audio_b64: str) -> dict:
    resp = openai_client.chat.completions.create(
        model="gpt-audio-mini",
        modalities=["text"],
        messages=[
            {"role": "system", "content": TONE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Classify the tone of this audio clip:"},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            ]},
        ],
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in classifier response: {text!r}")
    parsed = json.loads(m.group(0))
    label = str(parsed.get("label", "neutral")).lower()
    if label not in TONE_RAILS:
        label = "neutral"
    return {"label": label, "score": TONE_SCORES[label], "explanation": parsed.get("explanation", "")}


def main():
    recording = load_recording(sys.argv[1] if len(sys.argv) > 1 else None)
    if not recording:
        sys.exit("녹음이 비어 있습니다.")

    # 녹음 구간 ± 여유로 export 범위를 좁혀 다른 세션 스팬이 섞이지 않게 함
    rec_start = datetime.fromtimestamp(recording[0][0] / 1e9, timezone.utc) - timedelta(minutes=1)
    rec_end = datetime.fromtimestamp(recording[-1][0] / 1e9, timezone.utc) + timedelta(minutes=1)

    client = ArizeClient(api_key=os.environ["ARIZE_API_KEY"])
    df = client.spans.export_to_df(
        space_id=os.environ["ARIZE_SPACE_ID"],
        project_name=PROJECT_NAME,
        start_time=rec_start,
        end_time=rec_end,
    )
    print(f"Exported {len(df)} spans")
    if df.empty or "attributes.openinference.span.kind" not in df.columns:
        sys.exit("스팬이 아직 없습니다. Arize 수집 지연일 수 있으니 1~2분 후 다시 실행하세요.")

    user_spans = df[df["attributes.openinference.span.kind"] == "USER"]
    print(f"{len(user_spans)} user spans to evaluate")

    rows = []
    for _, span in user_spans.iterrows():
        audio_b64 = extract_audio_b64(span, recording)
        if audio_b64 is None:
            print(f"  skip {span['context.span_id']}: 구간 내 마이크 오디오 없음")
            continue
        r = classify_tone(audio_b64)
        transcript = span.get("attributes.input.audio.transcript", "")
        print(f"  {span['context.span_id']}  {r['label']:<8}  {transcript!s:.40}  — {r['explanation']}")
        rows.append({
            "context.span_id": span["context.span_id"],
            "eval.tone.label": r["label"],
            "eval.tone.score": r["score"],
            "eval.tone.explanation": r["explanation"],
        })

    if not rows:
        sys.exit("평가할 스팬이 없습니다.")
    client.spans.update_evaluations(
        space_id=os.environ["ARIZE_SPACE_ID"],
        project_name=PROJECT_NAME,
        dataframe=pd.DataFrame(rows),
    )
    print(f"Logged {len(rows)} tone evaluations to Arize.")


if __name__ == "__main__":
    main()
