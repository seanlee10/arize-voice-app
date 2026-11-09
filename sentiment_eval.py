"""오디오 스팬의 전사(transcript)로 감정(sentiment)을 분류해 Arize에 평가로 기록.

Claude는 오디오 입력을 받지 않으므로, 오디오 자체의 '톤'이 아니라 USER 오디오 스팬의
input.audio.transcript '내용'의 감정을 판단한다. (톤 평가는 evaluate.py — 오디오 모델 필요)

실행:
  .venv/bin/python sentiment_eval.py              # 최근 1시간, 결과 출력만 (dry run)
  .venv/bin/python sentiment_eval.py --hours 24   # 조회 범위 변경
  .venv/bin/python sentiment_eval.py --log        # Arize 스팬에 eval.sentiment.* 기록
"""
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import anthropic
import pandas as pd
from arize import ArizeClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent / ".env")

PROJECT_NAME = os.environ.get("ARIZE_PROJECT_NAME", "openai-realtime-voice")
MODEL = os.environ.get("SENTIMENT_MODEL", "claude-haiku-5-5")
SCORES = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}

SYSTEM_PROMPT = (
    "You classify the sentiment of one utterance a customer said to a voice assistant. "
    "The text is a speech-to-text transcript, often Korean, so it may be short or lack punctuation. "
    "positive: satisfied, grateful, pleased. negative: frustrated, annoyed, complaining, upset. "
    "neutral: plain requests or questions with no clear emotion (e.g. asking for the weather). "
    "Write the explanation as one short sentence in Korean."
)


class Sentiment(BaseModel):
    label: Literal["positive", "neutral", "negative"]
    explanation: str = Field(description="One short sentence in Korean explaining the label")


client = anthropic.Anthropic()


def classify(transcript: str) -> Sentiment:
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        output_config={"effort": "low"},  # 단순 3분류 → 낮은 effort로 충분
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Transcript:\n{transcript}"}],
        output_format=Sentiment,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refused: {resp.stop_details}")
    return resp.parsed_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=1)
    p.add_argument("--log", action="store_true", help="결과를 Arize에 기록")
    args = p.parse_args()

    arize = ArizeClient(api_key=os.environ["ARIZE_API_KEY"])
    end = datetime.now(timezone.utc)
    df = arize.spans.export_to_df(
        space_id=os.environ["ARIZE_SPACE_ID"], project_name=PROJECT_NAME,
        start_time=end - timedelta(hours=args.hours), end_time=end,
    )
    col = "attributes.input.audio.transcript"
    if df.empty or col not in df:
        raise SystemExit("전사가 있는 오디오 스팬이 없습니다.")
    # 사용자 발화 오디오 스팬 (USER, 이름 "user") 중 전사가 있는 것
    spans = df[(df["attributes.openinference.span.kind"] == "USER") & (df["name"] == "user")
               & df[col].fillna("").str.strip().ne("")]
    print(f"{len(spans)} user audio spans with transcripts  (model: {MODEL})")

    rows = []
    for _, s in spans.iterrows():
        r = classify(s[col])
        print(f"  {s['context.span_id']}  {r.label:<8}  {s[col]!s:.30}  — {r.explanation}")
        rows.append({
            "context.span_id": s["context.span_id"],
            "eval.sentiment.label": r.label,
            "eval.sentiment.score": SCORES[r.label],
            "eval.sentiment.explanation": r.explanation,
        })

    if not rows:
        return
    if not args.log:
        print("\n(dry run) Arize에 기록하려면 --log 를 붙여 실행하세요.")
        return
    arize.spans.update_evaluations(
        space_id=os.environ["ARIZE_SPACE_ID"], project_name=PROJECT_NAME,
        dataframe=pd.DataFrame(rows),
    )
    print(f"Logged {len(rows)} sentiment evaluations to Arize.")


if __name__ == "__main__":
    main()
