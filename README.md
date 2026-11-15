# Arize AX 음성 에이전트 트레이싱 예제

한국어 음성 대화를 턴마다 Arize AX 트레이스로 남기고, 같은 span 위에서 평가하는 예제입니다. 사용자 음성과 에이전트 응답 음성을 Arize 화면에서 바로 재생하고 전사본을 볼 수 있습니다.

두 가지 구성을 모두 다룹니다.

- **Realtime (speech-to-speech)**: OpenAI Realtime 또는 Grok Voice Agent API. `OpenAIAgentsInstrumentor`가 자동으로 기록합니다.
- **파이프라인 (STT → LLM → TTS)**: LLM 호출은 OpenInference가 자동 계측하고, 사용자 음성·STT·TTS span은 같은 규칙으로 직접 만듭니다.

Arize는 계측 방식이 아니라 span kind와 `input.audio.*` / `output.audio.*` 속성(`url`, `mime_type`, `transcript`)을 읽어 화면을 그리므로, 두 구성이 같은 화면으로 보입니다.

## 트레이스 구조 (파이프라인, 턴마다)

```
AUDIO "conversation.turn"     입력/출력 텍스트, 단계별 지연 시간
├─ USER  "user"               사용자 음성(재생 가능) + 전사본
├─ CHAIN "stt.<provider>"     음성 인식 모델, 처리 시간
├─ LLM   (자동 계측)           프롬프트, 응답, 토큰
├─ TOOL  "<tool_name>"        도구 인자와 결과
└─ LLM   "tts.<provider>"     생성된 응답 음성(재생 가능), 보이스·모델, 첫 바이트 지연
```

턴들은 `using_session`으로 하나의 세션으로 묶입니다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `voice_agent.py` | Realtime 음성 에이전트 (OpenAI 또는 Grok, 선택적으로 ElevenLabs TTS) |
| `pipeline.py` | STT → LLM → TTS 파이프라인 음성 에이전트 (xAI 또는 ElevenLabs, LLM은 Grok 또는 Claude) |
| `elevenlabs_tts.py` | Realtime 모델의 텍스트 응답을 ElevenLabs 스트리밍 TTS로 말하기 |
| `grok_compat.py` | Grok Voice Agent API 이벤트를 OpenAI Agents SDK 형식으로 맞추는 어댑터 |
| `sentiment_eval.py` | 사용자 발화 전사본의 감정을 LLM-as-a-judge로 분류해 span에 평가로 기록 |
| `evaluate.py` | 로컬 녹음을 span 구간별로 잘라 오디오 모델로 음성 톤을 분류해 기록 |

## 준비

- Python 3.11 이상, 마이크와 스피커 (에코를 피하려면 헤드폰 권장)
- Arize AX 스페이스 ID와 API 키
- 사용할 구성에 맞는 API 키 (OpenAI, xAI, ElevenLabs, Anthropic 중 필요한 것)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # 키 입력
```

## 실행

Realtime:

```bash
.venv/bin/python voice_agent.py                   # OpenAI Realtime, 1턴
.venv/bin/python voice_agent.py --provider grok   # Grok Voice Agent API
.venv/bin/python voice_agent.py --tts elevenlabs  # 응답 음성을 ElevenLabs로
.venv/bin/python voice_agent.py --turns 3
```

파이프라인:

```bash
.venv/bin/python pipeline.py                      # ElevenLabs 키가 있으면 STT/TTS에 ElevenLabs, 없으면 xAI
.venv/bin/python pipeline.py --llm claude --language ko --turns 3
```

실행 후 Arize의 `openai-realtime-voice` 프로젝트(`ARIZE_PROJECT_NAME`으로 변경 가능)에서 `conversation.turn` 트레이스를 열면 음성을 재생할 수 있습니다.

## 평가

전사본 감정 분류 (Claude 사용):

```bash
.venv/bin/python sentiment_eval.py            # 최근 1시간, 결과만 출력
.venv/bin/python sentiment_eval.py --log      # span에 eval.transcript_sentiment.* 기록
```

음성 톤 분류: 전사본이 아니라 음성 자체를 평가하려면 오디오 입력을 받는 모델이 필요합니다. `evaluate.py`는 `gpt-audio-mini`를 사용하며, 세션 중 `recordings/`에 저장된 마이크 녹음을 씁니다.

```bash
.venv/bin/python evaluate.py                  # 가장 최근 녹음
```

## 참고

- 인라인 base64 오디오는 기본 약 0.5초에서 잘립니다. 스크립트는 계측 전에 `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH=2000000`(약 30초)으로 올립니다.
- Arize는 출력 오디오를 LLM span에서만 재생 UI로 표시하므로, TTS span은 LLM kind로 기록합니다.
- 오디오가 span에 base64로 실리므로 span이 커집니다. 셀프호스팅 Arize AX 11.41.0 이상은 큰 오디오를 버킷으로 옮겨 저장하며, 이때 브라우저 재생을 위해 버킷 CORS 설정이 필요합니다.
- `recordings/`와 `.env`는 저장소에 포함되지 않습니다.
