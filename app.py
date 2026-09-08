"""
AI Study Pack Generator
Streamlit + Google Gemini

Includes resilient model handling:
- Automatic retry with exponential backoff for temporary 503/429 errors
- Retry with the same model
- Retry with another model
- Model selector
- Fallback model selector
- Quiz answers remain hidden until submission
"""

import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_FALLBACK = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash")

# These are only preferences/fallbacks. The app also discovers models that
# are actually available to the current API key at runtime.
CONFIGURED_MODELS = [
    x.strip()
    for x in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        ",".join([
            DEFAULT_MODEL,
            DEFAULT_FALLBACK,
            "gemini-3.7-flash",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]),
    ).split(",")
    if x.strip()
]

API_KEY = os.getenv("GEMINI_API_KEY")


@st.cache_data(ttl=300, show_spinner=False)
def discover_available_models(api_key: str):
    """Discover generation-capable Gemini models and order sensible fallbacks.

    We prefer models explicitly configured by the user, but if one of those
    names is unavailable we also use real generation-capable Flash models
    returned by the API. This avoids a hard-coded model name causing the whole
    application to fail.
    """
    if not api_key:
        return CONFIGURED_MODELS.copy()

    try:
        client = genai.Client(api_key=api_key)
        discovered = []

        for model_info in client.models.list():
            name = str(getattr(model_info, "name", "") or "")
            name = name.removeprefix("models/").strip()
            actions = getattr(model_info, "supported_actions", None) or []
            actions = [str(a) for a in actions]

            if not name.startswith("gemini-"):
                continue

            # Some SDK versions expose supported_actions; older versions may
            # omit it. If present, require generateContent.
            if actions and not any("generatecontent" in a.lower() for a in actions):
                continue

            if name not in discovered:
                discovered.append(name)

        if not discovered:
            print("[AI DEBUG] API returned no generation-capable Gemini models.")
            return CONFIGURED_MODELS.copy()

        # User-configured models get first priority when they really exist.
        preferred = [m for m in CONFIGURED_MODELS if m in discovered]

        # Prefer Flash models for this app: they are normally faster and more
        # appropriate for repeated study-pack generation. Keep all discovered
        # models as a final safety net.
        flash_models = [
            m for m in discovered
            if "flash" in m.lower() and m not in preferred
        ]
        other_models = [
            m for m in discovered
            if m not in preferred and m not in flash_models
        ]

        ordered = preferred + sorted(flash_models) + sorted(other_models)

        print(f"[AI DEBUG] Available Gemini generation models: {ordered}")
        return ordered

    except Exception as exc:
        # Discovery itself must never break the app. Generation will still
        # attempt the configured names and report only a generic UI message.
        print(f"[AI DEBUG] Model discovery failed: {type(exc).__name__}: {exc}")
        return CONFIGURED_MODELS.copy()


def get_available_models():
    return discover_available_models(API_KEY) if API_KEY else CONFIGURED_MODELS.copy()


AVAILABLE_MODELS = get_available_models()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

STATE_DEFAULTS = {
    "study_pack": None,
    "quiz": None,
    "quiz_submitted": False,
    "quiz_answers": {},
    "topic": "",
    "pending_request": None,
    "last_error": None,
    "last_model": None,
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def error_text(exc: Exception) -> str:
    return str(exc)


def is_temporary_error(exc: Exception) -> bool:
    """
    Detect temporary capacity/rate-limit failures.

    Gemini may return HTTP 503 when the selected model is temporarily
    unavailable/high demand and 429 for rate limiting/quota-related cases.
    """
    text = error_text(exc).lower()

    temporary_markers = [
        "503",
        "service unavailable",
        "unavailable",
        "high demand",
        "currently experiencing high demand",
        "429",
        "resource exhausted",
        "rate limit",
        "too many requests",
        "temporarily",
        "overloaded",
    ]

    return any(marker in text for marker in temporary_markers)


def is_model_error(exc: Exception) -> bool:
    text = error_text(exc).lower()
    return (
        "404" in text
        or "model not found" in text
        or "not found" in text and "model" in text
        or "invalid model" in text
    )


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def get_client() -> genai.Client:
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Add it to .env locally or Streamlit Secrets."
        )
    return genai.Client(api_key=API_KEY)


def parse_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini did not return valid JSON.")

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse Gemini JSON: {exc}") from exc


def upload_source_file(client: genai.Client, uploaded_file):
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower()
    supported = {".pdf", ".txt", ".md", ".csv", ".json", ".py"}

    if suffix not in supported:
        raise ValueError(
            "Supported source files: PDF, TXT, MD, CSV, JSON and PY."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        return client.files.upload(file=temp_path)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def call_gemini_once(
    client: genai.Client,
    model: str,
    prompt: str,
    uploaded_file=None,
) -> Dict[str, Any]:
    contents = []

    gemini_file = upload_source_file(client, uploaded_file)

    if gemini_file is not None:
        contents.append(gemini_file)
        contents.append(
            "Use the uploaded source document as the primary reference. "
            "Do not invent source-specific facts."
        )

    contents.append(prompt)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=20000,
        ),
    )

    return parse_json(response.text)


def generate_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    uploaded_file=None,
    max_attempts: int = 3,
) -> Tuple[Dict[str, Any], str]:
    """Retry transient failures on one model using exponential backoff."""
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = call_gemini_once(
                client=client,
                model=model,
                prompt=prompt,
                uploaded_file=uploaded_file,
            )
            return result, model

        except Exception as exc:
            last_exception = exc

            print(
                f"[AI DEBUG] Model {model} attempt {attempt}/{max_attempts} "
                f"failed: {type(exc).__name__}: {exc}"
            )

            # Invalid/auth/request errors should not be retried on the same
            # model. The caller will decide whether to try another model.
            if not is_temporary_error(exc):
                raise

            if attempt < max_attempts:
                delay = min(2 ** attempt, 8) + random.uniform(0, 1)
                time.sleep(delay)

    raise last_exception


def generate_with_fallbacks(
    client: genai.Client,
    primary_model: str,
    fallback_model: str,
    prompt: str,
    uploaded_file=None,
    max_attempts: int = 3,
) -> Tuple[Dict[str, Any], str]:
    """Try the selected model, then automatically try valid discovered models.

    Provider exceptions are never returned to Streamlit. They are logged only
    to the server console.
    """
    candidates = []

    for model in [primary_model, fallback_model, *AVAILABLE_MODELS]:
        if model and model not in candidates:
            candidates.append(model)

    failures = []

    for model in candidates:
        try:
            print(f"[AI DEBUG] Trying model: {model}")
            result, used_model = generate_with_retry(
                client=client,
                model=model,
                prompt=prompt,
                uploaded_file=uploaded_file,
                max_attempts=max_attempts,
            )
            print(f"[AI DEBUG] Generation succeeded with: {model}")
            return result, used_model

        except Exception as exc:
            failures.append((model, exc))
            print(
                f"[AI DEBUG] Skipping model {model}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

    # Never expose provider details to the UI.
    raise RuntimeError("ALL_MODELS_FAILED")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def study_pack_prompt(
    topic: str,
    level: str,
    language: str,
    pack_type: str,
    detail: str,
    question_count: int,
) -> str:
    return f"""
You are an expert teacher, instructional designer and examination specialist.

Create a complete study pack and an interactive quiz.

TOPIC: {topic}
LEARNER LEVEL: {level}
LANGUAGE: {language}
PACK TYPE: {pack_type}
DETAIL LEVEL: {detail}
QUIZ QUESTIONS: {question_count}

Requirements:
- Explain the topic accurately and progressively.
- Match the selected learner level.
- Use practical examples where useful.
- Focus on understanding and application.
- Avoid repetitive filler.
- If a source document is provided, use it as the primary reference.
- Do not invent source-specific facts.
- Quiz questions must be directly related to the topic.
- Correct answers must be unambiguous.
- Do not reveal quiz answers in the question text.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "title": "string",
  "summary": "string",
  "learning_objectives": ["string"],
  "notes": [
    {{
      "heading": "string",
      "content": "markdown string",
      "key_points": ["string"],
      "examples": ["string"]
    }}
  ],
  "flashcards": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ],
  "short_questions": ["string"],
  "study_plan": [
    {{
      "day": 1,
      "focus": "string",
      "activity": "string"
    }}
  ],
  "quiz": {{
    "title": "string",
    "instructions": "string",
    "questions": [
      {{
        "question": "string",
        "options": [
          "A. string",
          "B. string",
          "C. string",
          "D. string"
        ],
        "correct_answer": "A",
        "explanation": "string"
      }}
    ]
  }}
}}

The quiz must contain exactly {question_count} questions.
correct_answer must contain only A, B, C or D.
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_study_pack(pack: Dict[str, Any]) -> None:
    st.header(pack.get("title", "Study Pack"))

    if pack.get("summary"):
        st.info(pack["summary"])

    objectives = pack.get("learning_objectives", [])
    if objectives:
        st.subheader("🎯 Learning Objectives")
        for item in objectives:
            st.markdown(f"- {item}")

    notes = pack.get("notes", [])
    if notes:
        st.subheader("📚 Study Notes")
        for index, note in enumerate(notes, 1):
            with st.expander(
                f"{index}. {note.get('heading', 'Topic')}",
                expanded=index == 1,
            ):
                st.markdown(note.get("content", ""))

                if note.get("key_points"):
                    st.markdown("**Key Points**")
                    for point in note["key_points"]:
                        st.markdown(f"- {point}")

                if note.get("examples"):
                    st.markdown("**Examples**")
                    for example in note["examples"]:
                        st.markdown(f"- {example}")

    if pack.get("flashcards"):
        st.subheader("🧠 Flashcards")
        for i, card in enumerate(pack["flashcards"], 1):
            with st.expander(f"Flashcard {i}"):
                st.markdown(f"**Q:** {card.get('question', '')}")
                st.markdown(f"**A:** {card.get('answer', '')}")

    if pack.get("short_questions"):
        st.subheader("✍️ Short Questions")
        for i, question in enumerate(pack["short_questions"], 1):
            st.markdown(f"**{i}.** {question}")

    if pack.get("study_plan"):
        st.subheader("🗓️ Suggested Study Plan")
        for item in pack["study_plan"]:
            st.markdown(
                f"**Day {item.get('day')}:** "
                f"{item.get('focus')} — {item.get('activity')}"
            )


def render_quiz_result(
    quiz: Dict[str, Any],
    user_answers: Dict[int, str],
) -> None:
    questions = quiz.get("questions", [])

    score = 0

    for index, question in enumerate(questions):
        correct = str(question.get("correct_answer", "")).strip()
        selected = str(user_answers.get(index, "")).strip()

        if selected == correct:
            score += 1

    total = len(questions)
    percentage = (score / total * 100) if total else 0

    if percentage >= 80:
        st.success(f"🏆 Score: {score}/{total} — {percentage:.0f}%")
    elif percentage >= 60:
        st.warning(f"👍 Score: {score}/{total} — {percentage:.0f}%")
    else:
        st.error(f"📖 Score: {score}/{total} — {percentage:.0f}%")

    st.subheader("📋 Answer Review")

    for index, question in enumerate(questions):
        selected = user_answers.get(index, "Not answered")
        correct = question.get("correct_answer", "")
        is_correct = selected == correct

        if is_correct:
            st.success(
                f"**Question {index + 1}: Correct**\n\n"
                f"Your answer: **{selected}**"
            )
        else:
            st.error(
                f"**Question {index + 1}: Incorrect**\n\n"
                f"Your answer: **{selected}**\n\n"
                f"Correct answer: **{correct}**"
            )

        if question.get("explanation"):
            st.info(
                f"**Explanation:** {question['explanation']}"
            )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎓 AI Study Pack Generator")
st.caption(
    "Study Pack + Interactive Quiz • Resilient Gemini model fallback"
)

if not API_KEY:
    st.warning(
        "GEMINI_API_KEY is not configured. Add it to .env locally or "
        "Streamlit Secrets for deployment."
    )

with st.sidebar:
    st.header("⚙️ Study Settings")

    level = st.selectbox(
        "Learner Level",
        ["Beginner", "Intermediate", "Advanced", "Exam Preparation"],
    )

    language = st.selectbox(
        "Language",
        ["English", "Urdu", "English + Urdu"],
    )

    pack_type = st.selectbox(
        "Study Pack Type",
        [
            "Complete Study Pack",
            "Revision Pack",
            "Exam Preparation Pack",
            "Quick Notes + Quiz",
            "Flashcards + Quiz",
        ],
    )

    detail = st.selectbox(
        "Detail Level",
        ["Concise", "Balanced", "Detailed"],
        index=1,
    )

    question_count = st.slider(
        "Quiz Questions",
        min_value=5,
        max_value=30,
        value=10,
    )

    st.divider()
    st.subheader("🤖 Gemini Models")

    model_options = ["Automatic fallback"] + AVAILABLE_MODELS

    if DEFAULT_MODEL in model_options:
        default_index = model_options.index(DEFAULT_MODEL)
    else:
        default_index = 0

    selected_model = st.selectbox(
        "Primary Model",
        model_options,
        index=default_index,
        help="The selected model is tried first. The app automatically falls back to other available models if needed.",
    )

    selected_primary = DEFAULT_MODEL if selected_model == "Automatic fallback" else selected_model

    fallback_options = [
        model for model in AVAILABLE_MODELS
        if model != selected_primary
    ]

    if not fallback_options:
        fallback_options = [DEFAULT_FALLBACK]

    fallback_default = (
        DEFAULT_FALLBACK
        if DEFAULT_FALLBACK in fallback_options
        else fallback_options[0]
    )

    fallback_model = st.selectbox(
        "Fallback Model",
        fallback_options,
        index=fallback_options.index(fallback_default),
        help="This model is tried automatically if the primary model fails.",
    )

    st.caption(
        "Temporary failures are retried automatically. If a model is busy or unavailable, the app silently moves to the next available model."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📄 Optional Source Document",
        type=["pdf", "txt", "md", "csv", "json", "py"],
    )

    st.divider()
    st.markdown(
        "**Supported:** PDF, TXT, MD, CSV, JSON, PY\n\n"
        "**Languages:** English, Urdu, bilingual"
    )


topic = st.text_area(
    "📘 What topic do you want to study?",
    value=st.session_state.topic,
    placeholder=(
        "Examples:\n"
        "• BGP Route Selection\n"
        "• Python OOP\n"
        "• Photosynthesis\n"
        "• OSI Model"
    ),
    height=120,
)

generate_col, reset_col = st.columns([3, 1])

with generate_col:
    generate_button = st.button(
        "🚀 Generate Study Pack + Quiz",
        type="primary",
        use_container_width=True,
    )

with reset_col:
    if st.button("🔄 Reset", use_container_width=True):
        for key, value in STATE_DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()



def execute_generation(model: str):
    if not topic.strip():
        st.error("Please enter a topic.")
        return

    if not API_KEY:
        st.error("Please configure GEMINI_API_KEY first.")
        return

    prompt = study_pack_prompt(
        topic=topic.strip(),
        level=level,
        language=language,
        pack_type=pack_type,
        detail=detail,
        question_count=question_count,
    )

    try:
        client = get_client()
        primary = model if model != "Automatic fallback" else DEFAULT_MODEL

        fallback = fallback_model if fallback_model else DEFAULT_FALLBACK

        with st.spinner("Generating your study pack and quiz…"):
            result, used_model = generate_with_fallbacks(
                client=client,
                primary_model=primary,
                fallback_model=fallback,
                prompt=prompt,
                uploaded_file=uploaded_file,
                max_attempts=3,
            )

        if "quiz" not in result:
            raise ValueError("INVALID_STUDY_PACK")

        st.session_state.study_pack = result
        st.session_state.quiz = result["quiz"]
        st.session_state.quiz_submitted = False
        st.session_state.quiz_answers = {}
        st.session_state.topic = topic.strip()
        st.session_state.pending_request = None
        st.session_state.last_error = None
        st.session_state.last_model = used_model

        st.rerun()

    except Exception as exc:
        # Full provider error is deliberately logged only on the server.
        print(
            f"[AI DEBUG] Final generation failure: "
            f"{type(exc).__name__}: {exc}"
        )

        st.session_state.last_error = "GENERATION_FAILED"
        st.session_state.pending_request = {
            "primary_model": model,
            "fallback_model": fallback_model,
        }

        # Generic UI only — never display Gemini's raw error.
        st.error("❌ We couldn't generate the study pack right now. Please try again.")


# ---------------------------------------------------------------------------
# Initial generation
# ---------------------------------------------------------------------------

if generate_button:
    execute_generation(selected_model)


# ---------------------------------------------------------------------------
# Retry controls
# ---------------------------------------------------------------------------

if st.session_state.last_error and st.session_state.pending_request:
    st.divider()
    retry_col, clear_col = st.columns([3, 1])

    with retry_col:
        if st.button(
            "🔄 Try Again",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.last_error = None
            execute_generation(
                st.session_state.pending_request.get(
                    "primary_model", DEFAULT_MODEL
                )
            )

    with clear_col:
        if st.button("✖ Clear", use_container_width=True):
            st.session_state.last_error = None
            st.session_state.pending_request = None
            st.rerun()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.study_pack:
    study_tab, quiz_tab, json_tab = st.tabs(
        ["📚 Study Pack", "📝 Take Quiz", "🔧 JSON"]
    )

    with study_tab:
        if st.session_state.last_model:
            st.caption(
                f"Generated with: `{st.session_state.last_model}`"
            )
        render_study_pack(st.session_state.study_pack)

    with quiz_tab:
        quiz = st.session_state.quiz
        questions = quiz.get("questions", [])

        st.header(quiz.get("title", "Topic Quiz"))

        if quiz.get("instructions"):
            st.info(quiz["instructions"])

        if not st.session_state.quiz_submitted:
            st.markdown(
                "**Instructions:** Answer all questions and submit the quiz. "
                "Correct answers remain hidden until submission."
            )

            with st.form("quiz_form"):
                answers = {}

                for index, question in enumerate(questions):
                    st.markdown(
                        f"### {index + 1}. "
                        f"{question.get('question', '')}"
                    )

                    options = question.get("options", [])

                    selected = st.radio(
                        "Select your answer:",
                        options,
                        index=None,
                        key=f"quiz_q_{index}",
                        label_visibility="collapsed",
                    )

                    answers[index] = selected

                submitted = st.form_submit_button(
                    "✅ Submit Quiz",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                missing = [
                    str(i + 1)
                    for i, answer in answers.items()
                    if not answer
                ]

                if missing:
                    st.warning(
                        "Please answer every question before submitting. "
                        f"Missing: {', '.join(missing)}"
                    )
                else:
                    normalized = {}

                    for index, selected in answers.items():
                        normalized[index] = (
                            selected.split(".", 1)[0].strip()
                        )

                    st.session_state.quiz_answers = normalized
                    st.session_state.quiz_submitted = True
                    st.rerun()

        else:
            render_quiz_result(
                quiz,
                st.session_state.quiz_answers,
            )

            if st.button("🔁 Retry Quiz", type="primary"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.rerun()

    with json_tab:
        st.json(st.session_state.study_pack)

        st.download_button(
            "⬇️ Download Study Pack JSON",
            data=json.dumps(
                st.session_state.study_pack,
                ensure_ascii=False,
                indent=2,
            ),
            file_name="study_pack.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    st.info(
        "Enter a topic and click **Generate Study Pack + Quiz**."
    )
