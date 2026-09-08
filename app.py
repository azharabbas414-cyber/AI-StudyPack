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

# These are configurable through the UI and environment variables.
AVAILABLE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

API_KEY = os.getenv("GEMINI_API_KEY")

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
    """
    Retry temporary failures against the same model.

    Backoff:
        ~2 sec
        ~4 sec
        ~8 sec

    The UI also provides explicit same-model and fallback-model retry buttons.
    """
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

            # Do not waste retries on authentication, invalid request,
            # invalid model, malformed response, etc.
            if not is_temporary_error(exc):
                raise

            if attempt >= max_attempts:
                break

            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

    raise RuntimeError(
        f"Model `{model}` is temporarily unavailable after "
        f"{max_attempts} attempts. Original error: {last_exception}"
    )


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

    default_index = (
        model_options.index(DEFAULT_MODEL)
        if DEFAULT_MODEL in model_options
        else 1
    )

    selected_model = st.selectbox(
        "Primary Model",
        model_options,
        index=default_index,
        help=(
            "The primary model used for generation. Automatic fallback "
            "will retry and then use the fallback model if necessary."
        ),
    )

    fallback_options = [
        model for model in AVAILABLE_MODELS
        if model != (
            DEFAULT_MODEL if selected_model == "Automatic fallback"
            else selected_model
        )
    ]

    fallback_default = (
        DEFAULT_FALLBACK
        if DEFAULT_FALLBACK in fallback_options
        else fallback_options[0]
    )

    fallback_model = st.selectbox(
        "Fallback Model",
        fallback_options,
        index=fallback_options.index(fallback_default),
        help="Used when the primary model remains unavailable.",
    )

    st.caption(
        "Temporary 503/429 errors automatically retry the same model "
        "before the fallback option is offered."
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


def selected_primary_model() -> str:
    if selected_model == "Automatic fallback":
        return DEFAULT_MODEL
    return selected_model


def save_pending_request(model: str):
    st.session_state.pending_request = {
        "topic": topic.strip(),
        "level": level,
        "language": language,
        "pack_type": pack_type,
        "detail": detail,
        "question_count": question_count,
        "uploaded_file": uploaded_file,
        "primary_model": model,
        "fallback_model": fallback_model,
    }


def execute_generation(model: str, auto_fallback: bool = True):
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

        with st.spinner(
            f"Generating with `{model}`. Please wait..."
        ):
            result, used_model = generate_with_retry(
                client=client,
                model=model,
                prompt=prompt,
                uploaded_file=uploaded_file,
                max_attempts=3,
            )

        if "quiz" not in result:
            raise ValueError(
                "The AI response did not contain the required quiz."
            )

        st.session_state.study_pack = result
        st.session_state.quiz = result["quiz"]
        st.session_state.quiz_submitted = False
        st.session_state.quiz_answers = {}
        st.session_state.topic = topic.strip()
        st.session_state.pending_request = None
        st.session_state.last_error = None
        st.session_state.last_model = used_model

        st.success(f"Generated successfully using `{used_model}`.")
        st.rerun()

    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.session_state.pending_request = {
            "topic": topic.strip(),
            "level": level,
            "language": language,
            "pack_type": pack_type,
            "detail": detail,
            "question_count": question_count,
            "uploaded_file": uploaded_file,
            "primary_model": model,
            "fallback_model": fallback_model,
        }

        st.error(f"Generation failed: {exc}")


# ---------------------------------------------------------------------------
# Initial generation
# ---------------------------------------------------------------------------

if generate_button:
    execute_generation(selected_primary_model())


# ---------------------------------------------------------------------------
# Retry controls
# ---------------------------------------------------------------------------

if st.session_state.last_error and st.session_state.pending_request:
    error = st.session_state.last_error
    pending = st.session_state.pending_request

    st.divider()
    st.subheader("⚠️ Gemini Generation Failed")

    if is_temporary_error(Exception(error)):
        st.warning(
            "The selected Gemini model appears to be temporarily busy or "
            "rate-limited. You can retry the same model or switch to the "
            "fallback model."
        )

        retry1, retry2, retry3 = st.columns(3)

        with retry1:
            if st.button(
                f"🔁 Retry Same Model\n{pending['primary_model']}",
                use_container_width=True,
            ):
                st.session_state.last_error = None
                execute_generation(pending["primary_model"])

        with retry2:
            if st.button(
                f"🔄 Try Fallback\n{pending['fallback_model']}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.last_error = None
                execute_generation(pending["fallback_model"])

        with retry3:
            if st.button(
                "⚙️ Clear Error",
                use_container_width=True,
            ):
                st.session_state.last_error = None
                st.session_state.pending_request = None
                st.rerun()
    else:
        st.info(
            "This does not look like a temporary capacity error. "
            "Check the API key, model name, request, or uploaded document."
        )


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
