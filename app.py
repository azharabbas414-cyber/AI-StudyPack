"""
AI Study Pack Generator
-----------------------
Streamlit application for GitHub + Streamlit Community Cloud.

Features:
- AI-generated study packs
- Topic-based quizzes
- Quiz submission and scoring
- Answers/explanations revealed only after submission
- Optional source document support
- English / Urdu / bilingual output
- Google Gemini via the modern google-genai SDK

Run locally:
    streamlit run app.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Gemini's current flagship Flash model as of September 2026.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client() -> genai.Client:
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to .env locally or "
            "Streamlit Secrets when deploying."
        )
    return genai.Client(api_key=API_KEY)


def parse_json(text: str) -> Dict[str, Any]:
    """Safely parse JSON returned by Gemini."""
    if not text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Handle accidental text before/after JSON.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini did not return valid JSON.")

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse Gemini JSON: {exc}") from exc


def upload_source_file(client: genai.Client, uploaded_file):
    """Upload a supported source document to Gemini."""
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower()
    supported = {".pdf", ".txt", ".md", ".csv", ".json", ".py"}

    if suffix not in supported:
        raise ValueError(
            "Supported source files are PDF, TXT, MD, CSV, JSON and PY."
        )

    # Streamlit UploadedFile is file-like; Gemini accepts a file path or
    # file-like object depending on SDK version. Use a temporary file path
    # for maximum compatibility.
    import tempfile

    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        return client.files.upload(file=temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def generate_content(
    client: genai.Client,
    prompt: str,
    uploaded_file=None,
) -> Dict[str, Any]:
    contents = []

    gemini_file = upload_source_file(client, uploaded_file)
    if gemini_file is not None:
        contents.append(gemini_file)
        contents.append(
            "Use the uploaded source document as the primary reference. "
            "Do not contradict it unless you explicitly identify a correction."
        )

    contents.append(prompt)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=20000,
        ),
    )

    return parse_json(response.text)


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

                key_points = note.get("key_points", [])
                if key_points:
                    st.markdown("**Key Points**")
                    for point in key_points:
                        st.markdown(f"- {point}")

                examples = note.get("examples", [])
                if examples:
                    st.markdown("**Examples**")
                    for example in examples:
                        st.markdown(f"- {example}")

    flashcards = pack.get("flashcards", [])
    if flashcards:
        st.subheader("🧠 Flashcards")
        for i, card in enumerate(flashcards, 1):
            with st.expander(f"Flashcard {i}"):
                st.markdown(f"**Q:** {card.get('question', '')}")
                st.markdown(f"**A:** {card.get('answer', '')}")

    short_questions = pack.get("short_questions", [])
    if short_questions:
        st.subheader("✍️ Short Questions")
        for i, question in enumerate(short_questions, 1):
            st.markdown(f"**{i}.** {question}")

    study_plan = pack.get("study_plan", [])
    if study_plan:
        st.subheader("🗓️ Suggested Study Plan")
        for item in study_plan:
            st.markdown(
                f"**Day {item.get('day', '')}:** "
                f"{item.get('focus', '')} — {item.get('activity', '')}"
            )


def render_quiz_result(quiz: Dict[str, Any], user_answers: Dict[int, str]) -> None:
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
                f"Question {index + 1}: Correct\n\n"
                f"Your answer: {selected}"
            )
        else:
            st.error(
                f"Question {index + 1}: Incorrect\n\n"
                f"Your answer: {selected}\n\n"
                f"Correct answer: {correct}"
            )

        explanation = question.get("explanation", "")
        if explanation:
            st.info(f"**Explanation:** {explanation}")


# ---------------------------------------------------------------------------
# Prompts
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

Create a high-quality study pack for this topic:

TOPIC: {topic}
LEARNER LEVEL: {level}
LANGUAGE: {language}
PACK TYPE: {pack_type}
DETAIL LEVEL: {detail}
QUIZ QUESTION COUNT: {question_count}

Educational requirements:
- Explain concepts accurately and progressively.
- Match the learner's selected level.
- Use practical examples when appropriate.
- Avoid unnecessary repetition.
- Focus on understanding, application and retention.
- If a source document is supplied, use it as the primary reference.
- Do not fabricate source-specific facts.
- The quiz must test the same topic and concepts covered in the study pack.
- Correct answers must be unambiguous.

Return ONLY valid JSON with this structure:

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
The correct_answer must be only A, B, C, or D.
Do not reveal the correct answers in quiz question text.
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = {
    "study_pack": None,
    "quiz": None,
    "quiz_submitted": False,
    "quiz_answers": {},
    "topic": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎓 AI Study Pack Generator")
st.caption(
    f"Powered by Google Gemini {MODEL_NAME} • "
    "Study Pack + Interactive Quiz"
)

if not API_KEY:
    st.warning(
        "GEMINI_API_KEY is not configured yet. Add it to `.env` for local use "
        "or Streamlit Secrets for deployment."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Study Settings")

    level = st.selectbox(
        "Learner Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Exam Preparation",
        ],
    )

    language = st.selectbox(
        "Language",
        [
            "English",
            "Urdu",
            "English + Urdu",
        ],
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
        [
            "Concise",
            "Balanced",
            "Detailed",
        ],
        index=1,
    )

    question_count = st.slider(
        "Quiz Questions",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
    )

    uploaded_file = st.file_uploader(
        "Optional Source Document",
        type=["pdf", "txt", "md", "csv", "json", "py"],
        help=(
            "The AI will use the uploaded material as the primary source "
            "for the study pack and quiz."
        ),
    )

    st.divider()
    st.markdown("### Supported")
    st.markdown(
        "- 📄 PDF / TXT / MD / CSV / JSON / PY\n"
        "- 🇬🇧 English\n"
        "- 🇵🇰 Urdu\n"
        "- 🧠 Flashcards\n"
        "- 📝 Interactive quizzes\n"
        "- 📊 Quiz scoring"
    )


# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------

topic = st.text_area(
    "📘 What topic do you want to study?",
    value=st.session_state.topic,
    placeholder=(
        "Examples:\n"
        "• BGP Route Selection\n"
        "• Python Object-Oriented Programming\n"
        "• Photosynthesis\n"
        "• OSI Model\n"
        "• Algebra"
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
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

if generate_button:
    if not topic.strip():
        st.error("Please enter a topic.")
    elif not API_KEY:
        st.error("Please configure GEMINI_API_KEY first.")
    else:
        try:
            client = get_client()

            with st.spinner(
                f"Gemini {MODEL_NAME} is creating your study pack and quiz..."
            ):
                result = generate_content(
                    client,
                    study_pack_prompt(
                        topic=topic.strip(),
                        level=level,
                        language=language,
                        pack_type=pack_type,
                        detail=detail,
                        question_count=question_count,
                    ),
                    uploaded_file=uploaded_file,
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

            st.success("Study pack and quiz generated successfully.")
            st.rerun()

        except Exception as exc:
            st.error(f"Generation failed: {exc}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.study_pack:
    study_tab, quiz_tab, json_tab = st.tabs(
        ["📚 Study Pack", "📝 Take Quiz", "🔧 JSON"]
    )

    with study_tab:
        render_study_pack(st.session_state.study_pack)

    with quiz_tab:
        quiz = st.session_state.quiz
        questions = quiz.get("questions", [])

        st.header(quiz.get("title", "Topic Quiz"))

        if quiz.get("instructions"):
            st.info(quiz["instructions"])

        if not st.session_state.quiz_submitted:
            st.markdown(
                "**Instructions:** Select one answer for every question, "
                "then click **Submit Quiz**. Correct answers are hidden "
                "until submission."
            )

            with st.form("quiz_form"):
                answers = {}

                for index, question in enumerate(questions):
                    st.markdown(
                        f"### {index + 1}. {question.get('question', '')}"
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
                        "Please answer all questions before submitting. "
                        f"Missing: {', '.join(missing)}"
                    )
                else:
                    # Convert "A. answer" to "A" for comparison.
                    normalized = {}
                    for index, selected in answers.items():
                        normalized[index] = selected.split(".", 1)[0].strip()

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
        "Enter a topic and click **Generate Study Pack + Quiz** to begin."
    )

    st.markdown(
        """
### What this application can generate

**Study Material**
- 📚 Structured study notes
- 🎯 Learning objectives
- 💡 Key points
- 🔎 Examples
- 🧠 Flashcards
- ✍️ Short questions
- 🗓️ Study plan

**Interactive Quiz**
- Multiple-choice questions based on your topic
- Answers hidden while attempting
- Submit after completing the quiz
- Automatic score calculation
- Percentage
- Correct/incorrect review
- Correct answers
- Explanations
- Retry quiz

**Example**

Enter:

> `BGP Route Selection`

The AI creates the learning material and a quiz specifically about BGP Route Selection.

You can also upload a PDF or other supported source document and generate the study material and quiz from that material.
"""
    )
