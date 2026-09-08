# 🎓 AI Study Pack Generator

An AI-powered study assistant built with **Python + Streamlit + Google Gemini**.

The application generates a complete study pack from any topic and also creates an interactive quiz. The student attempts the quiz first; answers and explanations are revealed only after submission.

## ✨ Features

### 📚 Study Pack
- Topic-based study notes
- Learning objectives
- Key points
- Examples
- Flashcards
- Short-answer questions
- Suggested study plan

### 📝 Interactive Quiz
- Quiz generated specifically for the selected topic
- 5–30 multiple-choice questions
- Four options per question
- Correct answers hidden during the attempt
- Submit quiz when finished
- Automatic score calculation
- Percentage score
- Correct/incorrect answer review
- Correct answer shown after submission
- Explanation for every question
- Retry quiz

### 📄 Source Material

Optional source files:

- PDF
- TXT
- Markdown
- CSV
- JSON
- Python (`.py`)

When a source file is supplied, Gemini is instructed to use it as the primary reference.

### 🌐 Languages

- English
- Urdu
- English + Urdu

### 🤖 AI Model

The application uses Google's modern `google-genai` Python SDK and defaults to:

```text
gemini-3.8-flash
```

The model can be changed without editing the source code:

```text
GEMINI_MODEL=your-model-id
```

Google currently lists Gemini 3.8 Flash as its most capable Flash model and a generally available production model.

---

# 🚀 Run Locally

## 1. Install Python

Use Python 3.10 or newer.

Check:

```bash
python --version
```

## 2. Clone the GitHub repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-study-pack-generator.git
cd ai-study-pack-generator
```

## 3. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Create Gemini API key

Create an API key through Google AI Studio.

Do NOT put the API key directly inside `app.py`.

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.8-flash
```

## 6. Run Streamlit

```bash
streamlit run app.py
```

The browser should open the application.

---

# ☁️ Deploy on Streamlit Community Cloud

## 1. Create GitHub repository

Create a new repository, for example:

```text
ai-study-pack-generator
```

Upload:

```text
app.py
requirements.txt
README.md
```

You can also include:

```text
.env.example
.gitignore
```

## 2. NEVER upload `.env`

Your `.gitignore` should contain:

```text
.env
.venv/
__pycache__/
*.pyc
.DS_Store
```

Your Gemini API key must never be committed to GitHub.

## 3. Open Streamlit Community Cloud

Sign in using GitHub and create a new application.

Select:

```text
Repository: your GitHub repository
Branch: main
Main file path: app.py
```

Deploy the application.

## 4. Add Gemini API key

In the Streamlit application settings, open:

```text
Settings → Secrets
```

Add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-3.8-flash"
```

Save the secrets and reboot/redeploy the application.

The application reads the API key from the environment/Streamlit runtime and does not expose it in the UI.

---

# 🧪 How to Use

## Example 1 — Networking

Enter:

```text
BGP Route Selection
```

Choose:

```text
Level: Intermediate
Language: English
Quiz Questions: 10
```

Click:

```text
Generate Study Pack + Quiz
```

The application generates:

- BGP study notes
- Learning objectives
- Key points
- Examples
- Flashcards
- Short questions
- Study plan
- 10-question BGP quiz

Go to:

```text
Take Quiz
```

Answer all questions.

Click:

```text
Submit Quiz
```

Only then will the application show:

- Score
- Percentage
- Your answers
- Correct answers
- Explanations

---

# 🧠 Example 2 — Any Academic Topic

Try:

```text
Photosynthesis
```

or:

```text
Newton's Laws of Motion
```

or:

```text
Python Object Oriented Programming
```

or:

```text
OSI Model
```

The quiz is generated dynamically from the selected topic.

---

# 📄 Example 3 — Study From a PDF

Enter a topic and upload a PDF.

The application sends the source document to Gemini and asks the model to use it as the primary reference.

This is useful for:

- Course notes
- Training material
- Certification material
- Lecture notes
- Technical documentation
- Exam preparation material

---

# 🔐 Security

Never commit this:

```text
GEMINI_API_KEY=real-key-here
```

to GitHub.

Use:

```text
.env
```

locally and Streamlit Secrets in production.

Recommended `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
.DS_Store
```

---

# 📁 Project Structure

```text
ai-study-pack-generator/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

# 🔧 Future Roadmap

The application can later be expanded with:

1. PDF study-pack export
2. DOCX export
3. Printable quiz
4. Timed examination mode
5. Quiz difficulty selection
6. Easy / Medium / Hard questions
7. Retry only incorrect questions
8. Question bank
9. Student progress tracking
10. Score history
11. Topic performance analytics
12. AI tutor/chat
13. Voice-based learning
14. Spaced repetition
15. Adaptive quizzes
16. Teacher mode
17. Curriculum mapping
18. Image/diagram generation
19. Web-grounded research
20. User accounts and saved study packs

---

# 🛠️ Troubleshooting

## `GEMINI_API_KEY is not configured`

Check that `.env` exists locally:

```text
GEMINI_API_KEY=YOUR_KEY
```

For Streamlit Cloud, check:

```text
Settings → Secrets
```

## `ModuleNotFoundError`

Run:

```bash
pip install -r requirements.txt
```

## Gemini model error

The model is configurable:

```text
GEMINI_MODEL=gemini-3.8-flash
```

If Google changes model availability in the future, update the model ID in Streamlit Secrets or `.env` without changing the application code.

---

# 📌 Current Architecture

```text
                ┌─────────────────────┐
                │       Student       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Streamlit UI    │
                └──────────┬──────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Study Pack Mode               Quiz Mode
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Google Gemini  │
                  │  3.8 Flash      │
                  └────────┬────────┘
                           │
                           ▼
                  Structured JSON
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Study Material             Quiz Questions
                                      │
                                      ▼
                              Student Attempts
                                      │
                                      ▼
                                Submit Quiz
                                      │
                                      ▼
                            Score + Answer Review
```

---

## License

You can choose an open-source license for the GitHub repository, such as MIT, before publishing.
