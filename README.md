# NEET AI Tutor

AI-powered RAG chatbot for NEET preparation. Students upload their own NEET/AIPMT papers, answer keys, explanations, NCERT notes, or public educational PDFs, then ask questions in a ChatGPT-style Streamlit interface.

The app does **not** scrape copyrighted websites. It only learns from user-uploaded documents.

## What It Does

- Upload PDF files containing NEET questions, answers, explanations, and notes.
- Extract text with PyMuPDF, with pdfplumber fallback.
- Detect subject, chapter, question number, options, answer markers, and explanation markers where possible.
- Chunk NEET material into retrieval-friendly records.
- Generate embeddings with Gemini by default, with optional Sentence Transformers support.
- Store vectors in ChromaDB.
- Retrieve similar NEET questions and explanations.
- Send retrieved context to Gemini for teacher-style answers.
- Show step-by-step solutions, formulas, wrong-option analysis, shortcuts, confidence, and source references.
- Support English, Hindi, and Nepali explanations.
- Ask by voice with language/accent selection for English India, English US, English UK, Hindi India, and Nepali Nepal.
- Provide chat history, bookmarks, favorite chats, daily quiz, mock tests, analytics, and leaderboard.
- Use a dark, mobile-responsive Streamlit UI.

## Folder Structure

```text
data/
pdfs/
embeddings/
chatbot/
utils/
.env
app.py
requirements.txt
README.md
```

## Local Setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_google_gemini_api_key
EMBEDDING_PROVIDER=gemini
```

Run:

```bash
streamlit run app.py
```

## Embedding Options

Default:

```env
EMBEDDING_PROVIDER=gemini
```

Optional local Sentence Transformers mode:

```bash
pip install sentence-transformers
```

```env
EMBEDDING_PROVIDER=sentence_transformers
```

Gemini mode is lighter for Streamlit Community Cloud. Sentence Transformers mode is useful for local/offline-style embedding workflows, but it can make cloud builds heavier.

## Streamlit Cloud Deployment

1. Create a GitHub repository.
2. Upload this project to GitHub.
3. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
4. Select **New app**.
5. Choose your GitHub repo and branch.
6. Set the main file to `app.py`.
7. Add secrets:

```toml
GEMINI_API_KEY = "your_google_gemini_api_key"
EMBEDDING_PROVIDER = "gemini"
```

8. Click **Deploy**.
9. After deployment, Streamlit provides a public shareable URL.

## Usage Flow

1. Upload PDFs from the sidebar.
2. Choose subject tag or leave auto-detect on.
3. Add optional chapter tag.
4. Click **Index uploaded PDFs**.
5. Ask NEET questions in the Tutor Chat tab.
6. Review source references under the answer.
7. Bookmark useful answers or save full chats.
8. Generate daily quizzes and mock tests.
9. Record mock test scores to view analytics.

## Notes

- Upload only documents you have permission to use.
- Streamlit Cloud file storage may reset on redeploy, so keep your original PDFs available.
- Large PDF sets may take time to embed because each chunk is sent to the embedding provider.
- For production at school scale, use persistent object storage and a hosted vector database.
