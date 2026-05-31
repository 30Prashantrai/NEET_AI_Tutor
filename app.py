from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

from chatbot.gemini_client import generate_answer, transcribe_groq_audio
from chatbot.rag import answer_question
from utils.pdf_loader import pdf_to_chunks, save_uploaded_pdf
from utils.session_store import (
    add_bookmark,
    add_favorite_chat,
    list_bookmarks,
    list_favorites,
    performance_history,
    record_quiz_result,
)
from utils.vector_store import add_chunks, collection_stats


load_dotenv()


VOICE_LANGUAGES = {
    "Auto detect": "",
    "English": "en",
    "Hindi": "hi",
    "Nepali": "ne",
}

st.set_page_config(
    page_title="NEET AI Tutor",
    page_icon="NEET",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css(theme_mode: str) -> None:
    dark = theme_mode == "Dark"
    colors = {
        "app_bg": "#080b12" if dark else "#f6f8fc",
        "surface": "#101624" if dark else "#ffffff",
        "surface_2": "#0b1020" if dark else "#eef4fb",
        "text": "#edf6ff" if dark else "#0b1220",
        "muted": "#a8b3c7" if dark else "#526070",
        "border": "rgba(148, 163, 184, 0.18)" if dark else "rgba(15, 23, 42, 0.12)",
        "accent": "#38bdf8" if dark else "#0369a1",
        "accent_2": "#22c55e" if dark else "#047857",
        "glow": "rgba(56, 189, 248, 0.18)" if dark else "rgba(14, 165, 233, 0.16)",
        "source_bg": "rgba(56, 189, 248, 0.07)" if dark else "rgba(14, 165, 233, 0.08)",
    }
    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {"dark" if dark else "light"};
        }}
        .stApp {{
            background:
                radial-gradient(circle at top left, {colors["glow"]}, transparent 34rem),
                {colors["app_bg"]};
            color: {colors["text"]};
        }}
        .block-container {{
            max-width: 1120px;
            padding-top: 1rem;
            padding-bottom: 5rem;
        }}
        [data-testid="stSidebar"] {{
            background: {colors["surface_2"]};
            border-right: 1px solid {colors["border"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {colors["text"]};
        }}
        .neet-header {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            padding: 1rem 0 0.75rem;
        }}
        .brand-lockup {{
            display: flex;
            align-items: center;
            gap: 0.9rem;
            min-width: 0;
        }}
        .brand-mark {{
            width: 58px;
            height: 58px;
            flex: 0 0 58px;
            border-radius: 16px;
            display: grid;
            place-items: center;
            font-weight: 900;
            color: #ffffff;
            background:
                linear-gradient(135deg, {colors["accent"]}, {colors["accent_2"]});
            box-shadow: 0 14px 34px {colors["glow"]};
            border: 1px solid rgba(255, 255, 255, 0.24);
        }}
        .brand-eyebrow {{
            color: {colors["accent"]};
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0 0 0.12rem;
        }}
        .neet-title {{
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: 0;
            color: {colors["text"]};
        }}
        .neet-subtitle {{
            margin: 0.15rem 0 0;
            color: {colors["muted"]};
            font-size: 0.98rem;
        }}
        .theme-pill {{
            border: 1px solid {colors["border"]};
            background: {colors["surface"]};
            color: {colors["muted"]};
            border-radius: 999px;
            padding: 0.45rem 0.7rem;
            font-size: 0.82rem;
            white-space: nowrap;
        }}
        .metric-row {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem;
        }}
        .metric-tile {{
            border: 1px solid {colors["border"]};
            background: {colors["surface"]};
            border-radius: 8px;
            padding: 0.75rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, {"0.18" if dark else "0.06"});
        }}
        .metric-tile strong {{
            display: block;
            font-size: 1.25rem;
            color: {colors["text"]};
        }}
        .metric-tile span {{
            color: {colors["muted"]};
            font-size: 0.82rem;
        }}
        .source-box {{
            border: 1px solid {colors["border"]};
            background: {colors["source_bg"]};
            border-radius: 8px;
            padding: 0.7rem;
            margin: 0.45rem 0;
            font-size: 0.9rem;
            color: {colors["text"]};
        }}
        .source-box b {{
            color: {colors["accent"]};
        }}
        .small-note {{
            color: {colors["muted"]};
            font-size: 0.88rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: 8px;
            border: 1px solid {colors["border"]};
        }}
        [data-testid="stChatMessage"] {{
            border-radius: 8px;
            border: 1px solid {colors["border"]};
            background: {colors["surface"]};
        }}
        div[data-baseweb="select"] > div,
        textarea,
        input {{
            border-color: {colors["border"]} !important;
        }}
        @media (max-width: 760px) {{
            .block-container {{
                padding-left: 0.9rem;
                padding-right: 0.9rem;
                padding-top: 0.5rem;
            }}
            .neet-header {{
                align-items: flex-start;
                flex-direction: column;
            }}
            .brand-mark {{
                width: 50px;
                height: 50px;
                flex-basis: 50px;
                border-radius: 14px;
            }}
            .neet-title {{
                font-size: 1.45rem;
            }}
            .metric-row {{
                grid-template-columns: 1fr;
            }}
            [data-testid="stChatInput"] {{
                left: 0;
                right: 0;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("theme_mode", "Dark")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_answer", "")
    st.session_state.setdefault("last_question", "")
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("quiz", "")
    st.session_state.setdefault("daily_quiz", "")
    st.session_state.setdefault("voice_question", "")
    st.session_state.setdefault("voice_audio", None)


def api_key_ready() -> bool:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or get_streamlit_secret("GEMINI_API_KEY")
        return bool(key and key.strip() not in {"your_google_gemini_api_key", "your_google_gemini_api_key_here"})
    if provider == "xai":
        key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or get_streamlit_secret("XAI_API_KEY")
        return bool(key and key.strip() not in {"your_xai_grok_api_key", "your_xai_grok_api_key_here"})
    key = os.getenv("GROQ_API_KEY") or get_streamlit_secret("GROQ_API_KEY")
    return bool(key and key.strip() not in {"your_groq_api_key", "your_groq_api_key_here"})


def sync_secret_to_env() -> None:
    for name in (
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "GROQ_API_KEY",
        "LLM_PROVIDER",
        "XAI_MODEL",
        "GROQ_MODEL",
        "GROQ_STT_MODEL",
        "EMBEDDING_PROVIDER",
    ):
        secret_value = get_streamlit_secret(name)
        if secret_value and not os.getenv(name):
            os.environ[name] = secret_value


def get_streamlit_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except st.errors.StreamlitSecretNotFoundError:
        return ""


def render_header(stats: dict) -> None:
    st.markdown(
        f"""
        <div class="neet-header">
            <div class="brand-lockup">
                <div class="brand-mark">PR</div>
                <div>
                    <p class="brand-eyebrow">Prashant Rai</p>
                    <h1 class="neet-title">NEET AI Tutor</h1>
                    <p class="neet-subtitle">ChatGPT-style mentor for NEET PYQs, NCERT concepts, formulas, and shortcuts.</p>
                </div>
            </div>
            <div class="theme-pill">{st.session_state.theme_mode} mode</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-tile"><strong>{stats["chunks"]}</strong><span>Indexed text chunks</span></div>
            <div class="metric-tile"><strong>{len(stats["subjects"])}</strong><span>Subjects detected</span></div>
            <div class="metric-tile"><strong>{len(stats["chapters"])}</strong><span>Chapters detected</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(stats: dict) -> tuple[str, str, str, int]:
    with st.sidebar:
        st.markdown("### Prashant Rai NEET AI Tutor")
        st.segmented_control(
            "Theme",
            ["Dark", "Light"],
            key="theme_mode",
        )
        st.divider()
        st.subheader("Study Filters")
        subject = st.selectbox("Subject", ["All", "Physics", "Chemistry", "Biology"])
        chapter_options = ["All"] + stats["chapters"]
        chapter = st.selectbox("Chapter", chapter_options)
        language = st.selectbox("Explanation language", ["English", "Hindi", "Nepali"])
        retrieval_k = st.slider("Source matches", 3, 10, 6)

        st.divider()
        st.subheader("Upload NEET PDFs")
        uploads = st.file_uploader(
            "Questions, answers, explanations, NCERT notes",
            type=["pdf"],
            accept_multiple_files=True,
        )
        upload_subject = st.selectbox("PDF subject tag", ["Auto detect", "Physics", "Chemistry", "Biology"])
        upload_chapter = st.text_input("Chapter tag", placeholder="Optional, e.g. Human Physiology")

        if st.button("Index uploaded PDFs", use_container_width=True):
            if not uploads:
                st.warning("Upload at least one PDF first.")
            else:
                total = 0
                with st.spinner("Extracting text and creating embeddings..."):
                    for uploaded in uploads:
                        pdf_path = save_uploaded_pdf(uploaded)
                        chunks = pdf_to_chunks(pdf_path, subject=upload_subject, chapter=upload_chapter)
                        total += add_chunks(chunks)
                st.success(f"Indexed {total} chunks from {len(uploads)} PDF(s).")
                st.rerun()

        st.divider()
        st.subheader("Saved")
        if st.button("Save this chat", use_container_width=True):
            add_favorite_chat(st.session_state.messages)
            st.success("Chat saved.")
        if st.button("Bookmark last answer", use_container_width=True):
            if st.session_state.last_question and st.session_state.last_answer:
                add_bookmark(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    {"sources": [source.metadata for source in st.session_state.last_sources]},
                )
                st.success("Bookmarked.")
            else:
                st.info("Ask a question first.")

        st.caption("Voice recording uses Groq Whisper. On phones, allow microphone access in the browser.")
        return subject, chapter, language, retrieval_k


def render_sources() -> None:
    if not st.session_state.last_sources:
        return
    with st.expander("Source references from uploaded material", expanded=True):
        for index, source in enumerate(st.session_state.last_sources, start=1):
            meta = source.metadata
            st.markdown(
                f"""
                <div class="source-box">
                    <b>Source {index}</b> - {meta.get("source", "Uploaded material")} - page {meta.get("page", "?")}<br>
                    {meta.get("subject", "General")} - {meta.get("chapter", "Unclassified")}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(source.text[:450] + ("..." if len(source.text) > 450 else ""))


def submit_question(prompt: str, subject: str, chapter: str, language: str, retrieval_k: int) -> None:
    prompt = prompt.strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key_ready():
        answer = "Add `GROQ_API_KEY` in `.streamlit/secrets.toml` or Streamlit Cloud secrets to enable free-tier GroqCloud answers."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.warning(answer)
        return

    with st.chat_message("assistant"):
        with st.spinner("Searching NEET material and preparing a teacher-style answer..."):
            try:
                result = answer_question(
                    prompt,
                    subject=subject,
                    chapter=chapter,
                    language=language,
                    chat_history=st.session_state.messages,
                    k=retrieval_k,
                )
                st.markdown(result.answer)
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
                st.session_state.last_question = prompt
                st.session_state.last_answer = result.answer
                st.session_state.last_sources = result.sources
            except Exception as exc:
                error = f"I could not answer yet: {exc}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})


def render_voice_input(subject: str, chapter: str, language: str, retrieval_k: int) -> None:
    with st.expander("Voice question", expanded=False):
        st.caption(
            "Record your question, transcribe it with Groq Whisper, then send the transcript."
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            voice_language_label = st.selectbox("Speech language", list(VOICE_LANGUAGES.keys()))
        with col_b:
            st.caption("Allow microphone access. HTTPS deployment works best on phones.")

        audio = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=True,
            use_container_width=True,
            format="webm",
            key="voice_recorder",
        )
        if audio and audio.get("bytes"):
            st.session_state.voice_audio = audio
            st.audio(audio["bytes"], format="audio/webm")

        if st.button("Transcribe recording", use_container_width=True):
            audio_bytes = (st.session_state.voice_audio or {}).get("bytes")
            if not audio_bytes:
                st.warning("No recording found. Click Start recording, speak, then Stop recording.")
            elif not api_key_ready():
                st.warning("Add `GROQ_API_KEY` first. Voice transcription uses Groq Whisper.")
            else:
                with st.spinner("Transcribing your question..."):
                    try:
                        st.session_state.voice_question = transcribe_groq_audio(
                            audio_bytes,
                            language=VOICE_LANGUAGES[voice_language_label],
                        )
                        if st.session_state.voice_question:
                            st.success("Transcript ready. Review it below, then ask.")
                        else:
                            st.warning("I could not hear a transcript. Try recording again closer to the microphone.")
                    except Exception as exc:
                        st.error(f"Could not transcribe audio: {exc}")

        st.text_area(
            "Voice transcript",
            key="voice_question",
            placeholder="Your spoken question will appear here. You can edit it before sending.",
            height=90,
        )
        if st.button("Ask voice question", use_container_width=True):
            if st.session_state.voice_question.strip():
                submit_question(st.session_state.voice_question, subject, chapter, language, retrieval_k)
            else:
                st.warning("No transcript yet. Record audio and click Transcribe recording first.")


def render_chat(subject: str, chapter: str, language: str, retrieval_k: int) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    render_voice_input(subject, chapter, language, retrieval_k)

    prompt = st.chat_input("Ask a NEET question, e.g. Explain projectile motion PYQ shortcut")
    if not prompt:
        return

    submit_question(prompt, subject, chapter, language, retrieval_k)


def render_mock_test(subject: str, chapter: str, language: str) -> None:
    st.subheader("Daily NEET Quiz")
    if st.button("Generate today's 5-question quiz", use_container_width=True):
        if not api_key_ready():
            st.warning("Add `GROQ_API_KEY` to generate the daily quiz.")
        else:
            prompt = f"""
Create a daily NEET quiz with 5 MCQs for quick revision.
Subject: {subject}. Chapter: {chapter}. Language: {language}.
Use NCERT-first explanations. Include answer key, why wrong options are wrong, and one revision tip per question.
""".strip()
            with st.spinner("Preparing today's quiz..."):
                st.session_state.daily_quiz = generate_answer(prompt)

    if st.session_state.daily_quiz:
        st.markdown(st.session_state.daily_quiz)

    st.divider()
    st.subheader("Mock Test Generator")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        count = st.number_input("Questions", min_value=3, max_value=30, value=10)
    with col_b:
        difficulty = st.selectbox("Difficulty", ["NEET standard", "Easy", "Moderate", "Hard"])
    with col_c:
        minutes = st.number_input("Minutes", min_value=5, max_value=180, value=20)

    if st.button("Generate mock test", use_container_width=True):
        if not api_key_ready():
            st.warning("Add `GROQ_API_KEY` to generate tests.")
            return
        prompt = f"""
Generate a NEET mock test with {count} MCQs.
Subject: {subject}. Chapter: {chapter}. Difficulty: {difficulty}. Language: {language}.
Include four options, answer key, and brief explanations. Keep it NCERT aligned.
Suggest a {minutes}-minute strategy at the end.
""".strip()
        with st.spinner("Creating mock test..."):
            st.session_state.quiz = generate_answer(prompt)

    if st.session_state.quiz:
        st.markdown(st.session_state.quiz)
        score = st.number_input("Record score", min_value=0, max_value=int(count), value=0)
        if st.button("Save score"):
            record_quiz_result(int(score), int(count))
            st.success("Score saved.")


def render_saved_and_analytics() -> None:
    st.subheader("Bookmarks")
    bookmarks = list_bookmarks()
    if not bookmarks:
        st.info("No bookmarks yet.")
    for item in reversed(bookmarks[-10:]):
        with st.expander(item["question"][:90]):
            st.markdown(item["answer"])

    st.subheader("Favorite Chats")
    favorites = list_favorites()
    st.caption(f"{len(favorites)} saved chat(s)")

    st.subheader("Performance Analytics")
    history = performance_history()
    if not history:
        st.info("Generate a mock test and save a score to see analytics.")
        return
    df = pd.DataFrame(history)
    fig = px.line(df, x="created_at", y="percent", markers=True, labels={"percent": "Score %", "created_at": "Date"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Leaderboard")
    leaderboard = df.sort_values("percent", ascending=False).head(10)[["created_at", "score", "total", "percent"]]
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)


def main() -> None:
    sync_secret_to_env()
    init_state()
    inject_css(st.session_state.theme_mode)

    stats = collection_stats()
    subject, chapter, language, retrieval_k = render_sidebar(stats)
    render_header(stats)

    if stats["chunks"] == 0:
        st.info("Upload and index your 16 years of NEET PYQ PDFs from the sidebar to activate source-grounded answers.")

    chat_tab, mock_tab, saved_tab = st.tabs(["Tutor Chat", "Mock Test", "Saved & Analytics"])
    with chat_tab:
        render_chat(subject, chapter, language, retrieval_k)
        render_sources()
    with mock_tab:
        render_mock_test(subject, chapter, language)
    with saved_tab:
        render_saved_and_analytics()


if __name__ == "__main__":
    main()
