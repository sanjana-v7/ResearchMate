"""
ResearchMate - Streamlit Application
Your Documents. Your Knowledge. One AI Assistant.
A production-grade, multi-document RAG assistant with grounded citations.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    UPLOADS_DIR,
    INDEX_PATH,
    METADATA_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_GEMINI_MODEL,
    get_gemini_api_key
)
from src.document_processor import (
    DocumentProcessor,
    EmptyFileError,
    CorruptedFileError,
    NoTextExtractedError,
    UnsupportedFileTypeError
)
from src.text_splitter import DocumentTextSplitter
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever
from src.rag_pipeline import RAGPipeline
from src.summarizer import DocumentSummarizer

# Streamlit Page Configuration
st.set_page_config(
    page_title="ResearchMate - AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean, Modern Look
st.markdown("""
<style>
    /* Global Typography & Spacing */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .tagline {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .citation-box {
        background-color: #F1F5F9;
        border-left: 4px solid #2563EB;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-top: 6px;
        margin-bottom: 10px;
        font-size: 0.92rem;
    }
    .citation-header {
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #DBEAFE;
        color: #1D4ED8;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "vector_store" not in st.session_state:
        vs = VectorStore(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
        vs.load()
        st.session_state.vector_store = vs

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "api_key" not in st.session_state:
        st.session_state.api_key = get_gemini_api_key()

    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = [
            "What are the main topics discussed across these documents?",
            "Can you highlight key policies or guidelines mentioned?",
            "What are the important dates, deadlines, or numerical figures?"
        ]


init_session_state()

# Helper instances
vector_store: VectorStore = st.session_state.vector_store
retriever = DocumentRetriever(vector_store=vector_store, default_top_k=DEFAULT_TOP_K)
rag_pipeline = RAGPipeline(retriever=retriever, api_key=st.session_state.api_key, model_name=DEFAULT_GEMINI_MODEL)
summarizer = DocumentSummarizer(vector_store=vector_store, api_key=st.session_state.api_key, model_name=DEFAULT_GEMINI_MODEL)


# ==========================================
# SIDEBAR: DOCUMENT & SETTINGS MANAGEMENT
# ==========================================
with st.sidebar:
    st.markdown("## 📚 ResearchMate")
    st.markdown("*Your Documents. Your Knowledge. One AI Assistant.*")
    st.divider()

    # Gemini API Key Section
    st.markdown("### 🔑 API Configuration")
    env_key_available = bool(get_gemini_api_key())

    if env_key_available:
        st.success("✅ Gemini API Key detected", icon="🔒")
        with st.expander("Change API Key"):
            custom_key = st.text_input("Override API Key", value=st.session_state.api_key, type="password")
            if custom_key != st.session_state.api_key:
                st.session_state.api_key = custom_key
                rag_pipeline.update_api_key(custom_key)
                summarizer.update_api_key(custom_key)
                st.rerun()
    else:
        entered_key = st.text_input("Enter Gemini API Key", value=st.session_state.api_key, type="password", placeholder="AIzaSy...")
        if entered_key:
            st.session_state.api_key = entered_key
            rag_pipeline.update_api_key(entered_key)
            summarizer.update_api_key(entered_key)
        else:
            st.warning("Please provide a Gemini API Key to enable AI chat and summarization.")

    st.divider()

    # Document Upload Section
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload company policies, study guides, reports, or manuals."
    )

    if uploaded_files:
        if st.button("Index Uploaded Documents", type="primary", use_container_width=True):
            text_splitter = DocumentTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP
            )

            success_count = 0
            with st.spinner("Processing and indexing documents..."):
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    file_bytes = uploaded_file.getvalue()

                    try:
                        # 1. Extract raw pages and check for text
                        pages = DocumentProcessor.process_file(file_bytes, filename)

                        # Check for duplicate
                        is_dup, reason = vector_store.is_duplicate(filename, pages[0].doc_hash)
                        if is_dup:
                            st.warning(f"⚠️ Skipping '{filename}': {reason}")
                            continue

                        # 2. Split into semantic chunks
                        chunks = text_splitter.split_pages(pages)
                        if not chunks:
                            st.warning(f"⚠️ Document '{filename}' did not yield any text chunks.")
                            continue

                        # 3. Add to FAISS index
                        added_count = vector_store.add_chunks(chunks)

                        # 4. Save uploaded file to disk for persistence
                        save_path = UPLOADS_DIR / filename
                        with open(save_path, "wb") as f:
                            f.write(file_bytes)

                        success_count += 1
                        st.toast(f"Indexed '{filename}' ({len(pages)} pages, {added_count} chunks)", icon="✅")

                    except EmptyFileError:
                        st.error(f"❌ '{filename}' is completely empty (0 bytes).")
                    except NoTextExtractedError:
                        st.warning(f"⚠️ '{filename}' has no extractable text (e.g. scanned/image PDF without OCR).")
                    except CorruptedFileError as e:
                        st.error(f"❌ '{filename}' could not be read: {str(e)}")
                    except UnsupportedFileTypeError as e:
                        st.error(f"❌ '{filename}': {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error processing '{filename}': {str(e)}")

            if success_count > 0:
                vector_store.save()
                st.success(f"Successfully indexed {success_count} new document(s)!")
                st.rerun()

    st.divider()

    # Uploaded Documents Management List
    st.markdown("### 📑 Indexed Documents")
    stats = vector_store.get_document_stats()
    indexed_docs = stats.get("documents", [])

    if not indexed_docs:
        st.info("No documents indexed yet.")
    else:
        for doc in indexed_docs:
            col_info, col_del = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{doc['filename']}**  \n<small>{doc['pages']} page(s) • {doc['chunks']} chunks</small>", unsafe_allow_html=True)
            with col_del:
                if st.button("🗑️", key=f"del_{doc['filename']}", help=f"Remove {doc['filename']}"):
                    # Remove file from disk
                    disk_file = UPLOADS_DIR / doc['filename']
                    if disk_file.exists():
                        try:
                            disk_file.unlink()
                        except Exception:
                            pass
                    # Remove from vector store and rebuild index
                    vector_store.remove_document(doc['filename'])
                    vector_store.save()
                    st.toast(f"Removed '{doc['filename']}'", icon="🗑️")
                    st.rerun()

    st.divider()

    # Bulk Actions
    col_clear_docs, col_clear_chat = st.columns(2)
    with col_clear_docs:
        if st.button("Clear All Docs", use_container_width=True):
            vector_store.clear()
            vector_store.save()
            for f in UPLOADS_DIR.glob("*"):
                if f.is_file() and f.name != ".gitkeep":
                    try:
                        f.unlink()
                    except Exception:
                        pass
            st.toast("Cleared all documents from knowledge base.", icon="🧹")
            st.rerun()

    with col_clear_chat:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.toast("Chat history cleared.", icon="🧹")
            st.rerun()


# ==========================================
# MAIN PAGE: DASHBOARD & WORKSPACES
# ==========================================

st.markdown('<div class="main-title">📚 ResearchMate</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Your Documents. Your Knowledge. One AI Assistant.</div>', unsafe_allow_html=True)

# Overview Dashboard Metrics
stats = vector_store.get_document_stats()
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{stats['total_documents']}</div>
        <div class="metric-lbl">Documents Indexed</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{stats['total_pages']}</div>
        <div class="metric-lbl">Total Pages</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{stats['total_chunks']}</div>
        <div class="metric-lbl">Semantic Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    status_label = "Ready" if stats['total_chunks'] > 0 else "Empty"
    status_color = "#16A34A" if stats['total_chunks'] > 0 else "#64748B"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color: {status_color};">{status_label}</div>
        <div class="metric-lbl">Vector Index Status</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Main Tabs
tab_chat, tab_summarize, tab_kb = st.tabs([
    "💬 AI Chat Assistant",
    "📑 Document Summarizer",
    "🔍 Knowledge Base & Inspector"
])


# ------------------------------------------
# TAB 1: AI CHAT ASSISTANT
# ------------------------------------------
with tab_chat:
    if stats["total_documents"] == 0:
        st.info("👋 Welcome to **ResearchMate**! To get started, upload your documents (PDF, TXT, or DOCX) in the sidebar. You will then be able to ask questions across your documents and receive grounded answers with citations.")
    else:
        # Suggested questions
        st.markdown("**💡 Suggested Questions:**")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, sq in enumerate(st.session_state.suggested_questions):
            with cols[i]:
                if st.button(sq, key=f"sq_{i}", use_container_width=True):
                    # Set prompt to trigger chat input processing
                    st.session_state.active_prompt = sq
                    st.rerun()

    # Display Conversation History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If message contains citations, display expandable reference drawer
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"🔎 Source References & Retrieved Passages ({len(sources)} cited)"):
                    for j, src in enumerate(sources, 1):
                        fname = src.get("filename", "Unknown")
                        page = src.get("page", 1)
                        score = src.get("score", 0.0)
                        text = src.get("text", "")

                        st.markdown(f"""
                        <div class="citation-box">
                            <div class="citation-header">
                                📄 Source {j}: <b>{fname}</b> — Page {page}
                                <span class="badge">Score: {score:.3f}</span>
                            </div>
                            <i>"{text}"</i>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Handler
    prompt_to_handle = None
    if "active_prompt" in st.session_state:
        prompt_to_handle = st.session_state.pop("active_prompt")
    else:
        chat_user_input = st.chat_input("Ask a question about your uploaded documents...")
        if chat_user_input:
            prompt_to_handle = chat_user_input

    if prompt_to_handle:
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_handle})
        with st.chat_message("user"):
            st.markdown(prompt_to_handle)

        # Generate Grounded RAG Response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating grounded response..."):
                response = rag_pipeline.answer_question(
                    question=prompt_to_handle,
                    chat_history=st.session_state.chat_history,
                    top_k=DEFAULT_TOP_K
                )

                st.markdown(response.answer)

                # Show Sources if available
                if response.sources:
                    with st.expander(f"🔎 Source References & Retrieved Passages ({len(response.sources)} cited)"):
                        for j, src in enumerate(response.sources, 1):
                            fname = src.get("filename", "Unknown")
                            page = src.get("page", 1)
                            score = src.get("score", 0.0)
                            text = src.get("text", "")

                            st.markdown(f"""
                            <div class="citation-box">
                                <div class="citation-header">
                                    📄 Source {j}: <b>{fname}</b> — Page {page}
                                    <span class="badge">Score: {score:.3f}</span>
                                </div>
                                <i>"{text}"</i>
                            </div>
                            """, unsafe_allow_html=True)

                # Append Assistant Message to History
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources
                })


# ------------------------------------------
# TAB 2: DOCUMENT SUMMARIZER
# ------------------------------------------
with tab_summarize:
    st.markdown("### 📑 Structured Document Summarizer")
    st.markdown("Generate comprehensive, strictly grounded 5-section summaries: Main Topic, Key Points, Important Information, Dates & Figures, and Conclusions.")

    indexed_file_list = [d["filename"] for d in stats.get("documents", [])]

    if not indexed_file_list:
        st.info("Upload and index documents in the sidebar to enable summarization.")
    else:
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            summary_target = st.selectbox(
                "Choose document to summarize:",
                options=["All Uploaded Documents"] + indexed_file_list
            )
        with col_btn:
            st.write("")
            st.write("")
            summarize_clicked = st.button("Generate Summary", type="primary", use_container_width=True)

        if summarize_clicked:
            target_arg = None if summary_target == "All Uploaded Documents" else summary_target
            with st.spinner(f"Generating structured summary for '{summary_target}'..."):
                res = summarizer.summarize_document(target_arg)

                if res.success:
                    st.markdown("---")
                    st.markdown(res.content)
                    st.download_button(
                        label="📥 Download Summary as Markdown",
                        data=res.content,
                        file_name=f"{summary_target.replace(' ', '_')}_Summary.md",
                        mime="text/markdown"
                    )
                else:
                    st.error(f"❌ Summarization failed: {res.error_message}")


# ------------------------------------------
# TAB 3: KNOWLEDGE BASE INSPECTOR
# ------------------------------------------
with tab_kb:
    st.markdown("### 🔍 Knowledge Base & Index Diagnostics")
    st.markdown("Inspect extracted chunks, verify metadata fidelity, and test semantic retrieval.")

    if stats["total_documents"] == 0:
        st.info("Knowledge base is currently empty. Upload documents to populate the index.")
    else:
        tab_sub_docs, tab_sub_search = st.tabs(["Document & Chunk Browser", "Semantic Search Tester"])

        with tab_sub_docs:
            selected_inspect_doc = st.selectbox(
                "Select document to inspect chunks:",
                options=indexed_file_list
            )

            if selected_inspect_doc:
                doc_chunks = [
                    m for m in vector_store.chunks_metadata
                    if m.get("filename") == selected_inspect_doc
                ]

                st.write(f"Total chunks for **{selected_inspect_doc}**: `{len(doc_chunks)}`")
                for c in doc_chunks[:20]:  # Show first 20 chunks
                    with st.expander(f"Chunk ID: {c.get('chunk_id')} | Page: {c.get('page')}"):
                        st.code(c.get("text"), language="text")

        with tab_sub_search:
            st.markdown("Test raw vector retrieval independently of LLM generation.")
            test_query = st.text_input("Enter test query:", placeholder="e.g., reimbursement policy, warranty period")
            test_top_k = st.slider("Top K", min_value=1, max_value=10, value=3)

            if test_query:
                raw_results = retriever.retrieve(test_query, top_k=test_top_k)
                st.write(f"Found **{len(raw_results)}** matching passages:")
                for r in raw_results:
                    st.markdown(f"**Document:** `{r.get('filename')}` | **Page:** `{r.get('page')}` | **Cosine Similarity:** `{r.get('score'):.4f}`")
                    st.info(r.get("text"))
