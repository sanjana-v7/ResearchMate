# ResearchMate – AI-Powered Document Assistant using RAG

> **Tagline:** *Your Documents. Your Knowledge. One AI Assistant.*

ResearchMate is a production-ready, general-purpose Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **FAISS**, **Sentence Transformers**, and the official **Google GenAI SDK (Gemini)**.

Designed for real-world document workflows, ResearchMate lets you upload multiple diverse documents—company policies, study materials, product manuals, technical guidelines, and reports (PDF, TXT, DOCX)—and query them with context-grounded answers, exact page citations, and verifiable source passages.

---

## 🌟 Key Features

1. **Multi-Format Document Ingestion**
   - Seamlessly extracts text from **PDF** (via PyMuPDF), **TXT** (with multi-encoding fallback), and **DOCX** (paragraphs and tables via python-docx).
   - Preserves document names, 1-indexed page numbers, and generates cryptographic hashes (SHA-256) to detect duplicate uploads.
   - Detects and warns against blank, corrupted, or scanned/image-only PDFs lacking OCR.

2. **Semantic Text Chunking & Provenance**
   - Utilizes `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) with default chunk size of 800 characters and 150 character overlap.
   - Strips empty chunks and attaches complete provenance metadata (`chunk_id`, `filename`, `page`, `chunk_index`, `doc_hash`) to every chunk.

3. **High-Performance FAISS Vector Database**
   - Embeds text chunks into dense 384-dimensional vectors using `all-MiniLM-L6-v2` (`sentence-transformers`).
   - Normalizes vector representations so `faiss.IndexFlatIP` computes exact cosine similarity in milliseconds.
   - Supports incremental multi-document indexing, individual document deletion with automatic index rebuilding, and disk persistence.

4. **Grounded RAG Pipeline & Source Citations**
   - Strictly grounds the Gemini LLM (`gemini-3.8-flash`) via a rigorous system prompt.
   - Every answer includes exact citations: `[Document: <filename>, Page: <page>]`.
   - Never fabricates facts or citations: explicitly reports when information is not present in the uploaded documents.
   - Transparently highlights conflicting information across different documents.
   - Includes an expandable source drawer showing the exact retrieved passages and cosine similarity scores.

5. **Structured Document Summarization**
   - Produces comprehensive, 5-part summaries for single documents or the entire corpus:
     1. Main Topic
     2. Key Points
     3. Important Information & Operational Details
     4. Important Dates, Numbers & Figures
     5. Conclusions & Takeaways

6. **Interactive Knowledge Base Diagnostics**
   - Inspect indexed chunks, verify token boundaries and page numbers, and run test vector queries with live similarity scores.

---

## 🏗️ Architecture & Project Structure

```
ResearchMate/
│
├── app.py                     # Streamlit frontend & interactive UI
├── config.py                  # Central configuration & secrets manager
├── requirements.txt           # Production dependencies
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Excludes caches, uploads, vector stores & secrets
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── document_processor.py  # PDF, TXT, DOCX parser & validation
│   ├── text_splitter.py       # LangChain recursive character chunking
│   ├── embeddings.py          # Sentence-Transformers (all-MiniLM-L6-v2)
│   ├── vector_store.py        # FAISS vector store & document lifecycle
│   ├── retriever.py           # Semantic search & prompt context assembly
│   ├── rag_pipeline.py        # Gemini API integration & grounded RAG
│   └── summarizer.py          # 5-section structured summarizer
│
├── data/
│   └── uploads/               # Stored uploaded files (.gitkeep preserved)
│
├── vector_store/              # Persisted FAISS index & metadata
│
└── tests/
    ├── test_document_processor.py
    ├── test_text_splitter.py
    ├── test_vector_store.py
    └── test_rag_pipeline.py
```

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- A Google Gemini API Key ([Get one for free at Google AI Studio](https://aistudio.google.com/))

### 2. Clone and Setup Environment
```bash
git clone https://github.com/your-username/ResearchMate.git
cd ResearchMate

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Gemini API Key
Create a `.env` file in the root directory:
```bash
GEMINI_API_KEY="your_actual_gemini_api_key_here"
```
*(Alternatively, you can enter your API key directly in the sidebar of the Streamlit application at runtime.)*

### 4. Run ResearchMate
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the comprehensive pytest suite:
```bash
pytest tests/ -v
```

The test suite covers:
- Multi-page PDF extraction and 1-indexed page preservation.
- TXT extraction and Latin-1 encoding fallbacks.
- DOCX paragraph and table parsing.
- Empty file, corrupted file, and unscannable document error handling.
- Semantic text chunking and metadata preservation.
- Vector store indexing, FAISS similarity search, and duplicate detection.
- Document removal and dynamic index rebuilding.
- Prompt construction, conversational context handling, and citation verification.

---

## ☁️ Deployment on Streamlit Community Cloud

ResearchMate is fully prepared for zero-configuration deployment on **Streamlit Community Cloud**:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: ResearchMate RAG Application"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ResearchMate.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Click **New app**.
   - Select your repository (`<your-username>/ResearchMate`) and branch (`main`).
   - Set **Main file path** to `app.py`.
   - Click **Advanced settings...** and add your Gemini API key under **Secrets**:
     ```toml
     GEMINI_API_KEY = "your_actual_gemini_api_key_here"
     ```
   - Click **Deploy!**

---

## ⚠️ Known Limitations & Design Considerations

1. **Scanned Documents & Images**: Scanned PDFs that contain only bitmap images without an embedded text layer require an OCR engine (such as Tesseract or Cloud Vision). ResearchMate detects these and alerts the user rather than failing silently.
2. **Dynamic Word Pagination**: Word (`.docx`) documents use reflowable layouts rather than fixed physical pages. ResearchMate groups DOCX content cleanly with explicit paragraph and table preservation.
3. **Local Embedding Memory**: Sentence Transformers runs locally on CPU (`all-MiniLM-L6-v2` has ~22M parameters and consumes ~120MB RAM), providing fast offline embedding without incurring external API embedding costs.

---

## 📄 License
MIT License. Built for real-world document intelligence.
