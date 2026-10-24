import os
import re
from pathlib import Path

import chromadb
import streamlit as st

HERE = Path(__file__).parent
NAVY = "#0C2C5D"
LOGO = HERE / "logo.png"

st.set_page_config(page_title="RAG Demo", page_icon=str(LOGO), layout="wide")

st.markdown(
    f"""
    <style>
    h1, h2, h3 {{ color: {NAVY}; }}
    section[data-testid="stSidebar"] {{ border-right: 2px solid {NAVY}; }}
    .stButton > button, .stDownloadButton > button {{
        background-color: {NAVY}; color: #FFFFFF; border: none; border-radius: 8px;
    }}
    .stButton > button:hover {{ background-color: #1B4A94; color: #FFFFFF; }}
    .banner {{ background: {NAVY}; color: #FFFFFF; padding: 14px 20px; border-radius: 10px; margin-bottom: 14px; }}
    .banner h2 {{ color: #FFFFFF; margin: 0; }}
    .banner p {{ margin: 2px 0 0 0; color: #C9D6EC; }}
    .chunk {{ background: #EEF2F8; border-left: 4px solid {NAVY}; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Core RAG pieces ----------
def chunk_paragraphs(text: str, min_chars: int = 40) -> list[str]:
    """Paragraph-wise chunking: split on blank lines; tiny paragraphs are merged forward."""
    paras = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    chunks, buf = [], ""
    for p in paras:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        buf = f"{buf} {p}".strip()
        if len(buf) >= min_chars:
            chunks.append(buf)
            buf = ""
    if buf:
        if chunks:
            chunks[-1] += " " + buf
        else:
            chunks.append(buf)
    return chunks


@st.cache_resource
def get_collection():
    client = chromadb.EphemeralClient()  # in-memory; fine for a demo
    return client.get_or_create_collection("rag_demo", metadata={"hnsw:space": "cosine"})


def ingest(files) -> int:
    col, total = get_collection(), 0
    for f in files:
        text = f.getvalue().decode("utf-8", errors="ignore")
        chunks = chunk_paragraphs(text)
        if not chunks:
            continue
        col.upsert(
            ids=[f"{f.name}::{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": f.name, "chunk": i + 1} for i in range(len(chunks))],
        )
        total += len(chunks)
    return total


def clear_kb():
    col = get_collection()
    ids = col.get()["ids"]
    if ids:
        col.delete(ids=ids)


def retrieve(query: str, k: int):
    col = get_collection()
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    return [
        {"text": d, "source": m["source"], "chunk": m["chunk"], "score": 1 - dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def generate(query: str, hits, api_key: str, model: str):
    import anthropic

    context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    system = (
        "Answer the question using ONLY the context provided. If the answer is not in the "
        "context, say you do not know. Be concise and cite sources like [1], [2]."
    )
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model,
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    ) as stream:
        yield from stream.text_stream


def show_sources(hits):
    with st.expander(f"Retrieved context ({len(hits)} chunks)"):
        for i, h in enumerate(hits, 1):
            st.markdown(
                f"<div class='chunk'><b>[{i}] {h['source']} · chunk {h['chunk']} · "
                f"similarity {h['score']:.2f}</b><br>{h['text']}</div>",
                unsafe_allow_html=True,
            )


# ---------- Sidebar ----------
col = get_collection()
with st.sidebar:
    st.image(str(LOGO), width="stretch")
    st.markdown("### Knowledge base")
    files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    if c1.button("Ingest", width="stretch", disabled=not files):
        with st.spinner("Chunking and embedding..."):
            n = ingest(files)
        st.success(f"Added {n} chunks")
    if c2.button("Clear", width="stretch"):
        clear_kb()
        st.session_state.messages = []
        st.rerun()
    st.metric("Chunks stored", col.count())

    st.markdown("### Settings")
    top_k = st.slider("Top-k chunks", 1, 8, 3)
    api_key = st.text_input("Anthropic API key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""),
                            help="Optional. Without a key the app shows only the retrieval step.")
    model = st.text_input("Model", value="claude-sonnet-5-5")

# ---------- Main ----------
st.markdown(
    "<div class='banner'><h2>RAG Demo</h2>"
    "<p>Upload a text file, ask a question, and watch retrieval + generation work together.</p></div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["hits"])

if col.count() == 0:
    st.info("Start by uploading a .txt file in the sidebar and clicking **Ingest**. A sample.txt is included.")

if query := st.chat_input("Ask a question about your documents...", disabled=col.count() == 0):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            hits = retrieve(query, top_k)
        if api_key:
            try:
                answer = st.write_stream(generate(query, hits, api_key, model))
            except Exception as e:  # keep the demo alive on API errors
                answer = f"Generation failed: {e}"
                st.error(answer)
        else:
            answer = "No API key set, so here is the **retrieval step only**. These are the closest chunks to your question."
            st.markdown(answer)
        show_sources(hits)
    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
