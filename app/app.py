import os
import sys
import ast
import re
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import time

USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/847/847969.png"
BOT_AVATAR  = "https://cdn-icons-png.flaticon.com/512/4712/4712100.png"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import init_model, generate_with_context
from src.retriever import retrieve_top_sections, build_faiss_index


# ----------------------------------
# Helper to extract Section numbers
# ----------------------------------
def extract_section_number(text):
    """
    Detects section number from OSH section text.
    Handles patterns like:
    '12. Duties of employer'
    '23. Notice of certain diseases'
    '3. (1) ...'
    """
    # Look for pattern like "23." or "12."
    match = re.search(r"^\s*(\d+)\.", text.strip())
    if match:
        sec_num = match.group(1)
        return f"Section {sec_num}"
    
    return "Section ?"



# ----------------------------------
# Initialize all components (cached)
# ----------------------------------
@st.cache_resource
def init_all():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in .env")

    init_model(token)

    CSV_PATH = "data/processed/osh_sections_with_vectors.csv"
    df = pd.read_csv(CSV_PATH)

    # Parse vector embeddings if stored as string
    df["vector_embedding"] = df["vector_embedding"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Build FAISS index
    vecs = np.array(df["vector_embedding"].tolist()).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    return df, embedder, index


df, embedder, index = init_all()


# ----------------------------------
# Retrieval + Generation with Citations
# ----------------------------------
def answer_with_citations(query, top_k=3):
    # Retrieve sections
    top_sections = retrieve_top_sections(query, embedder, df, index, k=top_k)

    citations = []
    for section_text in top_sections:
        sec = extract_section_number(section_text)
        citations.append(sec)

    # Combine context
    context = "\n\n---\n\n".join(top_sections)

    # Generate answer
    answer = generate_with_context(query, context)

    return answer, top_sections, citations


# ----------------------------------
# PAGE SETTINGS
# ----------------------------------
st.set_page_config(
    page_title="OSH Compliance Chatbot",
    page_icon="🦺",
    layout="centered"
)

col1, col2 = st.columns([5, 1])   # Wider left, narrow right

# ----------------------------------
# TITLE
# ----------------------------------
with col1:
    st.markdown("""
        <h1 style="margin-bottom: 0; font-size:40px">🦺 O.S.C.A.R.</h1>
        <p style="font-size:16px; margin-top:0; margin-bottom:30px;">
            Occupational Safety Compliance & Regulation <br> 
            AI Assistant for the Occupational Safety, Health & Working Conditions Code, 2020
        </p>
    """, unsafe_allow_html=True)

# CLEAR CHAT BUTTON
with col2:
    st.markdown(
        """
        <div style="height: 30px;"></div>  """,
        unsafe_allow_html=True
    )
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------
# CHAT HISTORY
# ----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add the flag here:
if "generating_response" not in st.session_state:
    st.session_state.generating_response = False

# ----------------------------------
# DISPLAY MESSAGES
# ----------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin-bottom:20px;">
                <div style="
                    background-color:#2F2F2F;
                    padding:12px;
                    border-radius:10px;
                    max-width:70%;
                    text-align:right;
                ">
                    {msg['content']}
                </div>
                <img src="{USER_AVATAR}" width="40" style="margin-left:10px; border-radius:50%;">
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div style="display:flex; align-items:flex-start; margin-bottom:20px;">
                <img src="{BOT_AVATAR}" width="40" style="margin-right:15px; margin-top:5px; border-radius:50%; align-self:flex-start;">
                <div style="
                    background-color:#1E1E1E;
                    padding:16px;
                    border-radius:10px;
                    max-width:80%;
                ">
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # Show citations
        if "citations" in msg:
            st.markdown(f"🔍 **Cited Sections:** {', '.join(msg['citations'])}")

        # Expandable context viewer
        if "context" in msg:
            with st.expander("📘 View retrieved context"):
                for i, sec in enumerate(msg["context"], start=1):
                    st.markdown(f"**Context {i}:**\n\n{sec}")


# ----------------------------------
# INPUT HANDLER CALLBACK
# ----------------------------------
def submit():
    st.session_state["pending_user_msg"] = st.session_state["new_input"]
    st.session_state["new_input"] = ""


# ----------------------------------
# INPUT BOX
# ----------------------------------
if not st.session_state.generating_response:
    st.text_input(
        "Ask your OSH Code question:",
        placeholder="Type your question...",
        key="new_input",
        on_change=submit
    )

# ----------------------------------
# PROCESS USER QUERY
# ----------------------------------
if "pending_user_msg" in st.session_state and st.session_state["pending_user_msg"]:
    user_input = st.session_state["pending_user_msg"]

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state["pending_user_msg"] = "" 
    st.session_state.generating_response = True  

    st.rerun()

if st.session_state.generating_response:
    user_input = st.session_state.messages[-1]["content"]

    with st.spinner("Thinking..."):
        answer, context_list, citations = answer_with_citations(user_input)

    placeholder = st.empty()
    streamed_text = ""
    typing_delay = 0.03

    # --- NEW STREAMING LOOP ---
    for char in answer:
        streamed_text += char

        if char.isspace() or char in '.,:;?!':
            
            placeholder.markdown(
                f"""
                <div style="display:flex; align-items:flex-start; margin-bottom:20px;">
                    <img src="{BOT_AVATAR}" width="40" style="margin-right:15px; margin-top:5px; border-radius:50%; align-self:flex-start;">
                    <div style="
                        background-color:#1E1E1E;
                        padding:16px;
                        border-radius:10px;
                        max-width:80%;
                    ">
                        {streamed_text}<span style="opacity:0.8;">▌</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            time.sleep(typing_delay)

    placeholder.empty()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context_list,
        "citations": citations
    })

    st.session_state.generating_response = False

    st.rerun()

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("""
<hr>
<div style="text-align:center; color:#777;">
    Built using RAG (FAISS + MiniLM + Llama-3)
</div>
""", unsafe_allow_html=True)
