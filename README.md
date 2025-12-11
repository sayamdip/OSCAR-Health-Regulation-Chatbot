# 🦺 O.S.C.A.R (Occupational Safety Compliance & Regulation)
### **Retrieval-Augmented Generation (RAG) System for the Occupational Safety, Health & Working Conditions Code, 2020**

This project is an **AI-powered assistive chatbot** designed to interpret and answer questions based on the **Occupational Safety, Health and Working Conditions (OSH) Code, 2020**.  

It uses a **Retrieval-Augmented Generation (RAG)** pipeline combining:

✔ FAISS vector search  
✔ MiniLM sentence embeddings  
✔ Llama-3 language model  
✔ Streamlit front-end  

The system provides *high-accuracy factual answers*, complete with:

- Retrieved context  
- OSH section citations  
- Clean ChatGPT-style UI  
- Evaluation metrics (Confusion Matrix)

---

## 📁 **Project Structure**
```bash
OSHComplianceBot/
│
├── app/
│   └── app.py                    # Streamlit UI (ChatGPT-style chatbot)
│
├── data/
│   ├── raw/
│   │   └── OSH_Code_2020.pdf     # Original OSH Code PDF
│   │
│   ├── processed/
│   │   ├── osh_sections.json     # Extracted sections (cleaned + split)
│   │   ├── osh_sections_with_vectors.csv  # Embeddings for each section
│   └── └── evaluation_questions.csv       # 40 curated Q/A for evaluation
|
├── notebooks/
│   ├── data_preparation.ipynb     # PDF extraction + cleaning + splitting
│   ├── embedding_and_index.ipynb  # Embeddings + FAISS index construction
│   └── chatbot_pipeline.ipynb     # RAG logic + inference + evaluation
│
├── src/
│   ├── retriever.py              # FAISS index builder + top-K retrieval
│   ├── inference.py              # Llama-3 inference wrapper
│   └── evaluation.py (optional)  # Evaluation logic (confusion matrix)
│
├── .gitignore
├── main.py
├──README.md
└── requirements.txt
```

---

## ⚙️ **Tech Stack**

### **Backend (AI + Retrieval)**
| Component | Technology | Purpose |
|----------|------------|---------|
| Text Embeddings | **SentenceTransformers — MiniLM L6 v2** | Convert document sections into 384-dim embeddings |
| Vector Database | **FAISS (CPU)** | Fast similarity search for RAG |
| LLM | **Llama-3-8B-Instruct (via HuggingFace Inference API)** | Generate grounded answers |
| PDF Processing | **PyMuPDF (fitz)** | Extract text from OSH PDF |
| Evaluation | **scikit-learn** | Confusion Matrix & accuracy metrics |


### **Frontend**
| Component | Technology |
|----------|------------|
| **Streamlit** | ChatGPT-style chatbot UI |
| **HTML/CSS within Streamlit** | Message bubble styling |

---

## 🧱 **System Architecture**

### **RAG Pipeline**
```bash
User Query
     ↓
Sentence-Transformer embeddings
     ↓
FAISS similarity search → Top-K OSH Sections
     ↓
Context + Query sent to Llama-3
     ↓
AI generates grounded answer
     ↓
Streamlit UI displays:
  - Final answer
  - Retrieved context
  - Section citations
```



## 🤖 **Chatbot Capabilities**:

✔ Understands natural language <br>
✔ Retrieves the most relevant OSH sections <br>
✔ Provides citations (Section 12, Section 23) <br>
✔ Shows full context used <br>
✔ Never hallucinates (due to strict RAG prompt) <br>
✔ Clean ChatGPT-style UI <br>

---

### 🚀 How to Run Locally

1. Clone the repo
```bash
git clone https://github.com/Sayantan1024/OSHComplianceBot.git
cd OSHComplianceBot
```

2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add HuggingFace Token in .env
```bash
HF_TOKEN=hf_xxxxxxx
```

5. Run Streamlit App
```bash
streamlit run app/app.py
```

---

### 🎯 Future Enhancements

- Section summarization
- Voice input
- Admin dashboard for analytics
- Chat session export (PDF)