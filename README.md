# RAG-Powered Internal Knowledge Assistant using a Small Language Model

An AI assistant that answers employee questions about HR policies, IT support, expenses, and onboarding by retrieving relevant passages from company documents and generating grounded answers with a small language model (Microsoft Phi-3-mini). It runs without paid APIs, so internal documents never leave your own infrastructure.

## Problem Statement

In most organizations, employees repeatedly ask the same routine questions about leave, work-from-home rules, reimbursements, password resets, VPN setup, and onboarding. These are answered manually by HR/IT staff or by searching through lengthy documents, which wastes time, delays responses, and leads to inconsistent information.

This project automates those answers using Retrieval-Augmented Generation (RAG), so each response is based on the actual company documents rather than the model's general knowledge.

## Features

- Natural-language Q&A over HR, IT, and onboarding documents
- Retrieval with sentence embeddings and a ChromaDB vector store
- Answer generation with Phi-3-mini-4k-instruct (small, locally runnable model)
- Relevance-score threshold that refuses out-of-scope questions instead of guessing
- Source document tags shown under every answer
- Clean Streamlit chat interface

## Architecture

![Architecture Diagram](architecture_diagram.jpg)

**Ingestion (run once):** Documents → Loader → Chunking → MiniLM embeddings → ChromaDB

**Query (every question):** Question → MiniLM embedding → ChromaDB similarity search → Top-K chunks → Prompt (question + context) → Phi-3-mini → Grounded answer + sources

## Tech Stack

| Component | Tool |
|---|---|
| Language model | microsoft/Phi-3-mini-4k-instruct (Hugging Face Transformers) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| Orchestration | LangChain (document loaders, text splitter, vector store) |
| Interface | Streamlit |
| Runtime | Google Colab (free T4 GPU) |

## Project Structure

```
internal-knowledge-assistant/
├── app.py                    # Streamlit chat interface
├── rag_pipeline.ipynb        # Notebook: loading, chunking, embedding, testing
├── documents/                # Knowledge base (HR, IT, onboarding documents)
├── architecture_diagram.jpg  # System architecture
├── requirements.txt
└── README.md
```

## Knowledge Base

The knowledge base combines two kinds of documents:

1. **Self-written sample documents:** leave policy, work-from-home policy, expense reimbursement policy, IT support FAQ, and new employee onboarding checklist.
2. **Real public documents** from the OpenGov Foundation HR manual and HR resources repositories (CC0 public domain): HR policy manual, employee review guide, timesheets guide, status messages guide, and HR tools.

All documents are split into chunks of 400 characters with a 50-character overlap.

## Setup and Installation

### 1. Open Google Colab and enable a GPU
`Runtime` → `Change runtime type` → `T4 GPU`

### 2. Install dependencies
```bash
pip install transformers accelerate sentence-transformers chromadb \
    langchain langchain-community langchain-text-splitters streamlit
```

### 3. Mount Google Drive and add documents
```python
from google.colab import drive
drive.mount('/content/drive')
```
Place your documents in `/content/drive/MyDrive/rag_project/documents/`.

### 4. Build the vector database
Run the chunking and embedding cells in `rag_pipeline.ipynb`. The database is saved to `/content/drive/MyDrive/rag_project/chroma_db`, so it only needs to be built once.

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```
To open the app from Colab, expose port 8501 with a tunnel such as Cloudflare:
```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
./cloudflared-linux-amd64 tunnel --url http://localhost:8501
```

## Usage

Type a question in the chat box, for example:

- "How many leave days do I get?"
- "How do I reset my password?"
- "What should I do in my first week as a new employee?"
- "How do I get access to a shared drive?"

Each answer is shown with the source document(s) it was based on.

## Evaluation

The system was tested on a range of in-scope and out-of-scope questions.

| Question | Result |
|---|---|
| How many sick leave days do I get? | Correct (10 days), source: leave_policy.txt |
| How do I reset my password? | Correct, step-by-step, source: it_faq.txt |
| First week as a new employee? | Correct, source: onboarding_checklist.txt |
| What is the capital of France? | Correctly refused (out of scope) |

**Relevance threshold calibration:** relevant queries scored between 0.13 and 0.55, while irrelevant queries scored around -0.22 to -0.25. A threshold of 0.15 separates the two and prevents the model from answering unrelated questions.

## Limitations

- Phi-3-mini sometimes adds numbers from different categories (for example, summing annual and sick leave) even when instructed not to.
- On a question about alcohol as a business expense, the answer varied with prompt wording and retrieved context.
- Questions that are only partly covered (for example, part-time employee leave) may be answered with general policy text instead of a refusal.
- Mixing documents from different sources can produce overlapping or conflicting facts on the same topic.
- This is a single-user demo, not built for concurrent production use.

## Future Work

- Post-processing checks for numeric answers
- Hybrid retrieval (keyword + semantic) and re-ranking
- Support for PDF and DOCX uploads from the interface
- Evaluation with a labeled question set and automatic metrics
- Testing a larger or fine-tuned model

## Acknowledgements

- [OpenGov Foundation HR Manual](https://github.com/opengovfoundation/hr-manual) and [HR Resources](https://github.com/opengovfoundation/hr-resources) (CC0)
- Microsoft Phi-3, Hugging Face, LangChain, ChromaDB, Streamlit

## Author

**Shanmugam S**

## License

This project is released under the MIT License. The OpenGov Foundation documents remain under their original CC0 dedication.
