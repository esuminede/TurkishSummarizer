## TurkishSummarizer

A concise text summarization tool. It reads a PDF, processes and chunks sentences, applies embeddings and clustering, selects the most relevant context, and generates a summary using a local LLM.

### Requirements
- Python 3.9+
- Packages: pymupdf, nltk, sentence-transformers, scikit-learn, numpy, ollama

### Installation
```bash
pip install pymupdf nltk sentence-transformers scikit-learn numpy ollama
```
NLTK resources will be downloaded automatically on first run.

### Configuration
In `config.py`:
- `FILE_NAME`: PDF file to summarize (e.g., "sample2.pdf")
- `EMBEDDING_MODEL`: Sentence-Transformers model name (e.g., 'intfloat/multilingual-e5-large-instruct')

### Usage
```bash
python main.py
```
The generated summary is written to `summary.txt`.

Notes:
- `ollama` must be installed and the `mistral:7b-instruct` model available locally.
- `deneme.py` is the original single-file version; `main.py` runs the same logic using modular functions.