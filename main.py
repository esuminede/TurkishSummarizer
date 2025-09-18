import pymupdf
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import FILE_NAME, STOPWORDS_FILE, EMBEDDING_MODEL
import nltk
import nltk.tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
import ollama
from typing import List, Tuple


# ----------------------------- PDF -----------------------------

def read_pdf_lower(file_path: str) -> str:
    doc = pymupdf.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text().lower()
    return full_text


# ----------------------------- NLP -----------------------------

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() != ""]


def tokenize_without_stopwords(sentences: List[str]) -> List[str]:
    nltk.download('punkt_tab')
    # nltk.download('stopwords')
    stop_words = stopwords.words('turkish')
    tokens_no_stop: List[str] = []
    for line in sentences:
        tokens = word_tokenize(line)
        for token in tokens:
            if token not in stop_words:
                tokens_no_stop.append(token)
    return tokens_no_stop


def rebuild_sentences_from_tokens(tokens: List[str]) -> List[str]:
    combined_sentences: List[str] = []
    current_sentence: List[str] = []
    for word in tokens:
        current_sentence.append(word)
        if word in ['.', '!', '?']:
            new_sentence = ' '.join(current_sentence)
            combined_sentences.append(new_sentence)
            current_sentence = []
    if current_sentence:
        sentence = ' '.join(current_sentence)
        combined_sentences.append(sentence)
    return combined_sentences


# ----------------------------- Chunking -----------------------------

def chunk_sentences(sentences: List[str], chunk_sentence_size: int) -> List[List[str]]:
    return [sentences[i:i + chunk_sentence_size] for i in range(0, len(sentences), chunk_sentence_size)]


# ----------------------------- Embedding -----------------------------

def embed_chunks_mean(chunks: List[List[str]], model_name: str) -> Tuple[List[np.ndarray], np.ndarray]:
    embedding_model = SentenceTransformer(model_name)
    embedded_chunks: List[np.ndarray] = []
    for i in range(len(chunks) - 1):
        volitile_vector = embedding_model.encode(chunks[i])
        embedded_chunks.append(volitile_vector)
    embedded_chunks_2d = np.mean(embedded_chunks, axis=1)
    return embedded_chunks, embedded_chunks_2d


# ----------------------------- Clustering -----------------------------

def kmeans_cluster(embedded_chunks_2d: np.ndarray, num_clusters: int) -> Tuple[KMeans, np.ndarray]:
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(embedded_chunks_2d)
    cluster_labels = clustering_model.labels_
    return clustering_model, cluster_labels


# ----------------------------- Scoring -----------------------------

def compute_cluster_scores(embedded_chunks_2d: np.ndarray, cluster_labels: np.ndarray, clustering_model: KMeans) -> List[float]:
    cluster_scores: List[float] = []
    for embedding, cluster_id in zip(embedded_chunks_2d, cluster_labels):
        center = clustering_model.cluster_centers_[cluster_id]
        distance = np.linalg.norm(embedding - center)
        score = 1 / (1 + distance)
        cluster_scores.append(score)
    return cluster_scores


def select_top_with_context(cluster_scores: List[float], chunk_list: List[List[str]], top_percentage: float = 0.1) -> List[List[str]]:
    n_chunks = len(cluster_scores)
    n_select = int(n_chunks * top_percentage)
    sorted_indices = np.argsort(cluster_scores)[::-1]
    top_indices = sorted_indices[:n_select]

    selected_chunks: List[List[str]] = []
    extended_indices_for_context: List[int] = []

    for idx in top_indices:
        extended_indices_for_context.append(int(idx))
        if idx > 0:
            extended_indices_for_context.append(int(idx - 1))
            selected_chunks.append(chunk_list[int(idx)])
        if idx < n_chunks - 1:
            extended_indices_for_context.append(int(idx + 1))
            selected_chunks.append(chunk_list[int(idx)])
        selected_chunks.append(chunk_list[int(idx)])
    return selected_chunks


# ----------------------------- Summarization -----------------------------

def summarize_chunks(selected_chunks: List[List[str]]) -> str:
    response = ollama.chat(model='mistral:7b-instruct', messages=[
        {
            'role': 'user',
            'content': f"""Aşağıdaki metni özetle. KESİNLİKLE KENDİ YORUMUNU EKLEME. SADECE METİNDEKİ BİLGİLERİ KULLAN.
    ÖZETLEME TALİMATLARI:
    1. Metnin ana konusunu belirt
    2. Önemli olayları kronolojik sırayla anlat
    3. Karakter gelişimlerini ve ilişkilerini özetle
    4. Metnin atmosferini ve tonunu koru
    
    METİN: 
    {selected_chunks}""",
        }
    ], options={
        'temperature': 0.3,
        'top_p': 0.9,
        'min_tokens': 1000
    })
    return response['message']['content']


def write_summary(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ----------------------------- Orchestration -----------------------------

def main() -> None:
    full_text = read_pdf_lower(FILE_NAME)

    sentences = split_sentences(full_text)

    tokens_no_stop = tokenize_without_stopwords(sentences)

    combined_sentences = rebuild_sentences_from_tokens(tokens_no_stop)

    chunk_sentence_size = 1
    chunk_list = chunk_sentences(combined_sentences, chunk_sentence_size)

    embedded_chunks, embedded_chunks_2d = embed_chunks_mean(chunk_list, EMBEDDING_MODEL)

    num_clusters = 1
    clustering_model, cluster_labels = kmeans_cluster(embedded_chunks_2d, num_clusters)

    cluster_scores = compute_cluster_scores(embedded_chunks_2d, cluster_labels, clustering_model)

    selected_chunks = select_top_with_context(cluster_scores, chunk_list, top_percentage=0.1)

    summary = summarize_chunks(selected_chunks)

    write_summary("summary.txt", summary)


if __name__ == "__main__":
    main()