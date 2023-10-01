from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import streamlit as st

QUESTION_DATA_PATH = Path('data/processed/question_data.npy')
EMBEDDINGS_PATH = Path('data/embeddings/embeddings.npy')
MODEL_PATH = 'sentence-transformers/all-MiniLM-L6-v2'


@st.cache_resource
def load_model():
    model = SentenceTransformer(MODEL_PATH)
    return model


@st.cache_data
def load_embeddings():
    embeddings = np.load(EMBEDDINGS_PATH)
    return embeddings


@st.cache_data
def load_question_data():
    question_data = np.load(QUESTION_DATA_PATH, allow_pickle=True)
    return question_data


def find_similar_questions(text_input: str,  k: int) -> List[List[Dict[str, Any]]]:
    """
    Find similar questions based on text embeddings.

    Args:
        text_input (str): The input text for which similar questions are to be found.
        embeddings (np.array): An array of text embeddings.
        k (int): The number of similar questions to retrieve.

    Returns:
        List[List[Dict[str, Any]]]: A list of list of dictionaries containing similar question ids
        and their similarity scores.
    """
    model = load_model()
    embeddings = load_embeddings()
    text_input_vectorized = model.encode(text_input)
    similar_questions = semantic_search(text_input_vectorized, embeddings, top_k=k)
    return similar_questions


def get_similar_questions_with_score(text_input: str, k=5) -> List[Dict[str, Any]]:
    similar_questions = find_similar_questions(text_input, k)

    question_data = load_question_data()

    corpus_ids = [item['corpus_id'] for item in similar_questions[0]]
    similarity_scores = [round(item['score'] * 100, 1) for item in similar_questions[0]]

    similar_question_data = question_data[corpus_ids]

    results = [{'question': question, 'similarity_score': score}
               for question, score in zip(similar_question_data, similarity_scores)]

    return results
