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
    """
    Load a pre-trained SentenceTransformer model.

    Returns:
    -------
    SentenceTransformer
        A pre-trained SentenceTransformer model loaded from the specified MODEL_PATH.
    """
    model = SentenceTransformer(MODEL_PATH)
    return model


@st.cache_data
def load_embeddings():
    """
    Load pre-computed embeddings from a file.

    Returns:
    -------
    numpy.ndarray
        A NumPy array containing pre-computed embeddings loaded from the specified EMBEDDINGS_PATH.
    """
    embeddings = np.load(EMBEDDINGS_PATH)
    return embeddings


@st.cache_data
def load_question_data():
    """
    Load question data from a file.

    Returns:
    -------
    numpy.ndarray
        A NumPy array containing question data loaded from the specified QUESTION_DATA_PATH.
    """
    question_data = np.load(QUESTION_DATA_PATH, allow_pickle=True)
    return question_data


def find_similar_questions(text_input: str,  k: int) -> List[List[Dict[str, Any]]]:
    """
    Find similar questions to a given text input using pre-trained embeddings and a semantic search model.

    Parameters:
    ----------
    text_input : str
        The input text for which similar questions are to be found.
    k : int
        The number of similar questions to retrieve.

    Returns:
    -------
    List[List[Dict[str, Any]]]
        A list of lists, where each inner list contains dictionaries representing similar questions.
        Each dictionary has the following keys:
        - 'question': str
            The text of the similar question.
        - 'score': float
            The similarity score between the input text and the similar question.
    """
    model = load_model()
    embeddings = load_embeddings()
    text_input_vectorized = model.encode(text_input)
    similar_questions = semantic_search(text_input_vectorized, embeddings, top_k=k)
    return similar_questions


def get_similar_questions_with_score(text_input: str, k=5) -> List[Dict[str, Any]]:
    """
    Retrieve similar questions to a given text input along with their similarity scores.

    Parameters:
    ----------
    text_input : str
        The input text for which similar questions are to be retrieved.
    k : int, optional (default=5)
        The number of similar questions to retrieve. Default is 5.

    Returns:
    -------
    List[Dict[str, Any]]
        A list of dictionaries representing similar questions and their similarity scores.
        Each dictionary has the following keys:
        - 'question': str
            The text of the similar question.
        - 'similarity_score': float
            The similarity score between the input text and the similar question, rounded to one decimal place.

    Notes:
    -----
    This function uses the `find_similar_questions` function to retrieve similar questions to the input text.
    It also retrieves the corresponding similarity scores and returns the results as a list of dictionaries.
    """
    similar_questions = find_similar_questions(text_input, k)

    question_data = load_question_data()

    corpus_ids = [item['corpus_id'] for item in similar_questions[0]]
    similarity_scores = [round(item['score'] * 100, 1) for item in similar_questions[0]]

    similar_question_data = question_data[corpus_ids]

    results = [{'question': question, 'similarity_score': score}
               for question, score in zip(similar_question_data, similarity_scores)]

    return results
