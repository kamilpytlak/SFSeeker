import pandas as pd
import streamlit as st

from utils import get_similar_questions_with_score

LOGO_PATH = 'img/logo.jpg'

st.image(LOGO_PATH, width=200)

st.title('SF Seeker')

st.markdown("""
Sci-Fi Stack Exchange Seeker (aka SF Seeker) is an AI assistant that helps you write better questions and search for
semantically similar questions on Sci-Fi Stack Exchange (https://scifi.stackexchange.com/). An all-MiniLM-L6-v2
language model (transformer) was used.

**Features**
- 🔎 Based on a database of 71,013 questions, it searches for the most semantically similar questions to the one entered
by the user. This supports the process of fiding the same/similar questions already asked and prevents the creation of
duplicate threads.
- 👨‍⚕️ [IN PROGRESS] Indicates words in a question that have a negative and positive effect on the chance of
getting an answer. It supports the process of arranging more precise questions. A model based on gradient
reinforcement learned using TF-IDF features was used.
""")

question_input = st.text_area('Question')
k_similar_questions = st.number_input('k similar questions', min_value=1, max_value=100, value=5, step=1)

if st.button('Submit'):
    if not question_input:
        st.warning('⚠️ No question inputted!')
    else:
        question_score_results = get_similar_questions_with_score(question_input, k_similar_questions)
        question_score_results_df = pd.DataFrame(question_score_results)
        question_score_results_df.columns = ['Question', 'Similarity score (in %)']

        st.dataframe(question_score_results_df)
