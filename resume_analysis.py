# Description: This file contains the functions that performs the analysis of the job description and resumes.
import re
import streamlit as st

# Perform word-matching analysis of the job description and resumes
def word_analysis(job_desc_sections, resumes_data):
    import re
    import nltk
    from nltk.corpus import stopwords

    # Download stopwords if not already present
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    results = {}
    job_desc_text = ' '.join(job_desc_sections).lower()
    job_desc_words = set(re.findall(r'\b\w+\b', job_desc_text))
    # Remove stopwords
    job_desc_words = job_desc_words - stop_words

    for resume_name, resume_info in resumes_data.items():
        sections = resume_info.get('selected_sections', [])
        resume_text = ' '.join(sections).lower()
        resume_words = set(re.findall(r'\b\w+\b', resume_text))
        # Remove stopwords
        resume_words = resume_words - stop_words

        if len(job_desc_words) == 0:
            score = 0
        else:
            common_words = job_desc_words.intersection(resume_words)
            score = round(len(common_words) / len(job_desc_words) * 100, 0)  # percentage
        results[resume_name] = score

    return results


def embedding_analysis(job_desc_sections, resumes_data):
    import numpy as np
    from sentence_transformers import SentenceTransformer, util
    import time

    # Initialize model
    # multi-qa-mpnet-base-cos-v1: designed for semantic search
    # paraphrase-MiniLM-L6-v2: lightweight model
    # all-mpnet-base-v2 (default): good performance
    model_name = 'all-mpnet-base-v2'  # You can adjust the model as needed
    model = SentenceTransformer(model_name)

    # Combine job description sections into one text
    job_desc_text = ' '.join(job_desc_sections)

    # Embed job description
    job_desc_embedding = model.encode(job_desc_text, convert_to_tensor=True)

    results = {}
    total_resumes = len(resumes_data)
    progress_bar = st.progress(0)
    for idx, (resume_name, resume_info) in enumerate(resumes_data.items()):
        sections = resume_info.get('selected_sections', [])
        resume_text = ' '.join(sections)
        if resume_text.strip() == '':
            # No content to compare
            score = 0
        else:
            # Embed resume
            resume_embedding = model.encode(resume_text, convert_to_tensor=True)

            # Compute cosine similarity
            cosine_sim = util.cos_sim(job_desc_embedding, resume_embedding)
            score = round(cosine_sim.item() * 100, 0)  # Convert to percentage

        results[resume_name] = score

        # Update progress bar
        progress = (idx + 1) / total_resumes
        progress_bar.progress(progress)

    progress_bar.empty()  # Remove progress bar when done

    return results