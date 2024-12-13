import networkx as nx
from nltk.corpus import stopwords
import nltk
import numpy as np
import os
import re
import requests
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import time
import torch


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def word_analysis(job_desc_sections, resumes_data):
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

def transformers_embedding_analysis(job_desc_sections, resume_sections, model_name='all-mpnet-base-v2'):
    from sentence_transformers import SentenceTransformer, util
    import numpy as np

    model = SentenceTransformer(model_name)
    job_desc_embeddings = model.encode(job_desc_sections, convert_to_numpy=True)
    resume_embeddings = model.encode(resume_sections, convert_to_numpy=True)

    # Compute cosine distances using util.cos_sim on numpy arrays:
    # util.cos_sim can take torch tensors, but we want numpy. Let's do manual cosine on numpy.
    # Alternatively, we can convert embeddings to torch tensors. Let's do manual numpy-based similarity:
    # util.cos_sim expects torch, so let's write a numpy-based cosine similarity function:
    norm_job = job_desc_embeddings / np.linalg.norm(job_desc_embeddings, axis=1, keepdims=True)
    norm_resume = resume_embeddings / np.linalg.norm(resume_embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_job, norm_resume.T)  # shape [num_desc, num_resume]

    return similarity_matrix

def oai_embedding_analysis(job_desc_sections, resume_sections, model_name, api_key):
    import requests
    import numpy as np
    from scipy.spatial.distance import cdist

    def embed_text_openai(texts, model_name, api_key):
        import json
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "input": texts
        }
        response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
        if response.status_code != 200:
            st.error(f"OpenAI API error: {response.status_code}, {response.text}")
            return np.array([])
        json_resp = response.json()
        embeddings = [item['embedding'] for item in json_resp['data']]
        time.sleep(5)
        return np.array(embeddings)

    job_desc_embeddings = embed_text_openai(job_desc_sections, model_name, api_key)
    if job_desc_embeddings.size == 0:
        return np.array([[]])

    resume_embeddings = embed_text_openai(resume_sections, model_name, api_key)
    if resume_embeddings.size == 0:
        return np.array([[]])

    # Compute cosine similarity matrix
    norm_job = job_desc_embeddings / np.linalg.norm(job_desc_embeddings, axis=1, keepdims=True)
    norm_resume = resume_embeddings / np.linalg.norm(resume_embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_job, norm_resume.T)  # shape [num_desc, num_resume]

    return similarity_matrix

def max_per_section_average(similarity_matrix: np.ndarray) -> float:
    """
    Given a similarity matrix (job_desc_sections x resume_sections),
    for each job description section, find the maximum similarity to any resume section.
    Then compute the average of these maximum similarities.
    
    If there are no resume sections (N=0), assign -1.0 similarity for that job description section.
    
    Returns:
        A percentage score (float) representing the match quality.
    """
    max_similarities = np.max(similarity_matrix, axis=1)
    score = np.mean(max_similarities) * 100
    return round(score, 2)

def min_cost_max_flow(similarity_matrix: np.ndarray, C=3) -> float:
    """
    Use a min-cost max-flow approach to solve the assignment problem with capacity C on resume sections.
    Each resume section can cover up to C job description sections.

    similarity_matrix: shape [M, N]
    C: int, capacity per resume section node

    Returns:
        A percentage score (float) representing the match quality.

    Steps:
    1. Construct flow network:
       - source -> job desc nodes with capacity=1
       - resume nodes -> sink with capacity=C
       - job desc -> resume edges with capacity=1 and cost=-similarity
    2. Run min_cost_flow to find minimal cost flow.
    3. Interpret flow as assignments.
    4. Compute average similarity across all M job description sections, 
       if unmatched, -1 for that section.
    """

    M, N = similarity_matrix.shape

    # Create a directed graph
    G = nx.DiGraph()

    source = "S"
    sink = "T"
    G.add_node(source)
    G.add_node(sink)

    # Add edges from source to each job desc section node with capacity=1
    for i in range(M):
        G.add_node(f"J{i}")
        G.add_edge(source, f"J{i}", capacity=1, weight=0)

    # Add edges from each resume section node to sink with capacity=C
    for j in range(N):
        G.add_node(f"R{j}")
        G.add_edge(f"R{j}", sink, capacity=C, weight=0)

    # Add edges between job desc sections and resume sections
    for i in range(M):
        for j in range(N):
            sim = similarity_matrix[i, j]
            cost = -sim  # negative to maximize similarity
            # capacity=1 per pairing
            G.add_edge(f"J{i}", f"R{j}", capacity=1, weight=cost)

    # We want to match all M job desc sections if possible
    # If we can't, some remain unmatched
    # min_cost_flow attempts to find a flow that satisfies all demands, but we have no demands set here.
    # Instead, we'll push as much flow as possible. However, min_cost_flow returns a feasible min cost flow.

    # The trick: without demands, min_cost_flow will push minimal flow (possibly zero).
    # To ensure we push as many matches as possible, we can add demands or run successive shortest path methods.

    # A workaround: Add demands. Let’s say we want to match exactly M job desc sections, if possible.
    # If N*C < M, we can't match all. If N*C >= M, we can match all.
    flow_amount = min(M, N*C)

    G.nodes[source]['demand'] = -flow_amount
    G.nodes[sink]['demand'] = flow_amount

    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        # No feasible flow: all unmatched
        return round(np.mean([-1]*M)*100, 2)

    # Extract matches
    assigned_similarities = []
    for i in range(M):
        matched = False
        for j in range(N):
            f = flow_dict.get(f"J{i}", {}).get(f"R{j}", 0)
            if f > 0:
                # This job desc section i is matched to resume section j
                sim = similarity_matrix[i, j]
                assigned_similarities.append(sim)
                matched = True
                break
        if not matched:
            assigned_similarities.append(-1.0)

    score = np.mean(assigned_similarities) * 100
    return round(score, 2)

# Mapping of models to provider and function
model_provider_map = {
    "all-mpnet-base-v2": {"provider": "transformers", "function": transformers_embedding_analysis},
    "text-embedding-3-small": {"provider": "openai", "function": oai_embedding_analysis},
    "text-embedding-3-large": {"provider": "openai", "function": oai_embedding_analysis}
}