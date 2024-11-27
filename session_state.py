# session_state.py
import streamlit as st

def initialize_session_state():
    if 'job_desc_file' not in st.session_state:
        st.session_state['job_desc_file'] = None
    if 'job_desc_sections' not in st.session_state:
        st.session_state['job_desc_sections'] = None
    if 'uploaded_resumes' not in st.session_state:
        st.session_state['uploaded_resumes'] = None
    if 'resumes_data' not in st.session_state:
        st.session_state['resumes_data'] = None
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None

