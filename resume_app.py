import streamlit as st
from analysis_ui import run_analysis
from session_state import initialize_session_state
from results_ui import results_page
from upload_ui import upload_job_description, upload_resumes

def main():
    initialize_session_state()
    st.title("Resume Ranker")

    # Sidebar Navigation
    menu = ["Upload Job Description", "Upload Resumes", "Run Analysis", "Results"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Upload Job Description":
        upload_job_description()
    elif choice == "Upload Resumes":
        upload_resumes()
    elif choice == "Run Analysis":
        run_analysis()
    elif choice == "Results":
        results_page()

if __name__ == "__main__":
    main()