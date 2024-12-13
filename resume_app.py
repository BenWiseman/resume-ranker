import streamlit as st
import pandas as pd
from resume_parsing import scrape_resumes
from resume_analysis import *
from session_state import initialize_session_state
from upload_ui import upload_job_description, upload_resumes
import plotly.express as px
import os

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

def run_analysis():
    st.header("Run Analysis")

    if 'job_desc_sections' not in st.session_state or not st.session_state['job_desc_sections']:
        st.warning("Please upload a job description on the 'Upload Job Description' page.")
        return

    if 'resumes_data' not in st.session_state or not st.session_state['resumes_data']:
        st.warning("Please upload resumes on the 'Upload Resumes' page.")
        return

    # Add a radio button to select analysis type
    analysis_type = st.radio("Select Analysis Type:", ('Analyze Embeddings', 'Simple Word Count'), index=0)

    # Model selection and API key input as before
    model_options = ["all-mpnet-base-v2", "text-embedding-3-small", "text-embedding-3-large"]
    selected_model = st.selectbox("Select Model:", model_options, index=0)

    provider = "openai" if selected_model.startswith("text-embedding") else "transformers"

    if provider == 'openai':
        default_api_key = os.environ.get("OPENAI_API_KEY", "")
        if default_api_key:
            st.info("OPENAI_API_KEY found in environment.")
        api_key_input = st.text_input("OpenAI API Key:", value=default_api_key if default_api_key else "", disabled=False if not default_api_key else False)
        if not default_api_key and provider == 'openai' and api_key_input.strip() == '':
            st.warning("Please provide an OpenAI API key.")
            run_disabled = True
        else:
            run_disabled = False
    else:
        st.text_input("OpenAI API Key:", value="Not needed for this model", disabled=True)
        api_key_input = None
        run_disabled = False

    if st.button("Run Analysis", disabled=run_disabled):
        if analysis_type == 'Simple Word Count':
            results = word_analysis(st.session_state['job_desc_sections'], st.session_state['resumes_data'])
        else:
            # Embedding analysis: return similarity matrix and do optimal assignment
            # Create a unified function that returns similarity matrix
            resumes_data = st.session_state['resumes_data']
            job_desc_sections = st.session_state['job_desc_sections']

            # We'll build a function that loops over resumes, get similarity matrix and then do optimal assignment
            # For OAI:
            if provider == 'openai':
                def get_similarity_matrix(job_desc_sections, resume_sections):
                    return oai_embedding_analysis(job_desc_sections, resume_sections, selected_model, api_key_input if api_key_input else default_api_key)
            else:
                def get_similarity_matrix(job_desc_sections, resume_sections):
                    return transformers_embedding_analysis(job_desc_sections, resume_sections, model_name=selected_model)

            results = {}
            total_resumes = len(resumes_data)
            progress_bar = st.progress(0)
            for idx, (resume_name, resume_info) in enumerate(resumes_data.items()):
                sections = resume_info.get('selected_sections', [])
                if not sections:
                    score = 0
                else:
                    similarity_matrix = get_similarity_matrix(job_desc_sections, sections)
                    if similarity_matrix.size == 0 or similarity_matrix.shape[0] == 0 or similarity_matrix.shape[1] == 0:
                        score = 0
                    else:
                        score = min_cost_max_flow(similarity_matrix)
                results[resume_name] = score
                progress = (idx + 1) / total_resumes
                progress_bar.progress(progress)

            progress_bar.empty()
        st.session_state['analysis_results'] = results
        st.success("Analysis completed. Check the 'Results' page.")


def results_page():
    st.header("Results")

    if 'analysis_results' not in st.session_state:
        st.warning("No analysis results found. Please run the analysis first.")
        return

    results = st.session_state['analysis_results']

    # Convert results to DataFrame
    df_results = pd.DataFrame(list(results.items()), columns=['Resume', 'Score'])

    # Ensure 'Score' column is numeric
    df_results['Score'] = pd.to_numeric(df_results['Score'])

    # Sort the DataFrame by 'Score' in descending order
    df_results.sort_values(by='Score', ascending=False, inplace=True)

    st.subheader("File Relevance")
    st.write("0-100 scores represent the closeness of each resume to the job description.")
    # Display the DataFrame
    st.dataframe(df_results, height=400, use_container_width=True)

    # Provide option to download the DataFrame as CSV
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='analysis_results.csv',
        mime='text/csv',
    )

    # Slider to select top N resumes to display in the chart
    total_resumes = len(df_results)
    default_n = min(10, total_resumes)
    top_n = st.slider(
        'Select number of top resumes to display in the chart',
        min_value=1,
        max_value=total_resumes,
        value=default_n,
        step=1
    )

    # Filter the DataFrame to top N resumes
    df_top_n = df_results.head(top_n)

    st.subheader(f"Top {top_n} Resume Match Scores Chart")
    fig = px.bar(
        df_top_n,
        x='Resume',
        y='Score',
        labels={'Score': 'Match Score (%)', 'Resume': 'Resume'},
        title=f'Top {top_n} Resumes',
        height=400
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, 100],  # Assuming scores are percentages out of 100
        margin=dict(l=20, r=20, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()