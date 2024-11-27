# Description: This script is the main script that runs the Streamlit app.
import streamlit as st
import pandas as pd
from resume_parsing import scrape_resumes
from resume_analysis import word_analysis, embedding_analysis
from session_state import initialize_session_state
from upload_ui import upload_job_description, upload_resumes
import plotly.express as px

def main():
    initialize_session_state()
    st.title("Resume Ranker")
    #st.text("By Guddge")

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

    #st.write("Click the button below to run the analysis.")
    if st.button("Run Analysis"):
        if analysis_type == 'Analyze Embeddings':
            results = embedding_analysis(st.session_state['job_desc_sections'], st.session_state['resumes_data'])
        else:
            results = word_analysis(st.session_state['job_desc_sections'], st.session_state['resumes_data'])
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

    """
    
    
    """    

    # Slider to select top N resumes to display in the chart
    total_resumes = len(df_results)
    default_n = min(10, total_resumes)

    # Filter the DataFrame to top N resumes
    df_top_n = df_results.head(default_n)

    #st.subheader(f"Top 10 Resumes")
    # Create a Plotly bar chart
    fig = px.bar(
        df_top_n,
        x='Resume',
        y='Score',
        labels={'Score': 'Match Score (%)', 'Resume': 'Resume'},
        title=f'Top 10 Resumes',
        height=400
    )
    #fig.update_layout(
    #    xaxis_tickangle=-45,
    #    yaxis_range=[0, 100],  # Assuming scores are percentages out of 100
    #    margin=dict(l=20, r=20, t=50, b=50)
    #)
    st.plotly_chart(fig, use_container_width=True)
    
    #top_n = st.slider(
    #'Select number of top resumes to display',
    #min_value=1,
    #max_value=total_resumes,
    #value=default_n,
    #step=1
    #)


if __name__ == "__main__":
    main()