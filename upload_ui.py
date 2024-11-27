# depends
import streamlit as st
from resume_parsing import scrape_resumes
import pandas as pd


def upload_job_description():
    #import streamlit as st
    st.header("Upload Job Description")

    # Adjustable min_words parameter
    min_words = st.number_input("Minimum Words per Section", min_value=1, value=4, key='min_words_job_desc')

    # Upload Job Description
    job_desc_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt', 'json'], key='job_desc_uploader')

    if job_desc_file is not None:
        # Save the uploaded file in session_state
        st.session_state['job_desc_file'] = job_desc_file
    elif st.session_state['job_desc_file'] is not None:
        # Use the file from session_state
        job_desc_file = st.session_state['job_desc_file']
        st.info(f"Using previously uploaded file: {job_desc_file.name}")
    else:
        st.session_state['job_desc_sections'] = None
        st.warning("Please upload a job description.")
        return

    # Process the job description if not already processed
    if st.session_state['job_desc_sections'] is None:
        job_desc_data = scrape_resumes([job_desc_file], remove_below_n=min_words)
        if job_desc_data:
            filename = job_desc_file.name
            sections = job_desc_data[filename]['sections']
            st.session_state['job_desc_sections'] = sections
        else:
            st.error("Failed to process the job description.")
            return
    else:
        sections = st.session_state['job_desc_sections']

    #if st.button("Clear Job Description"):
    #    st.session_state['job_desc_sections'] = None
    #    st.session_state['job_desc_file'] = None
    #    st.success("Job description cleared.")  

    st.subheader("Job Description Sections")

    # Create a DataFrame with the sections
    df = pd.DataFrame({
        'Include this section': True,
        'Section': sections
    })

    # Display the DataFrame using st.data_editor
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key='job_desc_editor'
    )

    # Get the selected sections
    selected_sections = edited_df[edited_df['Include this section']]['Section'].tolist()

    st.session_state['job_desc_sections'] = selected_sections

def upload_resumes():
    st.header("Upload Resumes")

    # Adjustable min_words parameter
    min_words = st.number_input("Minimum Words per Section", min_value=1, value=4, key='min_words_resumes')

    # Upload Resumes
    uploaded_resumes = st.file_uploader("Upload Resumes", type=['pdf', 'docx', 'txt', 'json'], accept_multiple_files=True, key='resumes_uploader')

    if uploaded_resumes:
        # Save the uploaded resumes in session_state
        st.session_state['uploaded_resumes'] = uploaded_resumes
    elif st.session_state['uploaded_resumes']:
        # Use the resumes from session_state
        uploaded_resumes = st.session_state['uploaded_resumes']
        st.info(f"Using previously uploaded resumes: {[file.name for file in uploaded_resumes]}")
    else:
        st.session_state['resumes_data'] = None
        st.warning("Please upload resumes.")
        return

    # Process the resumes if not already processed
    if st.session_state['resumes_data'] is None:
        resumes_data = scrape_resumes(uploaded_resumes, remove_below_n=min_words)
        if resumes_data:
            st.session_state['resumes_data'] = resumes_data
        else:
            st.error("Failed to process the resumes.")
            return
    else:
        resumes_data = st.session_state['resumes_data']

    resume_tabs = st.tabs([filename for filename in resumes_data.keys()])
    for idx, (filename, resume_info) in enumerate(resumes_data.items()):
        sections = resume_info['sections']

        with resume_tabs[idx]:
            st.subheader(f"Resume: {filename}")

            # Create a DataFrame with the sections
            df = pd.DataFrame({
                'Include this section': True,
                'Section': sections
            })

            # Display the DataFrame using st.data_editor
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=f'resume_editor_{filename}'
            )

            # Get the selected sections
            selected_sections = edited_df[edited_df['Include this section']]['Section'].tolist()

            st.session_state['resumes_data'][filename]['selected_sections'] = selected_sections

