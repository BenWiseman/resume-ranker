import streamlit as st
from resume_analysis import *

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
                        similarity_matrix = stable_tie_break(similarity_matrix)
                        score = min_cost_max_flow(similarity_matrix) #max_per_section_average(similarity_matrix)#
                results[resume_name] = score
                progress = (idx + 1) / total_resumes
                progress_bar.progress(progress)

            progress_bar.empty()
        st.session_state['analysis_results'] = results
        st.success("Analysis completed. Check the 'Results' page.")
