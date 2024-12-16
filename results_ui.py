import plotly.express as px
import streamlit as st
import pandas as pd

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

