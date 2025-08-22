
import streamlit as st
import plotly.express as px
import pandas as pd
from collections import Counter
import re

st.title("Analytics Dashboard")

# Check if analysis history exists
if 'analysis_history' not in st.session_state or not st.session_state.analysis_history:
    st.warning("No analysis has been performed yet. Please go to the 'AI-Powered Resume Analyzer' to analyze a resume.")
else:
    st.header("Analysis History")
    
    # Match Score Trend
    history_df = pd.DataFrame(st.session_state.analysis_history)
    fig = px.line(history_df, x='timestamp', y='match_score', title='Match Score Trend', markers=True)
    st.plotly_chart(fig)

    st.header("Match Score Distribution")
    fig_hist = px.histogram(history_df, x='match_score', nbins=10, title='Distribution of Match Scores')
    st.plotly_chart(fig_hist)

    # Keyword Analysis from the latest analysis
    st.header("Keyword Analysis (from all job descriptions)")
    all_jd_text = " ".join(history_df['jd_text'].tolist())
    
    # Simple keyword extraction (we can make this more sophisticated later)
    words = re.findall(r'\b\w+\b', all_jd_text.lower())
    # You would typically have a list of stopwords to remove
    stopwords = ['the', 'a', 'an', 'in', 'to', 'of', 'and', 'for', 'with', 'is', 'on', 'at', 'as'] 
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_counts = Counter(filtered_words)
    
    keyword_df = pd.DataFrame(word_counts.most_common(15), columns=['Keyword', 'Frequency'])
    fig_keywords = px.bar(keyword_df, x='Frequency', y='Keyword', orientation='h', title='Top 15 Keywords in All Job Descriptions')

    st.header("Historical Resume Matches")
    if st.session_state.analysis_history:
        # Group by job description for better organization
        grouped_by_jd = history_df.groupby('jd_text')
        
        for jd, group in grouped_by_jd:
            st.subheader(f"Job Description: {jd[:100]}...") # Display first 100 chars of JD
            
            # Sort resumes within each JD group by match score
            group_sorted = group.sort_values(by='match_score', ascending=False)
            
            for i, row in group_sorted.iterrows():
                st.markdown(f"**Resume: {row['resume_name']}** (Match Score: {row['match_score']}%)")
                with st.expander("View Analysis"):
                    st.markdown(row['analysis'])
                with st.expander("View Enhancement Suggestions"):
                    st.markdown(row['enhancement_suggestions'])
                st.markdown("---")
    else:
        st.info("No historical resume analysis results to display yet.")

