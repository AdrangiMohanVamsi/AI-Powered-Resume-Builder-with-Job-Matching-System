import streamlit as st
import plotly.express as px
import pandas as pd
from collections import Counter
import spacy


nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

st.title("📊 Analytics Dashboard")
st.markdown("Track your resume performance and gain valuable insights.")

if 'analysis_history' not in st.session_state or not st.session_state.analysis_history:
    st.warning("No analysis has been performed yet. Please go to the 'AI-Powered Resume Analyzer' to analyze a resume.")
else:
    history_df = pd.DataFrame(st.session_state.analysis_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

    st.header("Match Score Trend")
    fig = px.line(history_df, x='timestamp', y='match_score', title='Match Score Over Time', markers=True, labels={"timestamp": "Date", "match_score": "Match Score (%)"})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Match Score Distribution")
        fig_hist = px.histogram(history_df, x='match_score', nbins=10, title='Distribution of Match Scores')
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.header("Keyword Analysis")
        all_jd_text = " ".join(history_df['jd_text'].tolist())
        doc = nlp(all_jd_text.lower())
        words = [token.text for token in doc if token.is_alpha]
        stopwords = ['the', 'a', 'an', 'in', 'to', 'of', 'and', 'for', 'with', 'is', 'on', 'at', 'as'] 
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
        word_counts = Counter(filtered_words)
        keyword_df = pd.DataFrame(word_counts.most_common(15), columns=['Keyword', 'Frequency'])
        fig_keywords = px.bar(keyword_df, x='Frequency', y='Keyword', orientation='h', title='Top 15 Keywords in Job Descriptions')
        fig_keywords.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_keywords, use_container_width=True)

    st.header("Historical Resume Matches")
    if st.session_state.analysis_history:
        grouped_by_jd = history_df.groupby('jd_text')
        for jd, group in grouped_by_jd:
            st.subheader(f"Job Description: {jd[:100]}...") 
            group_sorted = group.sort_values(by='match_score', ascending=False)
            for i, row in group_sorted.iterrows():
                with st.container():
                    st.markdown(f"**Resume:** {row['resume_name']}  **Match Score:** <span style='color: #4CAF50;'>{row['match_score']}%</span>", unsafe_allow_html=True)
                    with st.expander("View Full Analysis"):
                        st.markdown(row['analysis'])
                    with st.expander("View Enhancement Suggestions"):
                        st.markdown(row['enhancement_suggestions'])
                    st.markdown("---")
    else:
        st.info("No historical resume analysis results to display yet.")