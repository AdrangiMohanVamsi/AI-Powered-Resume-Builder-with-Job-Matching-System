import streamlit as st

st.set_page_config(
    page_title="Welcome to AI-Powered Resume Toolkit",
    page_icon="👋",
    layout="wide"
)

st.title("Welcome to the AI-Powered Resume Toolkit! 👋")

st.markdown("""
This application is a comprehensive toolkit designed to assist you in your job application process. 
It leverages AI to analyze, build, and manage your resumes effectively.

### What would you like to do today?
""")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Build a New Resume", use_container_width=True):
        st.switch_page("pages/1_Resume_Builder.py")
    if st.button("📊 View Analytics Dashboard", use_container_width=True):
        st.switch_page("pages/2_Analytics_Dashboard.py")

with col2:
    if st.button("🔬 Analyze a Resume", use_container_width=True):
        st.switch_page("pages/4_Resume_Analyzer.py")
    if st.button("✨ Visualize Workflow", use_container_width=True):
        st.switch_page("pages/3_Workflow_Visualization.py")

st.markdown("""

---

#### How to get started:

1.  **Build a New Resume:** Head over to the Resume Builder to create a professional resume from scratch.
2.  **Analyze a Resume:** Upload your existing resume and a job description to get an in-depth analysis and a match score.
3.  **View Analytics Dashboard:** Track your resume analysis history and view trends.
4.  **Visualize Workflow:** Understand the inner workings of the AI analysis process.

Choose an option above to begin!
""")
