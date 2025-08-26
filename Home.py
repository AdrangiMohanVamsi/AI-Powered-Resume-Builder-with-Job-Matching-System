import streamlit as st

st.set_page_config(
    page_title="Welcome to AI-Powered Resume Toolkit",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more professional look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .st-emotion-cache-1y4p8pa {
        width: 100%;
        padding: 2rem 1rem 10rem;
        max-width: 100rem;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 12px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .st-emotion-cache-1v0mbdj:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Welcome to the AI-Powered Resume Toolkit! ✨")

st.markdown("#### Your one-stop solution for crafting the perfect resume and acing your job applications.")

st.markdown("---")

st.markdown("### What would you like to do today?")

# Using columns for a card-based layout
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("### 🚀 Build a New Resume")
        st.markdown("Create a professional resume from scratch with our dynamic builder.")
        if st.button("Get Started", key="build_resume", use_container_width=True):
            st.switch_page("pages/1_Resume_Builder.py")

    with st.container():
        st.markdown("### 🔬 Analyze a Resume")
        st.markdown("Upload your resume and a job description to get an in-depth analysis and match score.")
        if st.button("Analyze Now", key="analyze_resume", use_container_width=True):
            st.switch_page("pages/4_Resume_Analyzer.py")

with col2:
    with st.container():
        st.markdown("### 📊 View Analytics Dashboard")
        st.markdown("Track your resume analysis history and view trends to improve your strategy.")
        if st.button("View Dashboard", key="view_dashboard", use_container_width=True):
            st.switch_page("pages/2_Analytics_Dashboard.py")

    with st.container():
        st.markdown("### ✨ Visualize Workflow")
        st.markdown("Understand the AI-powered analysis process from start to finish.")
        if st.button("Visualize", key="visualize_workflow", use_container_width=True):
            st.switch_page("pages/3_Workflow_Visualization.py")

st.markdown("---")

st.markdown("#### How to get the most out of this toolkit:")
st.markdown("""
- **Start with the Resume Builder:** If you don't have a resume, this is the best place to start.
- **Tailor Your Resume:** Use the Resume Analyzer to tailor your resume for each job application.
- **Track Your Progress:** The Analytics Dashboard will help you see how your resume performs over time.
""")
