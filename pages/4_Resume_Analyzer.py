import streamlit as st
import PyPDF2
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
import numpy as np
import chromadb
import pandas as pd

load_dotenv()

ANALYSIS_HISTORY_FILE = Path("analysis_history.json")
CHROMA_DB_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


resume_collection = chroma_client.get_or_create_collection(
    name="resume_embeddings",
    
)


class GraphState(TypedDict):
    resume_text: str
    jd_text: str
    resume_embedding: Any
    jd_embedding: Any
    similarity_score: float
    analysis: str
    enhancement_suggestions: str
    api_key: str


def resume_parser_node(state):
    st.session_state.messages.append("Parsing resume...")
    resume_text = state["resume_text"]
    st.session_state.messages.append(f"Resume parsed. Length: {len(resume_text)} characters.")
    return {"resume_text": resume_text}

def jd_parser_node(state):
    st.session_state.messages.append("Parsing job description...")
    jd_text = state["jd_text"]
    st.session_state.messages.append(f"Job description parsed. Length: {len(jd_text)} characters.")
    return {"jd_text": jd_text}

def semantic_matcher_node(state):
    st.session_state.messages.append("Performing semantic matching with ChromaDB...")
    resume_text = state["resume_text"]
    jd_text = state["jd_text"]

    model = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        
        resume_embedding_np = model.encode(resume_text, convert_to_numpy=True)
        
        
        jd_embedding_np = model.encode(jd_text, convert_to_numpy=True)

        
        resume_id = str(hash(resume_text))

        
        try:
            
            existing_entry = resume_collection.get(ids=[resume_id], include=[])
            if not existing_entry['ids']:
                resume_collection.add(
                    embeddings=[resume_embedding_np.tolist()], 
                    documents=[resume_text], 
                    metadatas=[{"type": "resume", "id": resume_id}],
                    ids=[resume_id]
                )
                st.session_state.messages.append(f"Added resume {resume_id} to ChromaDB.")
            else:
                st.session_state.messages.append(f"Resume {resume_id} already in ChromaDB.")
        except Exception as e:
            st.session_state.messages.append(f"Error adding resume to ChromaDB: {e}")
            

        
        results = resume_collection.query(
            query_embeddings=[jd_embedding_np.tolist()],
            n_results=1, 
        )
        
        similarity_score = 0.0
        if results and results['distances'] and results['distances'][0]:
            
            if results['embeddings'] and results['embeddings'][0]:
                top_resume_embedding_from_db = results['embeddings'][0][0] 
                
                similarity = cosine_similarity(jd_embedding_np.reshape(1, -1), 
                                               np.array(top_resume_embedding_from_db).reshape(1, -1))[0][0]
                similarity_score = round(similarity * 100, 2) 
                st.session_state.messages.append(f"Semantic similarity score (from ChromaDB query): {similarity_score:.2f}%")
            else:
                st.session_state.messages.append("No embeddings returned from ChromaDB query.")
        else:
            st.session_state.messages.append("No results found in ChromaDB for the job description query.")

    except Exception as e:
        st.session_state.messages.append(f"Error during semantic matching with ChromaDB: {e}")
        similarity_score = 0.0 
    

    return {"resume_embedding": resume_embedding_np, "jd_embedding": jd_embedding_np, "similarity_score": similarity_score}

def gemini_analyzer_node(state):
    st.session_state.messages.append("Analyzing with Gemini...")
    api_key = state["api_key"]
    resume_text = state["resume_text"]
    jd_text = state["jd_text"]
    similarity_score = state["similarity_score"]
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an expert HR analyst. Please analyze the following resume and job description.
        A semantic similarity score between the resume and job description is {similarity_score:.2f}%.
        Provide a detailed analysis of how well the resume matches the job description.
        Include the following sections in your analysis:
        1.  **Overall Summary:** A brief overview of the match.
        2.  **Strengths:** Key areas where the candidate's experience aligns with the job requirements.
        3.  **Areas for Improvement:** Suggestions for how the resume could be tailored to better fit the role.
        4.  **Match Score:** A percentage score indicating the overall match quality. IMPORTANT: The score should be on its own line, like this: **Match Score:** 85%
        5.  **Overall Match Score:** A final combined score considering all factors, including semantic similarity. IMPORTANT: This score should be on its own line, like this: **Overall Match Score:** 90%

        **Resume:**
        {resume_text}

        **Job Description:**
        {jd_text}
        """
        response = model.generate_content(prompt)
        analysis = response.text
        st.session_state.messages.append("Gemini analysis complete.")
    except Exception as e:
        analysis = f"An error occurred: {e}"
        st.session_state.messages.append(f"Gemini analysis failed: {e}")
    
    return {"analysis": analysis}

def content_enhancer_node(state):
    st.session_state.messages.append("Generating enhancement suggestions...")
    api_key = state["api_key"]
    resume_text = state["resume_text"]
    jd_text = state["jd_text"]
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are a career coach. Based on the following resume and job description, 
        provide specific, actionable suggestions for how the candidate can improve their resume.
        Focus on quantifiable achievements, tailoring the language to the job description, and explicitly identifying any skill gaps.
        Include a dedicated section for "Identified Skill Gaps" if applicable.

        **Resume:**
        {resume_text}

        **Job Description:**
        {jd_text}
        """
        response = model.generate_content(prompt)
        enhancement_suggestions = response.text
        st.session_state.messages.append("Enhancement suggestions generated.")
    except Exception as e:
        enhancement_suggestions = f"An error occurred: {e}"
        st.session_state.messages.append(f"Enhancement suggestion generation failed: {e}")
        
    return {"enhancement_suggestions": enhancement_suggestions}


workflow = StateGraph(GraphState)
graph_representation = {"nodes": set(), "edges": []}

def add_node_and_represent(name, node):
    workflow.add_node(name, node)
    graph_representation["nodes"].add(name)

def add_edge_and_represent(start, end):
    workflow.add_edge(start, end)
    graph_representation["edges"].append((start, end))

add_node_and_represent("resume_parser", resume_parser_node)
add_node_and_represent("jd_parser", jd_parser_node)
add_node_and_represent("semantic_matcher", semantic_matcher_node)
add_node_and_represent("gemini_analyzer", gemini_analyzer_node)
add_node_and_represent("content_enhancer", content_enhancer_node)

workflow.set_entry_point("resume_parser")
add_edge_and_represent("resume_parser", "jd_parser")
add_edge_and_represent("jd_parser", "semantic_matcher")
add_edge_and_represent("semantic_matcher", "gemini_analyzer")
add_edge_and_represent("gemini_analyzer", "content_enhancer")
add_edge_and_represent("content_enhancer", END)

app = workflow.compile()


st.session_state.graph_representation = graph_representation


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_match_score(analysis_text):
    
    overall_match = re.search(r"Overall Match Score:.*?(\d+)%", analysis_text)
    if overall_match:
        return int(overall_match.group(1))
    
    
    match = re.search(r"Match Score:.*?(\d+)%", analysis_text)
    if match:
        return int(match.group(1))
    
    return 0 

def load_analysis_history():
    if ANALYSIS_HISTORY_FILE.exists():
        with open(ANALYSIS_HISTORY_FILE, "r") as f:
            history = json.load(f)
            
            for entry in history:
                if "timestamp" in entry and isinstance(entry["timestamp"], str):
                    entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
            return history
    return []

def save_analysis_history(history):
    with open(ANALYSIS_HISTORY_FILE, "w") as f:
        
        serializable_history = []
        for entry in history:
            serializable_entry = entry.copy()
            if "timestamp" in serializable_entry and isinstance(serializable_entry["timestamp"], datetime):
                serializable_entry["timestamp"] = serializable_entry["timestamp"].isoformat()
            serializable_history.append(serializable_entry)
        json.dump(serializable_history, f, indent=4)


st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")
st.title("AI-Powered Resume Analyzer 🤖")

api_key = os.getenv("GOOGLE_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = load_analysis_history()
if 'all_resume_results' not in st.session_state:
    st.session_state.all_resume_results = []

if not api_key or api_key == "YOUR_API_KEY":
    st.error("Please set your GOOGLE_API_KEY in the .env file.")
else:
    st.header("Upload Your Resume and Job Description")
    uploaded_resumes = st.file_uploader("Upload resumes (PDFs)", type=["pdf"], accept_multiple_files=True)
    job_description = st.text_area("Paste the job description here")

    if st.button("Analyze Resumes"):
        if uploaded_resumes and job_description:
            all_results = []
            st.session_state.messages = [] 
            st.session_state.all_resume_results = [] 

            with st.spinner("Running analysis for all resumes..."):
                total_resumes = len(uploaded_resumes)
                progress_bar = st.progress(0)
                for i, uploaded_resume in enumerate(uploaded_resumes):
                    st.session_state.messages.append(f"Processing resume: {uploaded_resume.name}")
                    resume_text = extract_text_from_pdf(uploaded_resume)
                    
                    inputs = {"resume_text": resume_text, "jd_text": job_description, "api_key": api_key}
                    
                    
                    workflow_steps = []
                    accumulated_state = {}
                    for step in app.stream(inputs):
                        node_name = list(step.keys())[0]
                        node_output = step[node_name]
                        
                        
                        
                        accumulated_state.update(node_output)

                        
                        display_output = node_output.copy()
                        if "resume_embedding" in display_output:
                            display_output["resume_embedding"] = "Embedding generated (not shown)"
                        if "jd_embedding" in display_output:
                            display_output["jd_embedding"] = "Embedding generated (not shown)"
                        
                        workflow_steps.append({
                            "node": node_name,
                            "output": display_output
                        })

                    result = accumulated_state
                    match_score = extract_match_score(result['analysis'])
                    
                    all_results.append({
                        "resume_name": uploaded_resume.name,
                        "match_score": match_score,
                        "analysis": result['analysis'],
                        "enhancement_suggestions": result['enhancement_suggestions'],
                        "timestamp": datetime.now(),
                        "workflow_steps": workflow_steps 
                    })
                    st.session_state.messages.append(f"Finished processing {uploaded_resume.name}. Match Score: {match_score}%")
                    progress_bar.progress((i + 1) / total_resumes, text=f"Analyzing resume {i + 1} of {total_resumes}")

            st.success("Analysis complete for all resumes!")

            all_results.sort(key=lambda x: x['match_score'], reverse=True) 
            st.session_state.all_resume_results = all_results 

 
            for result in st.session_state.all_resume_results:
                st.session_state.analysis_history.append({
                    "timestamp": result['timestamp'],
                    "match_score": result['match_score'],
                    "jd_text": job_description, 
                    "analysis": result['analysis'],
                    "resume_name": result['resume_name'], 
                    "enhancement_suggestions": result['enhancement_suggestions'] 
                })
            save_analysis_history(st.session_state.analysis_history)

        else:
            st.warning("Please upload at least one resume and provide a job description.")

    if 'all_resume_results' in st.session_state and st.session_state.all_resume_results:
        st.header("Top Resumes Matching the Job Description")
        
        max_n = len(st.session_state.all_resume_results)
        if max_n > 0:
            num_to_display = st.number_input("Select number of top resumes to display", min_value=1, max_value=max_n, value=max_n, step=1)

            for i, result in enumerate(st.session_state.all_resume_results[:num_to_display]):
                st.subheader(f"{i+1}. {result['resume_name']} (Match Score: {result['match_score']}%)")
                with st.expander("View Analysis"):
                    st.markdown(result['analysis'])
                with st.expander("View Enhancement Suggestions"):
                    st.markdown(result['enhancement_suggestions'])
                with st.expander("View Explainable AI Workflow Trace"):
                    for step in result.get("workflow_steps", []):
                        st.write(f"**Node:** `{step['node']}`")
                        st.json(step['output'])
                st.markdown("---")


    st.header("Workflow Log")
    if st.session_state.messages:
        log_container = st.container()
        with log_container:
            for msg in st.session_state.messages:
                st.text(msg)
    else:
        st.info("No workflow messages to display yet. Run an analysis to see the log.")
        
    st.header("ChromaDB Status")
    try:
        count = resume_collection.count()
        st.success(f"ChromaDB is active. Collection '{resume_collection.name}' contains {count} resume embeddings.")
    except Exception as e:
        st.error(f"Could not connect to ChromaDB: {e}")
        st.warning("Semantic matching features will be limited. Please ensure ChromaDB is running.")
        
    st.header("Historical Resume Matches")
    if st.session_state.analysis_history:
        
        history_df = pd.DataFrame(st.session_state.analysis_history)
        grouped_by_jd = history_df.groupby('jd_text')
        
        for jd, group in grouped_by_jd:
            st.subheader(f"Job Description: {jd[:100]}...") 
            
            
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