
import streamlit as st
import graphviz

# Define descriptions for each node
node_descriptions = {
    "resume_parser": "Parses the uploaded resume to extract text content.",
    "jd_parser": "Parses the provided job description to extract text content.",
    "semantic_matcher": "Calculates the semantic similarity between the resume and job description.",
    "gemini_analyzer": "Analyzes the resume and job description using Gemini AI, providing a detailed match analysis.",
    "content_enhancer": "Generates actionable suggestions to enhance the resume based on the job description.",
    "END": "The end point of the workflow."
}

# Define more appealing colors
node_colors = {
    "default": "#AEC6CF", # Light blue-gray
    "end": "#FFDDC1" # Light peach
}

st.title("Workflow Visualization")

if 'graph_representation' not in st.session_state:
    st.warning("The graph representation is not available. Please run the application from the main page.")
else:
    st.header("Workflow Diagram")
    with st.spinner("Generating beautiful graph visualization..."):
        # Create a Graphviz Digraph object
        dot = graphviz.Digraph(comment='LangGraph Workflow', graph_attr={'rankdir': 'LR'})

        # Add nodes
        for node_name in st.session_state.graph_representation["nodes"]:
            label = node_name.replace('_', ' ').title() # Make label more readable
            description = node_descriptions.get(node_name, "No description available.")
            dot.node(node_name, label, shape='box', style='filled', fillcolor=node_colors["default"], tooltip=description)
        
        # Handle the END node explicitly
        dot.node("END", "End Workflow", shape='doublecircle', style='filled', fillcolor=node_colors["end"], tooltip=node_descriptions["END"])

        # Add edges
        for start_node, end_node in st.session_state.graph_representation["edges"]:
            # LangGraph uses the actual END object, convert to string for Graphviz
            if end_node == "__end__":
                end_node_str = "END"
            else:
                end_node_str = str(end_node)
            dot.edge(str(start_node), end_node_str)

        # Render the graph to a Streamlit image
        st.graphviz_chart(dot)

