import streamlit as st
import graphviz

# Page Configuration
st.set_page_config(
    page_title="Fake News Detector - ESTG",
    page_icon="🛡️",
    layout="wide"
)

# Header
st.title("🛡️ Fake News Detection System: Multi-Model Approach")
st.write("---")

# Introdução baseada no relatório
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 1. Contextualization and Objectives")
    st.info("""
    This system aims to combat misinformation through a hierarchical modular architecture. 
    Unlike monolithic models, we use **independent expert models** (TTopics, Stance, Consistency) 
    whose outputs feed a **Final Meta-Classifier**.
    """)
    
    st.markdown("#### 2. System Modules")
    st.markdown("""
    * **Model 1 (TTopics):** Identifies if it is Politics, Health, Technology, etc.
    * **Model 2 (Stance):** Checks if the headline matches the body of the news.
    * **Model 3 (Consistency):** Analyzes semantic contradictions.
    * **Meta-Model:** Aggregates probabilities for the final decision.
    """)
with col2:
    st.markdown("#### 🔄 Solution Architecture")
    # Visual diagram of the architecture described in section 2 of the PDF
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    
    graph.node('I', 'Input (News)', shape='box', style='filled', fillcolor='lightblue')
    graph.node('P', 'Pre-processing', shape='box', style='filled', fillcolor='lightgrey')
    
    with graph.subgraph(name='cluster_models') as c:
        c.attr(label='Expert Models (Level 1)')
        c.node('M1', 'M1: TTopics')
        c.node('M2', 'M2: Stance')
        c.node('M3', 'M3: Consistency')
    
    graph.node('Meta', 'Meta-Classifier\n(Level 2)', shape='hexagon', style='filled', fillcolor='gold')
    graph.node('Out', 'Output:\nReal vs Fake', shape='ellipse', style='filled', fillcolor='lightgreen')

    graph.edge('I', 'P')
    graph.edge('P', 'M1')
    graph.edge('P', 'M2')
    graph.edge('P', 'M3')
    graph.edge('M1', 'Meta')
    graph.edge('M2', 'Meta')
    graph.edge('M3', 'Meta')
    graph.edge('Meta', 'Out')

    st.graphviz_chart(graph)

st.success("👈 Select a tool from the sidebar to get started.")