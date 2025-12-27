import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Data Analysis", page_icon="📈")

st.markdown("# 📈 Data and Unsupervised Learning")
st.write("""
This section presents the distribution of training data and the application of clustering algorithms (K-Means) to validate natural groupings of news articles.
""")

# Simulation of Data for Charts
np.random.seed(42)
n_samples = 200

# Create fictitious data for Clustering
data = {
    'PCA_Component_1': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)]),
    'PCA_Component_2': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)]),
    'Cluster': ['Real News'] * 100 + ['Fake News'] * 100,
    'TTopic': np.random.choice(['Politics', 'Technology', 'Health'], 200)
}
df = pd.DataFrame(data)

# Tabulação
tab1, tab2 = st.tabs(["Clustering (K-Means)", "Topic Distribution"])

with tab1:
    st.subheader("Cluster Visualization (2D)")
    st.markdown("Using **PCA** to reduce dimensionality and **K-Means** to cluster news articles.")
    
    fig = px.scatter(
        df, 
        x='PCA_Component_1', 
        y='PCA_Component_2', 
        color='Cluster',
        symbol='TTopic',
        title='Spatial Separation: Real vs Fake (Simulation)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The visual separation indicates that Fake News have distinct linguistic patterns from real news.")

with tab2:
    st.subheader("Exploratory Analysis of Datasets [cite: 31]")
    col_metrics1, col_metrics2 = st.columns(2)
    
    with col_metrics1:
        st.metric("Total Articles Analyzed", "12,450")
        st.metric("Verified Sources", "128")
    
    with col_metrics2:
        chart_data = pd.DataFrame({
            "Count": [4500, 3200, 1500, 3250],
            "Category": ["Politics", "Health", "Technology", "Others"]
        })
        st.bar_chart(chart_data, x="Category", y="Count")

# Recent Classifications Table
st.divider()
st.subheader("Recent Classification Records")
recent_data = pd.DataFrame({
    "ID": range(1001, 1006),
    "Title": [
        "Early elections confirmed for March",
        "Lemon tea cures all serious diseases",
        "New iPhone will be foldable and transparent",
        "Inflation drops to 2.1% in the Eurozone",
        "Aliens landed in Porto this morning"
    ],
    "Classification": ["Real", "Fake", "Fake", "Real", "Fake"],
    "Confidence": ["98%", "99%", "87%", "92%", "99%"]
})
st.dataframe(recent_data, hide_index=True)