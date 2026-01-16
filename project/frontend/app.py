from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.modules.news import get_article
from frontend.modules.inference import predict


def initialize_page():
    """Sets up the Streamlit page configuration and header."""
    st.set_page_config(page_title="Live Detection", page_icon="🔮")
    st.markdown("# Live Detection Interface")
    st.markdown("Insert a news article below to submit to the classification models.")


def handle_input():
    """Handles user input and returns the URL and button state."""
    news_url = st.text_input("News URL", placeholder="Paste the news URL here...")
    analyze_clicked = st.button("🔍 Analyze Veracity", type="primary")
    return news_url, analyze_clicked


def display_article(article):
    """Displays the fetched article details in the UI."""
    if not article:
        return
    st.divider()
    st.subheader("📚 Article Fetched")
    st.markdown(f"**Title:** {article.get('title', '(no title)')}")
    st.text_area("News Body", value=article.get('text', ''), height=200)


def run_pipeline(title, text):
    """Executes the prediction pipeline with a status spinner and error handling."""
    with st.status("Running classification pipeline...", expanded=True) as status:
        st.write("🚀 Initializing pipeline...")
        try:
            # Make a prediction
            df_result = predict(title, text)
            
            # Extract the first (and only) row of results
            result_row = df_result.iloc[0]
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            return result_row

        except Exception as e:
            status.update(label="Analysis Failed", state="error", expanded=False)
            st.error(f"Pipeline Error: {e}")
            st.stop()


def display_results(result_row):
    """Displays the analysis results in a three-column layout."""
    st.divider()
    st.subheader("📊 Analysis Results")


    # Main Veracity Card
    is_fake = result_row['final_prediction'] == "Fake"
    confidence = result_row['confidence']
    
    if is_fake:
        st.error(f"## FAKE NEWS\nConfidence: {confidence:.1%}")
    else:
        st.success(f"## REAL NEWS\nConfidence: {confidence:.1%}")


def main():
    """Main execution flow."""
    initialize_page()
    news_url, analyze_clicked = handle_input()

    if analyze_clicked:
        if not news_url:
            st.warning("Please enter a URL to analyze.")
            return

        # Fetch Article
        try:
            article = get_article(news_url)
        except Exception as e:
            st.error(f"Error fetching article: {e}")
            return

        if article:
            display_article(article)
            
            # Run Inference
            result_row = run_pipeline(article['title'], article['text'])
            
            # Show Results
            display_results(result_row)


if __name__ == "__main__":
    main()