from pathlib import Path
import streamlit as st

# Base path for visualizations
VISUALIZATIONS_PATH = Path(__file__).resolve().parents[2] / "files" / "visualizations"


def initialize_page():
    """Sets up the Streamlit page configuration and header."""
    st.set_page_config(page_title="Model Visualizations", page_icon="📊", layout="wide")
    st.markdown("# 📊 Model Visualizations")
    st.markdown("Explore the performance metrics and visualizations of the trained models.")


def get_images_from_folder(folder_path: Path) -> list:
    """Returns a list of image paths from a folder."""
    if not folder_path.exists():
        return []
    return sorted([f for f in folder_path.glob("*.png")])


def display_images(images: list, columns: int = 2):
    """Displays images in a grid layout."""
    if not images:
        st.info("No visualizations available for this module.")
        return
    
    cols = st.columns(columns)
    for idx, img_path in enumerate(images):
        with cols[idx % columns]:
            st.image(str(img_path), caption=img_path.stem.replace("_", " ").title(), use_container_width=True)


def display_topic_classification():
    """Displays Topic Classification visualizations with sub-tabs for LSA and NMF."""
    st.header("🏷️ Topic Classification")
    st.markdown("""
    Topic classification uses unsupervised learning techniques to identify and categorize 
    the main topics present in news articles. This helps in understanding the thematic 
    distribution of content.
    """)
    
    base_path = VISUALIZATIONS_PATH / "topic_classification"
    
    # Main visualizations
    main_images = get_images_from_folder(base_path)
    if main_images:
        st.subheader("Overview")
        display_images(main_images)
    
    # Sub-tabs for LSA and NMF
    tab_lsa, tab_nmf = st.tabs(["LSA (Latent Semantic Analysis)", "NMF (Non-negative Matrix Factorization)"])
    
    with tab_lsa:
        st.markdown("**LSA** uses singular value decomposition to reduce dimensionality and identify latent topics.")
        lsa_images = get_images_from_folder(base_path / "LSA")
        display_images(lsa_images)
    
    with tab_nmf:
        st.markdown("**NMF** decomposes the document-term matrix into non-negative factors representing topics.")
        nmf_images = get_images_from_folder(base_path / "NMF")
        display_images(nmf_images)


def display_stance_detection():
    """Displays Stance Detection visualizations."""
    st.header("🎯 Stance Detection")
    st.markdown("""
    Stance detection identifies the position or attitude expressed in a news article's 
    headline relative to its body. This is crucial for detecting potential inconsistencies 
    that may indicate misleading content.
    """)
    
    images = get_images_from_folder(VISUALIZATIONS_PATH / "stance_detection")
    display_images(images)


def display_clickbait_detection():
    """Displays Clickbait Detection visualizations."""
    st.header("🎣 Clickbait Detection")
    st.markdown("""
    Clickbait detection identifies sensationalized or misleading headlines designed to 
    attract clicks rather than inform. High clickbait scores can be an indicator of 
    low-quality or potentially fake news.
    """)
    
    images = get_images_from_folder(VISUALIZATIONS_PATH / "clickbait_detection")
    display_images(images)


def display_anomaly_detection():
    """Displays Anomaly Detection visualizations."""
    st.header("🔍 Anomaly Detection")
    st.markdown("""
    Anomaly detection identifies unusual patterns in news articles that deviate from 
    normal content. These anomalies can indicate fabricated or manipulated content.
    """)
    
    base_path = VISUALIZATIONS_PATH / "anomaly_detection"
    
    # Main visualizations
    main_images = get_images_from_folder(base_path)
    if main_images:
        st.subheader("Overview")
        display_images(main_images)
    
    # Sub-folder for specific anomaly models
    anomaly_models_path = base_path / "Anomaly models"
    if anomaly_models_path.exists():
        st.subheader("Anomaly Models Details")
        model_images = get_images_from_folder(anomaly_models_path)
        display_images(model_images)


def display_fake_news_detection():
    """Displays Fake News Detection (Meta Model) visualizations."""
    st.header("🤖 Fake News Detection (Meta Model)")
    
    st.markdown("""
    <div style="background-color: #1e3a5f; padding: 15px; border-radius: 10px; border-left: 5px solid #4da6ff;">
        <h4 style="margin: 0; color: #4da6ff;">⭐ Meta Model Architecture</h4>
        <p style="margin-top: 10px;">
        The Fake News Detection module is a <strong>meta model</strong> that combines predictions 
        from all other specialized modules to make the final classification decision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### How it works:
    
    The meta model aggregates features and predictions from:
    - **Topic Classification** - Thematic context of the article
    - **Stance Detection** - Headline-body consistency
    - **Clickbait Detection** - Sensationalism score
    - **Anomaly Detection** - Content irregularities
    
    These combined signals enable more robust and accurate fake news detection than 
    any single model could achieve alone.
    """)
    
    st.divider()
    
    images = get_images_from_folder(VISUALIZATIONS_PATH / "fake_news_detection")
    display_images(images)


def main():
    """Main execution flow."""
    initialize_page()
    
    # Create tabs for each module
    tab_topic, tab_stance, tab_clickbait, tab_anomaly, tab_fake_news = st.tabs([
        "🏷️ Topic Classification",
        "🎯 Stance Detection", 
        "🎣 Clickbait Detection",
        "🔍 Anomaly Detection",
        "🤖 Fake News (Meta Model)"
    ])
    
    with tab_topic:
        display_topic_classification()
    
    with tab_stance:
        display_stance_detection()
    
    with tab_clickbait:
        display_clickbait_detection()
    
    with tab_anomaly:
        display_anomaly_detection()
    
    with tab_fake_news:
        display_fake_news_detection()


if __name__ == "__main__":
    main()
