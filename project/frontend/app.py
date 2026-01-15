import streamlit as st
import time
import pandas as pd
import random
from modules.news import get_article


st.set_page_config(page_title="Live Detection", page_icon="🔮")

st.markdown("# 🔮 Live Detection Interface")
st.markdown("Insert a news article below to submit to the classification models.")

# Input Area
news_text = st.text_input("News URL", 
                        placeholder="Paste the news URL here...")

# Action Button
if st.button("🔍 Analyze Veracity", type="primary"):
    if not news_text:
        st.warning("Please enter text to analyze.")
    else:
        # Try to fetch article
        try:
            article = get_article(news_text)
        except Exception as e:
            st.error(f"Erro ao obter artigo: {e}")
            article = None
        
        if article:
            st.divider()
            st.subheader("📚 Artigo obtido")
            st.markdown(f"**Título:** {article.get('title','(sem título)')}")
            st.text_area("Corpo da notícia", value=article.get('text',''), height=300)
            
        # PROCESSING SIMULATION (Loading)
        with st.status("Processing news...", expanded=True) as status:
            st.write("📝 Tokenization and Lemmatization...")
            time.sleep(1)
            st.write("🤖 Querying Model 1 (TTopics)...")
            time.sleep(0.8)
            st.write("⚖️ Verifying Stance (Headline vs Body)...")
            time.sleep(0.8)
            st.write("🧠 Aggregating results in Meta-Classifier...")
            time.sleep(0.5)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # RANDOM RESULT GENERATION (MOCKUP)
        # For demonstration, let's pretend the result depends on the text length
        is_fake = random.choice([True, False])
        fake_prob = random.uniform(0.75, 0.99) if is_fake else random.uniform(0.05, 0.30)
        
        # Results Layout
        st.divider()
        st.subheader("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)

        # Main Card
        with col1:
            if fake_prob > 0.5:
                st.error(f"## FAKE NEWS\nProbability: {fake_prob:.1%}")
            else:
                st.success(f"## REAL NEWS\nProbability: {(1-fake_prob):.1%}")

        # Intermediate Models Details
        with col2:
            st.markdown("**Detailed Analysis:**")
            st.progress(random.uniform(0.6, 0.9), text="Semantic Consistency")
            st.progress(random.uniform(0.4, 0.8), text="Headline-Body Agreement (Stance)")
            
        with col3:
            topic = random.choice(["Politics", "Economy", "Health", "Sports"])
            st.metric(label="Classified Topic", value=topic)
            st.metric(label="Clickbait Score", value=f"{random.randint(10, 90)}/100")

        # Explanation (Simulated)
        st.info(f"""
        **Model Justification:** The system detected {random.randint(2, 5)} semantic inconsistencies and 
        a significant divergence between the tone of the headline and the body content. 
        The topic was correctly identified as **{topic}**.
        """)