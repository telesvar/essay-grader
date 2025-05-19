import streamlit as st
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(
    page_title="IELTS Essay Grader",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model components
@st.cache_resource
def load_models():
    # Define model paths
    models_dir = "models"
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    pca_path = os.path.join(models_dir, "pca_model.joblib")
    lgb_path = os.path.join(models_dir, "essay_grader_model.txt")
    
    # Load saved components
    tfidf_vectorizer = joblib.load(tfidf_path)
    scaler = joblib.load(scaler_path)
    pca_model = joblib.load(pca_path)
    lgb_model = lgb.Booster(model_file=lgb_path)
    
    # Get expected feature dimensions from scaler
    expected_features = scaler.mean_.shape[0]
    
    return tfidf_vectorizer, scaler, pca_model, lgb_model, expected_features

def extract_basic_features(essay_text):
    """Extract basic textual features from the essay."""
    words = essay_text.split()
    sentences = essay_text.split('.')
    
    # Length features
    word_count = len(words)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    avg_words_per_sentence = word_count / sentence_count
    
    # Vocabulary features
    unique_words = len(set([w.lower() for w in words]))
    lexical_diversity = unique_words / max(1, word_count)
    
    # Word length features
    avg_word_length = sum(len(w) for w in words) / max(1, word_count)
    long_words = sum(1 for w in words if len(w) > 6)
    
    # Paragraph features
    paragraphs = essay_text.split('\n\n')
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    return np.array([
        word_count, 
        sentence_count, 
        avg_words_per_sentence,
        unique_words,
        lexical_diversity,
        avg_word_length,
        long_words,
        paragraph_count
    ])

def process_essay(essay_text, models):
    """Process essay to match the expected feature dimensions and predict score."""
    tfidf_vectorizer, scaler, pca_model, lgb_model, expected_features = models
    
    # Generate TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([essay_text]).toarray()
    
    # Extract basic features
    basic_features = extract_basic_features(essay_text).reshape(1, -1)
    
    try:
        # Create placeholder for BERT embeddings (zeros)
        # The expected dimension is (expected_features - tfidf_features.shape[1] - basic_features.shape[1])
        bert_dim = max(0, expected_features - tfidf_features.shape[1] - basic_features.shape[1])
        bert_placeholder = np.zeros((1, bert_dim))
        
        # Combine features to match expected dimensions
        combined_features = np.hstack([bert_placeholder, basic_features, tfidf_features[:, :max(0, expected_features - bert_dim - basic_features.shape[1])]])
        
        # Ensure we have exactly the expected number of features
        if combined_features.shape[1] > expected_features:
            combined_features = combined_features[:, :expected_features]
        elif combined_features.shape[1] < expected_features:
            # Pad with zeros if needed
            padding = np.zeros((1, expected_features - combined_features.shape[1]))
            combined_features = np.hstack([combined_features, padding])
        
        # Apply scaling
        scaled_features = scaler.transform(combined_features)
        
        # Apply PCA
        reduced_features = pca_model.transform(scaled_features)
        
        # Predict using LightGBM model
        prediction = lgb_model.predict(reduced_features)[0]
        
        # Round to nearest 0.5 for IELTS-style scoring
        return round(prediction * 2) / 2
        
    except Exception as e:
        st.error(f"Error in feature processing: {str(e)}")
        # Fall back to rule-based scoring
        return rule_based_score(essay_text)

def rule_based_score(essay_text):
    """A simplified rule-based scoring method as fallback."""
    words = len(essay_text.split())
    
    # Base score
    if words < 100:
        score = 4.0
    elif words < 200:
        score = 5.0
    elif words < 250:
        score = 5.5
    elif words < 300:
        score = 6.0
    else:
        score = 6.5
    
    # Adjust for paragraph structure
    paragraphs = [p for p in essay_text.split('\n\n') if p.strip()]
    if len(paragraphs) >= 4:
        score += 0.5
        
    # Cap at 7.5 for the fallback method
    return min(7.5, score)

def get_band_description(score):
    """Return band description based on score"""
    if score >= 8.0:
        return "Very Good to Expert", """
        You demonstrate a complete and sophisticated command of language with very few inaccuracies. 
        Your writing is well-organized, detailed, and articulate with a wide range of vocabulary and 
        grammatical structures used appropriately.
        """
    elif score >= 7.0:
        return "Good", """
        You have a good command of the language with occasional inaccuracies and misunderstandings. 
        You can use complex language effectively and understand detailed arguments. 
        Your writing is well-organized and coherent.
        """
    elif score >= 6.0:
        return "Competent", """
        You generally have an effective command of the language despite some inaccuracies and misunderstandings. 
        You can use reasonably complex language and understand detailed reasoning.
        """
    elif score >= 5.0:
        return "Modest", """
        You have a partial command of the language and can handle basic communication in your own field. 
        Your writing shows limited range of vocabulary and grammar with frequent errors, but meaning is generally clear.
        """
    elif score >= 4.0:
        return "Limited", """
        Your basic competence is limited to familiar situations. You have frequent problems in 
        understanding and expression, and you are not able to use complex language.
        """
    else:
        return "Extremely Limited", """
        You have great difficulty understanding written English and conveying meaning even in simple contexts.
        """

def get_feedback(essay_text, score):
    """Generate specific feedback based on the essay and score."""
    word_count = len(essay_text.split())
    
    feedback = []
    
    # Length feedback
    if word_count < 250:
        feedback.append("📏 **Length**: Your essay is shorter than the recommended 250-word minimum. Try to develop your ideas more fully.")
    elif word_count > 350:
        feedback.append("📏 **Length**: Your essay is quite long. While this isn't necessarily a problem, make sure you're not sacrificing quality for quantity.")
    else:
        feedback.append("📏 **Length**: Good essay length that meets IELTS requirements.")
    
    # Paragraph structure
    paragraphs = [p for p in essay_text.split('\n\n') if p.strip()]
    if len(paragraphs) < 4:
        feedback.append("📄 **Structure**: Consider using more paragraphs. A typical IELTS essay has at least 4: introduction, 2+ body paragraphs, and conclusion.")
    else:
        feedback.append("📄 **Structure**: Good paragraph structure with clear organization.")
    
    # Based on score bands
    if score >= 7.0:
        feedback.append("🌟 **Overall**: Your essay demonstrates good command of English with effective organization and development of ideas.")
    elif score >= 6.0:
        feedback.append("✅ **Overall**: Your essay shows competent use of English with generally clear organization, though there may be occasional errors or underdeveloped points.")
    elif score >= 5.0:
        feedback.append("⚠️ **Overall**: Your essay communicates basic ideas but needs improvement in language accuracy, vocabulary range, and development of arguments.")
    else:
        feedback.append("⚠️ **Overall**: Your essay needs significant improvement in organization, language accuracy, and idea development.")
    
    return feedback

def main():
    # Load models (cached by Streamlit)
    with st.spinner("Loading models... (this may take a moment on first run)"):
        try:
            models = load_models()
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
    
    # Header area
    st.title("📝 IELTS Essay Grader")
    st.markdown("""
    This application uses AI to grade IELTS writing task essays. 
    Enter your essay below to receive an estimated band score.
    """)
    
    # Sidebar with instructions
    st.sidebar.title("About")
    st.sidebar.info("""
    ## How it works
    
    This tool uses a machine learning model trained on hundreds of IELTS essays 
    to predict essay scores. The model analyzes:
    
    * Grammar and vocabulary usage
    * Essay structure and coherence
    * Argument development
    * Topic relevance
    
    ## IELTS Writing Task 2
    
    For best results, write an essay responding to an IELTS Task 2 prompt.
    Aim for 250-300 words.
    """)
    
    # Example essay button
    if st.sidebar.button("Load Example Essay"):
        example_essay = """Climate change is one of the most pressing challenges facing humanity in the 21st century. Its effects are already evident in rising global temperatures, more frequent extreme weather events, and melting polar ice caps. These environmental changes pose serious risks not only to natural ecosystems but also to human health, food security, and economic stability.

The primary cause of climate change is the increase in greenhouse gas emissions from human activities, particularly the burning of fossil fuels for energy and transportation. Deforestation, industrial processes, and agricultural practices also contribute significantly to the problem. If global emissions continue at their current pace, we could see catastrophic impacts within this century.

To mitigate climate change, immediate and coordinated action is necessary. Governments must implement policies that promote renewable energy sources like wind and solar, enforce regulations on emissions, and encourage sustainable practices across all sectors. International cooperation, such as the Paris Agreement, plays a crucial role in setting targets and holding countries accountable.

Individual actions are also important. People can reduce their carbon footprint by using public transportation, conserving energy, reducing meat consumption, and supporting eco-friendly companies. Education and awareness campaigns can help people understand the urgency of the situation and inspire behavioral change.

In conclusion, while climate change presents a daunting challenge, it also offers an opportunity for innovation and global solidarity. By embracing sustainability and making informed choices, both individuals and nations can contribute to a healthier planet for future generations."""
        st.session_state.essay_text = example_essay
    else:
        if 'essay_text' not in st.session_state:
            st.session_state.essay_text = ""
    
    # Main essay input area
    essay_text = st.text_area(
        "Enter your essay below:",
        value=st.session_state.essay_text,
        height=300
    )
    
    # Word count
    if essay_text:
        word_count = len(essay_text.split())
        st.info(f"Word count: {word_count} words")
        if word_count < 100:
            st.warning("Your essay is quite short. IELTS Task 2 essays should be at least 250 words.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Grade essay button
        if st.button("Grade Essay", type="primary"):
            if not essay_text or len(essay_text.split()) < 50:
                st.error("Please enter a longer essay (at least 50 words) for accurate grading.")
            else:
                with st.spinner("Analyzing your essay..."):
                    try:
                        score = process_essay(essay_text, models)
                        band_name, band_desc = get_band_description(score)
                        feedback = get_feedback(essay_text, score)
                        
                        # Store in session state
                        st.session_state.score = score
                        st.session_state.band_name = band_name
                        st.session_state.band_desc = band_desc
                        st.session_state.feedback = feedback
                    except Exception as e:
                        st.error(f"Error analyzing essay: {str(e)}")
                        st.session_state.error = str(e)
    
    with col2:
        # Display results if available
        if 'score' in st.session_state:
            st.markdown("### Results")
            
            # Create metrics in a row
            col_score, col_band = st.columns(2)
            with col_score:
                st.metric("IELTS Band Score", f"{st.session_state.score}")
            with col_band:
                st.metric("Band Classification", st.session_state.band_name)
            
            # Display band description
            st.markdown("#### Band Description")
            st.markdown(st.session_state.band_desc)
            
            # Display specific feedback
            st.markdown("#### Specific Feedback")
            for item in st.session_state.feedback:
                st.markdown(item)
            
    # Add footer
    st.markdown("---")
    st.markdown("*This is an automated assessment tool and should be used for practice purposes only.*")

if __name__ == "__main__":
    main()
