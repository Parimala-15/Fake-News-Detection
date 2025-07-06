import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from model_utils import load_model, predict_news
from preprocessing import preprocess_text
from analytics import PredictionTracker, display_analytics_dashboard

# Configure page
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("📰 Fake News Detection System")
st.markdown("""
This application uses machine learning to analyze news articles and headlines to determine 
if they are likely to be **fake** or **real**. Simply paste your text below and get instant results.
""")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.vectorizer = None

# Initialize session ID for tracking
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Initialize prediction tracker
if 'tracker' not in st.session_state:
    st.session_state.tracker = PredictionTracker()

# Load model on first run
@st.cache_resource
def initialize_model():
    """Load the trained model and vectorizer"""
    try:
        model, vectorizer = load_model()
        return model, vectorizer, True
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None, False

# Load model
if not st.session_state.model_loaded:
    with st.spinner("Loading AI model..."):
        model, vectorizer, success = initialize_model()
        if success:
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.model_loaded = True
            if os.path.exists("trained_model.pkl"):
                # Check if vectorizer is properly trained
                vocab_size = len(getattr(st.session_state.vectorizer, 'vocabulary_', {}))
                if vocab_size > 1000:
                    st.success("✅ Your trained model loaded successfully!")
                else:
                    st.error("❌ Your vectorizer file is corrupted or incomplete!")
                    st.markdown("**Issue:** Your vectorizer only has {} words instead of thousands.".format(vocab_size))
                    with st.expander("🔧 How to fix this"):
                        st.markdown("""
                        **Your vectorizer file is damaged. Please re-save it in Google Colab:**
                        
                        ```python
                        import pickle
                        
                        # Make sure vectorizer is properly trained
                        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
                        
                        # Re-save with compatibility
                        with open('trained_model.pkl', 'wb') as f:
                            pickle.dump(model, f, protocol=2)
                        with open('vectorizer.pkl', 'wb') as f:
                            pickle.dump(vectorizer, f, protocol=2)
                            
                        # Download files
                        from google.colab import files
                        files.download('trained_model.pkl')
                        files.download('vectorizer.pkl')
                        ```
                        
                        **Note:** Your vocabulary should have thousands of words, not just 2!
                        """)
                    st.info("Using improved placeholder model for now. Please fix and re-upload your vectorizer.")
            else:
                st.warning("⚠️ Using placeholder model. Upload your trained model files for better accuracy.")
                with st.expander("📥 How to upload your Google Colab model"):
                    st.markdown("""
                    **Quick Steps:**
                    1. In your Google Colab, save your model:
                    ```python
                    import pickle
                    # Save your trained model
                    with open('trained_model.pkl', 'wb') as f:
                        pickle.dump(your_model, f, protocol=2)
                    # Save your vectorizer  
                    with open('vectorizer.pkl', 'wb') as f:
                        pickle.dump(your_vectorizer, f, protocol=2)
                    # Download files
                    from google.colab import files
                    files.download('trained_model.pkl')
                    files.download('vectorizer.pkl')
                    ```
                    2. Upload both files to this project
                    3. Refresh the page - your model will load automatically!
                    """)
        else:
            st.error("❌ Failed to load the model. Please check if the model file exists.")
            st.stop()

# Navigation tabs
tab1, tab2 = st.tabs(["🔍 Analyze News", "📊 Analytics"])

with tab1:
    # Main interface
    st.markdown("### Enter News Article or Headline")

# Text input area
user_input = st.text_area(
    "Paste your news text here:",
    height=200,
    placeholder="Enter the news article or headline you want to analyze...",
    help="You can paste a full article or just a headline. The model will analyze the text and provide a prediction."
)

# Analysis section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_button = st.button(
        "🔍 Analyze News",
        type="primary",
        use_container_width=True,
        disabled=not user_input.strip()
    )

# Results section
if analyze_button and user_input.strip():
    with st.spinner("Analyzing text..."):
        try:
            # Add small delay for better UX
            time.sleep(0.5)
            
            # Get prediction
            prediction, confidence = predict_news(
                user_input, 
                st.session_state.model, 
                st.session_state.vectorizer
            )
            
            # Track the prediction
            model_type = "user_model" if os.path.exists("trained_model.pkl") else "placeholder_model"
            st.session_state.tracker.add_prediction(
                user_input, prediction, confidence, model_type
            )
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Create result columns
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                # Prediction result
                if prediction == "FAKE":
                    st.error("🚨 **FAKE NEWS DETECTED**")
                    st.markdown(
                        f"<div style='background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;'>"
                        f"<h4 style='color: #c62828; margin: 0;'>This appears to be fake news</h4>"
                        f"<p style='margin: 5px 0 0 0; color: #666;'>Please verify from reliable sources</p>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.success("✅ **LIKELY AUTHENTIC**")
                    st.markdown(
                        f"<div style='background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;'>"
                        f"<h4 style='color: #2e7d32; margin: 0;'>This appears to be authentic news</h4>"
                        f"<p style='margin: 5px 0 0 0; color: #666;'>Still recommend cross-checking sources</p>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
            
            with result_col2:
                # Confidence score
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence:.1%}",
                    help="Higher confidence indicates the model is more certain about its prediction"
                )
                
                # Confidence bar
                confidence_color = "#f44336" if prediction == "FAKE" else "#4caf50"
                st.markdown(
                    f"""
                    <div style='background-color: #f0f0f0; border-radius: 10px; padding: 3px;'>
                        <div style='background-color: {confidence_color}; width: {confidence*100}%; height: 20px; border-radius: 7px; transition: width 0.3s ease;'></div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Additional information
            st.markdown("---")
            st.markdown("### ℹ️ Important Notes")
            st.info("""
            **Disclaimer**: This tool is for educational and informational purposes only. 
            - Always verify news from multiple reliable sources
            - AI models can make mistakes - use critical thinking
            - Consider the source, date, and context of the news
            - Be aware of your own biases when interpreting results
            """)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.markdown("Please try again or contact support if the problem persists.")

with tab2:
    # Analytics dashboard
    display_analytics_dashboard(st.session_state.tracker)

# Sidebar with additional information
with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown("""
    This fake news detection system uses:
    - **Natural Language Processing** to analyze text patterns
    - **Machine Learning** trained on news datasets
    - **Statistical analysis** of linguistic features
    """)
    
    st.markdown("### How to Use")
    st.markdown("""
    1. **Paste** news text in the main area
    2. **Click** 'Analyze News' button
    3. **Review** the prediction and confidence
    4. **Verify** with additional sources
    """)
    
    st.markdown("### Tips for Best Results")
    st.markdown("""
    - Include complete sentences
    - Longer text generally provides better accuracy
    - Headlines alone may be less reliable
    - Check multiple sources regardless of prediction
    """)
    
    # Quick stats in sidebar
    recent_count = len(st.session_state.tracker.get_recent_predictions(7))
    if recent_count > 0:
        st.markdown("### Recent Activity")
        st.markdown(f"**{recent_count}** predictions this week")
        
        analytics = st.session_state.tracker.get_analytics_data(7)
        if analytics["total_predictions"] > 0:
            st.markdown(f"**{analytics['fake_percentage']:.0f}%** flagged as fake")
            st.markdown(f"**{analytics['avg_confidence']:.0%}** avg confidence")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "Built with Streamlit • Powered by Machine Learning<br>"
    "Remember: Always verify news from multiple reliable sources"
    "</div>", 
    unsafe_allow_html=True
)
