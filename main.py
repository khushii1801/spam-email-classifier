import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import gdown

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner styling
st.markdown("""
<style>
    .stApp {
        background: #f0f2f6;
    }
    .main > div {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .spam-result {
        background: rgba(239, 68, 68, 0.1);
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
    }
    .ham-result {
        background: rgba(16, 185, 129, 0.1);
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 4px solid #059669;
        margin: 1rem 0;
    }
    h1 {
        color: #333;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", ' link ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model_id = "14G5dD8-KxQY94bAVI1zWGyxyDQCfBpAo"
    vectorizer_id = "17gpEgFMxPz0HLWFG0_O3F9Feju2UcODZ"

    if not os.path.exists("model.pkl"):
        gdown.download(id=model_id, output="model.pkl", quiet=False)
    if not os.path.exists("vectorizer.pkl"):
        gdown.download(id=vectorizer_id, output="vectorizer.pkl", quiet=False)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# Main app
def main():
    # Header
    st.markdown("<h1>AI Spam Guardian</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced Email Classification System</p>", unsafe_allow_html=True)
    
    # Load model
    model, vectorizer = load_model()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("Model Performance")
        
        st.markdown("""
        <div class='metric-container'>
            <h3 style='color: #333; margin: 0;'>97%</h3>
            <p style='margin: 0; color: #666;'>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown("""
        <div class='metric-container'>
            <h3 style='color: #333; margin: 0;'>0.97</h3>
            <p style='margin: 0; color: #666;'>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-container'>
            <h3 style='color: #333; margin: 0;'>19,364+</h3>
            <p style='margin: 0; color: #666;'>Training Samples</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Algorithm:** Logistic Regression")
        st.markdown("**Features:** TF-IDF Vectorization")
        st.markdown("**Preprocessing:** Stemming, Stop-word removal")
    
    # Demo examples
    st.subheader("Try These Examples")
    
    demo_emails = {
        "Spam Example": "CONGRATULATIONS! You've won $1,000,000! Click here immediately to claim your prize! Limited time offer! Act now or lose forever! Send your bank details to claim your cash reward NOW!",
        "Legitimate Example": "Hi Sarah, hope you're doing well. Just wanted to check in and see how the project is going. Let me know if you need any help with the quarterly report. Best regards, Mike"
    }
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Spam Example", use_container_width=True):
            st.session_state.email_text = demo_emails["Spam Example"]
    with col2:
        if st.button("Load Ham Example", use_container_width=True):
            st.session_state.email_text = demo_emails["Legitimate Example"]
    
    # Input section
    st.subheader("Email Classification")
    email_text = st.text_area(
        "Enter email content to classify:",
        value=st.session_state.get('email_text', ''),
        height=200,
        placeholder="Paste your email content here..."
    )
    
    # Classification buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        classify_button = st.button("Classify Email", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.email_text = ""
            st.experimental_rerun()
    
    # Classification logic
    if classify_button and email_text.strip():
        with st.spinner("Analyzing email content..."):
            try:
                cleaned_text = clean_text(email_text)
                text_vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vectorized)[0]
                probability = model.predict_proba(text_vectorized)[0]
                
                is_spam = prediction == 1
                confidence = max(probability)
                
                st.subheader("Classification Results")
                
                if is_spam:
                    st.markdown(f"""
                    <div class='spam-result'>
                        <h2 style='color: #dc2626; margin: 0;'>SPAM DETECTED</h2>
                        <h3 style='color: #dc2626; margin: 0.5rem 0;'>Confidence: {confidence:.1%}</h3>
                        <p style='color: #666; margin: 0;'>
                            This email contains characteristics commonly found in spam messages.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(confidence, text=f"Spam Probability: {confidence:.1%}")
                else:
                    st.markdown(f"""
                    <div class='ham-result'>
                        <h2 style='color: #059669; margin: 0;'>LEGITIMATE EMAIL</h2>
                        <h3 style='color: #059669; margin: 0.5rem 0;'>Confidence: {confidence:.1%}</h3>
                        <p style='color: #666; margin: 0;'>
                            This email appears to be legitimate.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(confidence, text=f"Legitimate Probability: {confidence:.1%}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
