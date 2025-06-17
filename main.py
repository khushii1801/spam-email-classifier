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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with neutral colors
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { 
        background-color: #f8f9fa;
        margin-bottom: 0 !important;
    }
    .main > div:last-child {
        display: none !important;
    }
    .main {
        padding: 0rem 1rem;
    }
    .main > div {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background: #f1f3f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #495057;
    }
    .spam-result {
        background: #fff5f5;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
    }
    .ham-result {
        background: #f0fdf4;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #059669;
        margin: 1rem 0;
    }
    h1 {
        color: #1a365d;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .contact-form {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid #e2e8f0;
    }
    .contact-form input,
    .contact-form textarea {
        width: 100%;
        padding: 10px;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        margin: 6px 0 12px 0;
        font-size: 14px;
    }
    .contact-form button[type=submit] {
        background: #1a365d;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        width: 100%;
        font-weight: bold;
        margin-top: 8px;
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

# Download model files if they don't exist
def download_model_files():
    model_url = 'https://drive.google.com/uc?id=14G5dD8-KxQY94bAVI1zWGyxyDQCfBpAo'
    vectorizer_url = 'https://drive.google.com/uc?id=17gpEgFMxPz0HLWFG0_O3F9Feju2UcODZ'
    
    if not os.path.exists('model.pkl'):
        gdown.download(model_url, 'model.pkl', quiet=False)
    if not os.path.exists('vectorizer.pkl'):
        gdown.download(vectorizer_url, 'vectorizer.pkl', quiet=False)

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        download_model_files() 
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

def main():
    # Header with neutral color scheme
    st.markdown("<h1>🛡️ AI Spam Guardian</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced Email Classification System</p>", unsafe_allow_html=True)
    
    # Load model (will download if needed)
    with st.spinner("🔍 Loading model..."):
        model, vectorizer = load_model()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class='metric-container'>
                <h3 style='margin: 0;'>97%</h3>
                <p style='margin: 0;'>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='metric-container'>
                <h3 style='margin: 0;'>0.97</h3>
                <p style='margin: 0;'>F1-Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Algorithm:** Logistic Regression")
        st.markdown("**Features:** TF-IDF Vectorization")
        st.markdown("**Preprocessing:** Stemming, Stop-word removal")
        
        # Contact Us section
        st.markdown("---")
        st.header("📬 Contact Us")
        contact_form = """
<div class="contact-form">
<form action="https://formsubmit.co/khushi23112004@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here" rows="4"></textarea>
     <button type="submit">Send Message</button>
</form>
</div>
"""
        st.markdown(contact_form, unsafe_allow_html=True)
    
    # Rest of your app code remains the same...
    # Demo examples, classification logic, etc.

if __name__ == "__main__":
    main()
