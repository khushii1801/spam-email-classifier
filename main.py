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

# Custom CSS with purple border and white content area
hide = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    .main > div {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 0 auto;
        max-width: 1200px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .stSidebar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    .sidebar-content {
        background: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
</style>
"""
st.markdown(hide, unsafe_allow_html=True)

# Rest of your CSS (keep your original styles)
st.markdown("""
<style>
    .metric-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #6c757d;
    }
    .spam-result {
        background: #ffecec;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .ham-result {
        background: #e8f5e9;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    h1 {
        color: #343a40;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Keep all your other original CSS styles below */
    .demo-button {
        background: #e9ecef;
        border: 1px solid #dee2e6;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
    }
    .contact-form {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid #dee2e6;
    }
    .contact-form input,
    .contact-form textarea {
        width: 100%;
        padding: 10px;
        border: 1px solid #ced4da;
        border-radius: 6px;
        margin: 6px 0 12px 0;
        font-size: 14px;
    }
    .contact-form button[type=submit] {
        background: #343a40;
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

# [KEEP ALL YOUR ORIGINAL FUNCTIONS EXACTLY THE SAME]
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
    model_url = 'https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID'
    vectorizer_url = 'https://drive.google.com/uc?id=YOUR_VECTORIZER_FILE_ID'
    
    if not os.path.exists('model.pkl'):
        gdown.download(model_url, 'model.pkl', quiet=False)
    if not os.path.exists('vectorizer.pkl'):
        gdown.download(vectorizer_url, 'vectorizer.pkl', quiet=False)

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        download_model_files()  # Download files if needed
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

# [KEEP YOUR MAIN FUNCTION EXACTLY THE SAME]
def main():
    # Header with clean styling
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
    
    # [KEEP ALL YOUR ORIGINAL MAIN CONTENT EXACTLY THE SAME]
    # Demo examples, classification logic, etc.

if __name__ == "__main__":
    main()
