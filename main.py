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
hide = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide, unsafe_allow_html=True)

# Custom CSS for professional styling
st.markdown("""
<style>
    html, body, .stApp {
        height: 100%;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        background-size: cover;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
    }

    .metric-container {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.08);
    }

    .spam-result, .ham-result {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
    }

    .spam-result {
        background: rgba(255, 0, 0, 0.08);
        border-left: 6px solid #dc2626;
    }

    .ham-result {
        background: rgba(0, 128, 0, 0.08);
        border-left: 6px solid #059669;
    }

    h1 {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.25rem;
        color: #dc143c;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .demo-button {
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.3);
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }

    .demo-button:hover {
        background: rgba(255,255,255,0.4);
    }

    .stTextArea > div > textarea {
        border-radius: 12px;
        padding: 10px;
        font-size: 15px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        background-color: #667eea !important;
        color: white !important;
    }

    .stButton > button:hover {
        background-color: #5a67d8 !important;
    }

    .contact-form {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }

    .contact-form input, .contact-form textarea {
        width: 100%;
        padding: 10px;
        border: 1px solid rgba(102, 126, 234, 0.5);
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.9);
        font-size: 14px;
        margin-bottom: 12px;
    }

    .contact-form button[type=submit] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
        transition: 0.3s ease;
    }

    .contact-form button[type=submit]:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)


# Text preprocessing function (matching your notebook exactly)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", ' link ', text)  # keep link token
    text = re.sub(r'\d+', ' number ', text)  # keep number token
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
    st.markdown("<h1>🛡️ AI Spam Guardian</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced Email Classification System</p>", unsafe_allow_html=True)
    
    # Load model
    model, vectorizer = load_model()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class='metric-container'>
                <h3 style='color: #667eea; margin: 0;'>97%</h3>
                <p style='margin: 0; color: #6b7280;'>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='metric-container'>
                <h3 style='color: #667eea; margin: 0;'>0.97</h3>
                <p style='margin: 0; color: #6b7280;'>F1-Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-container'>
            <h3 style='color: #667eea; margin: 0;'>19,364+</h3>
            <p style='margin: 0; color: #6b7280;'>Training Samples</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Algorithm:** Logistic Regression")
        st.markdown("**Features:** TF-IDF Vectorization")
        st.markdown("**Preprocessing:** Stemming, Stop-word removal")
        st.markdown("---")
       
        st.header("📬 Contact Us")
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
            Have questions or feedback?<br>
            We'd love to hear from you!
        </div>
        """, unsafe_allow_html=True)
        
        # Contact form in sidebar
        contact_form = """
<div class="contact-form">
<form action="https://formsubmit.co/khushi23112004@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send Message</button>
</form>
</div>
"""
        st.markdown(contact_form, unsafe_allow_html=True)
    
    # Demo examples
    st.subheader("🎯 Try These Examples")
    
    demo_emails = {
        "🚨 Spam Example": "CONGRATULATIONS! You've won $1,000,000! Click here immediately to claim your prize! Limited time offer! Act now or lose forever! Send your bank details to claim your cash reward NOW!",
        "✅ Legitimate Example": "Hi Sarah, hope you're doing well. Just wanted to check in and see how the project is going. Let me know if you need any help with the quarterly report. Best regards, Mike"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚨 Load Spam Example", use_container_width=True):
            st.session_state.email_text = demo_emails["🚨 Spam Example"]
    
    with col2:
        if st.button("✅ Load Ham Example", use_container_width=True):
            st.session_state.email_text = demo_emails["✅ Legitimate Example"]
    
    # Input section
    st.subheader("📧 Email Classification")
    
    # Text input
    email_text = st.text_area(
        "Enter email content to classify:",
        value=st.session_state.get('email_text', ''),
        height=200,
        placeholder="Paste your email content here...\n\nThe AI will analyze the text and classify it as SPAM or HAM (legitimate) with confidence scores.",
        help="Enter the full email content including subject line if available."
    )
    
    # Classification buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        classify_button = st.button("🧠 Classify Email", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.email_text = ""
            st.experimental_rerun()
    
    # Classification logic
    if classify_button and email_text.strip():
        with st.spinner("🔍 Analyzing email content..."):
            try:
                # Use trained model
                cleaned_text = clean_text(email_text)
                text_vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vectorized)[0]
                probability = model.predict_proba(text_vectorized)[0]
                
                is_spam = prediction == 1
                confidence = max(probability)
                
                # Display results
                st.subheader("🎯 Classification Results")
                
                if is_spam:
                    st.markdown(f"""
                    <div class='spam-result'>
                        <h2 style='color: #dc2626; margin: 0;'>🚨 SPAM DETECTED</h2>
                        <h3 style='color: #dc2626; margin: 0.5rem 0;'>Confidence: {confidence:.1%}</h3>
                        <p style='color: #6b7280; margin: 0;'>
                            This email contains characteristics commonly found in spam messages. 
                            Exercise caution and avoid clicking links or providing personal information.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for spam
                    st.progress(confidence, text=f"Spam Probability: {confidence:.1%}")
                    
                else:
                    st.markdown(f"""
                    <div class='ham-result'>
                        <h2 style='color: #059669; margin: 0;'>✅ LEGITIMATE EMAIL</h2>
                        <h3 style='color: #059669; margin: 0.5rem 0;'>Confidence: {confidence:.1%}</h3>
                        <p style='color: #6b7280; margin: 0;'>
                            This email appears to be legitimate based on its content and structure. 
                            It shows characteristics of normal communication.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for ham
                    st.progress(confidence, text=f"Legitimate Probability: {confidence:.1%}")
                
            except Exception as e:
                st.error(f"❌ An error occurred during classification: {str(e)}")
                st.info("Please check your input and try again.")
    
    elif classify_button and not email_text.strip():
        st.warning("⚠️ Please enter some email content to classify.")

if __name__ == "__main__":
    main()
