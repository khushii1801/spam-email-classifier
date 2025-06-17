# 🛡️ AI Spam Guardian - Streamlit App

An intuitive and beginner-friendly Streamlit app for classifying emails as **SPAM** or **HAM** (legitimate), using a Logistic Regression model with TF-IDF vectorization.

---

## 🚀 Features

- Classifies email content using a trained ML model
- Clean, modern UI with example buttons
- Shows confidence score and analysis
- Automatically downloads model and vectorizer if not present

---

## 🧠 How It Works

- Preprocesses text (stopword removal, stemming, etc.)
- Uses TF-IDF features with Logistic Regression to predict
- If `model.pkl` or `vectorizer.pkl` not found, it downloads them from Google Drive

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/spam-classifier-app.git
cd spam-classifier-app
```

2. **Install dependencies**

We recommend using a virtual environment:

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run main.py
```

---

## 🔗 Model Download Info

The following files are large and not included in the GitHub repo:

| File | Description | Download |
|------|-------------|----------|
| `model.pkl` | Trained Logistic Regression model | [Google Drive Link](https://drive.google.com/file/d/YOUR_MODEL_FILE_ID/view?usp=sharing) |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer | [Google Drive Link](https://drive.google.com/file/d/YOUR_VECTORIZER_FILE_ID/view?usp=sharing) |

They will be automatically downloaded on first run using [`gdown`](https://github.com/wkentaro/gdown).

---

## 📂 File Structure

```
.
├── main.py
├── requirements.txt
├── README.md
```

---


## 📃 License

This project is licensed under the MIT License.
