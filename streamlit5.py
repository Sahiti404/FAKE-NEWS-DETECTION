import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2


def custom_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())


@st.cache_data
def load_data():
    true_df = pd.read_csv("C:\\Users\\pisip\\Desktop\\3-1ps\\python\\fake news\\Newsdataset\\True.csv")
    fake_df = pd.read_csv("C:\\Users\\pisip\\Desktop\\3-1ps\\python\\fake news\\Newsdataset\\Fake.csv")

    true_df['label'] = 'TRUE'
    fake_df['label'] = 'FAKE'

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['content'] = df['title'] + " " + df['text']

    return df


def preprocess_data(df):
    X = df['content']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    selector = SelectKBest(chi2, k=3000)
    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = selector.transform(X_test_tfidf)

    return X_train_selected, X_test_selected, y_train, y_test, vectorizer, selector


def train_model(X_train_selected, y_train):
    model = LogisticRegression(max_iter=300, C=0.1)
    model.fit(X_train_selected, y_train)
    return model


def predict_fake_or_true_(title, text, model, vectorizer, selector):
    new_content = title + " " + text
    tfidf_vector = vectorizer.transform([new_content])
    selected = selector.transform(tfidf_vector)

    prediction = model.predict(selected)[0]
    prediction_probabilities = model.predict_proba(selected)
    fake_probability = prediction_probabilities[0][1] * 100  # as percentage

    return prediction, fake_probability


# --- Streamlit UI ---
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #0078D4;
        font-size: 36px;
        font-weight: bold;
    }
    .header {
        font-size: 18px;
        color: #4CAF50;
    }
    .input-box {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">Fake News Detection</div>', unsafe_allow_html=True)

st.markdown('<span class="header">Enter the title of the news article:</span>', unsafe_allow_html=True)
new_title = st.text_input('')

st.markdown('<span class="header">Enter the content of the news article:</span>', unsafe_allow_html=True)
new_text = st.text_area('')

# Load and preprocess data
df = load_data()
X_train_selected, X_test_selected, y_train, y_test, vectorizer, selector = preprocess_data(df)
model = train_model(X_train_selected, y_train)

# Predict
if new_title and new_text:
    predicted_label, fakeness_percentage = predict_fake_or_true_(new_title, new_text, model, vectorizer, selector)

    st.subheader("Prediction Results")
    st.write(f"📰 The article is classified as: {predicted_label}")

    if predicted_label == "FAKE":
        st.write(f"Fakeness Percentage: {fakeness_percentage:.2f}%")

    test_accuracy = accuracy_score(y_test, model.predict(X_test_selected))
    st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")
