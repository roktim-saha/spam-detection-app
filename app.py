import streamlit as st
import joblib
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------- Page config ----------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ---------- Load model & vectorizer ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_artifacts()

# ---------- Text cleaning (must match training preprocessing) ----------
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# ---------- UI ----------
st.title("📩 AI-Based Spam Email/SMS Detector")
st.write("Enter a message below and the model will predict whether it's **Spam** or **Ham (legitimate)**.")

st.markdown("---")

message = st.text_area("✍️ Enter your message here:", height=150, placeholder="e.g. Congratulations! You've won a free prize, click here to claim...")

col1, col2 = st.columns([1, 1])
with col1:
    check_btn = st.button("🔍 Check Message", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if check_btn:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        cleaned = clean_text(message)
        vec = tfidf.transform([cleaned])
        prediction = model.predict(vec)[0]

        # Probability, if the model supports it
        try:
            proba = model.predict_proba(vec)[0]
            spam_prob = proba[1] * 100
            ham_prob = proba[0] * 100
        except AttributeError:
            spam_prob = ham_prob = None

        st.markdown("---")
        if prediction == 1:
            st.error("🚨 This message is predicted as **SPAM**")
        else:
            st.success("✅ This message is predicted as **HAM (Not Spam)**")

        if spam_prob is not None:
            st.write("**Confidence:**")
            st.progress(spam_prob / 100)
            st.write(f"Spam probability: **{spam_prob:.2f}%**  |  Ham probability: **{ham_prob:.2f}%**")

st.markdown("---")
st.caption("Built for SE-334 Artificial Intelligence Lab | Daffodil International University | Roktim Saha (232-35-558)")
