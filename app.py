import streamlit.components.v1 as components

import requests
import streamlit as st
import joblib
from claude_explainer import generate_explanation
from googletrans import Translator
import re
import speech_recognition as sr
from gtts import gTTS
import os

# Title
st.title("📰 Fake News Detector + Claude Analysis")

# Language & Style Options
language = st.selectbox("🌐 Select Language", ["English", "Hindi", "Telugu", "Marathi", "Tamil", "Kannada", "Malayalam"])
style = st.selectbox("🧠 Choose Claude's Style", ["Friendly", "Formal", "Sarcastic", "Detailed"])

language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml"
}
lang_code = language_codes[language]

# Translation function
def translate_to_english(text):
    translator = Translator()
    detected = translator.detect(text)
    if detected.lang != 'en':
        translated = translator.translate(text, dest='en')
        return translated.text, detected.lang
    else:
        return text, 'en'

def translate_to_selected(text, lang_code):
    translator = Translator()
    return translator.translate(text, dest=lang_code).text

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    return text.lower()

# 🎤 Speech Input Option
if st.button("🎤 Speak Your News"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak your news clearly.")
        audio = recognizer.listen(source, timeout=5)
    try:
        spoken_text = recognizer.recognize_google(audio, language=lang_code)
        news_input = spoken_text
        st.success("Captured: " + news_input)
    except Exception as e:
        st.error(f"Sorry, couldn't understand. {str(e)}")
else:
    news_input = st.text_area("🗞 Paste your news article here")

# Prediction logic
def verify_with_newsapi(query):
    api_key = "bcee99d546804281bb20c4eb998af302"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    if len(articles) >= 3:
        return "REAL", articles[:3]
    else:
        return "UNKNOWN", []

show_ml_only = st.checkbox("Use only ML model (skip NewsAPI verification)")

if st.button("Detect"):
    if news_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        cleaned = clean_text(news_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        translator = Translator()

        # Check with NewsAPI or ML
        if show_ml_only:
            label = prediction
            if prediction == "REAL":
                message = "This news is REAL (predicted by ML Model)"
                color = "success"
            else:
                message = "This news is FAKE (predicted by ML Model)"
                color = "error"
        else:
            label, articles = verify_with_newsapi(news_input)
            if label == "REAL":
                message = "This news is REAL (verified from NewsAPI)"
                color = "success"
            elif label == "FAKE":
                message = "This news is FAKE (predicted by ML Model)"
                color = "error"
            else:
                message = "This news is UNKNOWN (not found in reliable sources)"
                color = "warning"

        # Translate message if needed
        final_message = translate_to_selected(message, lang_code) if lang_code != 'en' else message

        # Display translated prediction
        if color == "success":
            st.success(final_message)
        elif color == "error":
            st.error(final_message)
        else:
            st.warning(final_message)

        # Show related articles if real
        if label == "REAL" and not show_ml_only:
            st.markdown("### 🗞 Top Related Articles")
            for article in articles:
                st.markdown(f"- [{article['title']}]({article['url']})")

        # 🧠 Claude Explanation
        st.subheader("🤖 Claude's Explanation")
        translated_input, _ = translate_to_english(news_input)
        explanation_en = generate_explanation(translated_input, label)

        explanation_final = translate_to_selected(explanation_en, lang_code) if lang_code != 'en' else explanation_en
        st.session_state['explanation_final'] = explanation_final

        st.info(explanation_final)



if st.button("🔊 Speak Explanation"):
    if 'explanation_final' in st.session_state:
        explanation_final = st.session_state['explanation_final']
        escaped_text = explanation_final.replace("\n", " ").replace('"', '\\"')

        components.html(f"""
            <script>
                const text = "{escaped_text}";
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = "{lang_code}";
                utterance.rate = 1;

                const el = window.parent.document.querySelector('section.main');
                if (el) {{
                    el.style.backgroundColor = '#fffacc';
                }}

                utterance.onend = () => {{
                    if (el) {{
                        el.style.backgroundColor = '';
                    }}
                }}

                speechSynthesis.cancel();  // Stop if anything else is speaking
                speechSynthesis.speak(utterance);
            </script>
        """, height=0)
    else:
        st.warning("⚠️ Please detect news first before using voice explanation.")
