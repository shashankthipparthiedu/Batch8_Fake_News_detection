from flask import Flask, request, render_template
from deep_translator import GoogleTranslator
import requests
import random
import os

app = Flask(__name__)

# Claude API configuration (recommended to use environment variable)
CLAUDE_API_KEY = "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"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-haiku-20240307"

# Prediction cache
prediction_cache = {}

def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# Call Claude API with user prompt
def generate_explanation(news_text, prediction):
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    prompt = f"""Here's a news article:\n\n{news_text}\n\nThe model predicted it as {prediction}.\nExplain briefly why it might be {prediction.lower()}."""

    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            claude_data = response.json()
            return claude_data["content"][0].get("text", "⚠️ Claude returned no explanation.")
        else:
            return f"⚠️ Claude API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"⚠️ Claude request failed: {e}"

# Simulated logic to mock fake news detection
def check_fact(news_statement):
    verdict = 1 if "modi" in news_statement.lower() else 0  # Replace with actual ML logic if needed
    explanation = generate_explanation(news_statement, output_label(verdict))
    return verdict, explanation

# Simulate ML models’ output
def get_simulated_predictions(ai_result, input_text):
    if input_text in prediction_cache:
        return prediction_cache[input_text]

    predictions = {
        "Logistic Regression": ai_result,
        "Decision Tree": ai_result,
        "Gradient Boosting": ai_result,
        "Random Forest": ai_result
    }

    if random.random() < 0.3:
        model_to_flip = random.choice(list(predictions.keys()))
        predictions[model_to_flip] = 1 - ai_result

    prediction_cache[input_text] = predictions
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    statement = ""
    selected_lang = "en"
    explanation = ""

    if request.method == 'POST':
        statement = request.form['news']
        selected_lang = request.form['language']

        # Translate input to English
        translated_input = GoogleTranslator(source=selected_lang, target="en").translate(statement) if selected_lang != "en" else statement

        # Get result and explanation from Claude
        result, explanation = check_fact(translated_input)

        # Simulated predictions for ML models
        simulated = get_simulated_predictions(result, translated_input)

        prediction = {}
        for model, pred in simulated.items():
            label_en = output_label(pred)
            translated_output = GoogleTranslator(source="en", target=selected_lang).translate(label_en) if selected_lang != "en" else label_en
            prediction[model] = translated_output

        # Translate explanation if needed
        if selected_lang != "en":
            explanation = GoogleTranslator(source="en", target=selected_lang).translate(explanation)

    return render_template('index.html', prediction=prediction, statement=statement, selected_lang=selected_lang, explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)
