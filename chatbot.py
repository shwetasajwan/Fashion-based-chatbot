from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# OpenAI API Key (Make sure you put your API key here)
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key

# Load SVM model and vectorizer
with open("svm_model.pkl", "rb") as model_file:
    svm_model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the Excel data (your outfits file)
uploaded_excel_path = "outfits.xlsx"  # Path to the outfits.xlsx file
data = pd.read_excel(uploaded_excel_path)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')

        if not user_input:
            return jsonify({'reply': "Please type something to start the conversation."})

        # Classify user intent using SVM
        user_vector = vectorizer.transform([user_input])
        intent = svm_model.predict(user_vector)[0]

        # Handle intents
        if intent == "greet":
            reply = "Hello! How can I assist you today?"
        elif intent == "product_query":
            # Suggest outfits based on occasion and weather
            reply = recommend_outfit(user_input)
        elif intent == "complaint":
            reply = "I'm sorry to hear that. Please provide more details about your issue."
        elif intent == "thanks":
            reply = "You're welcome! Let me know if there's anything else I can assist with."
        elif intent == "recommend":
            reply = "Sure! What occasion or weather are you dressing for?"
        elif intent == "problem":
            reply = "Could you please describe the issue you're facing in more detail?"
        else:
            # Fallback to OpenAI GPT for complex queries
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"User: {user_input}\nAssistant:",
                max_tokens=150,
                temperature=0.7,
            )
            reply = response['choices'][0]['text'].strip()

        return jsonify({'reply': reply})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'reply': "I'm sorry, something went wrong. Please try again later."})

def recommend_outfit(user_input):
    """
    Recommend outfit based on the occasion and weather mentioned by the user.
    """
    # Convert user input to lowercase for easier matching
    user_input = user_input.lower()

    # Match the user input with both the 'Occasion' and 'Weather' columns from the dataset
    matched_occasions = data[data['Occasion'].str.lower().apply(lambda x: any(word in x for word in user_input.split()))]
    matched_weather = matched_occasions[matched_occasions['Weather'].str.lower().apply(lambda x: any(word in x for word in user_input.split()))]

    # If a match is found, return the relevant outfit suggestion
    if not matched_weather.empty:
        outfit_suggestion = matched_weather.iloc[0]  # Get the first match
        reply = f"For a {outfit_suggestion['Occasion']} in {outfit_suggestion['Weather']} weather, here’s a suggestion:\n"
        reply += f"Top: {outfit_suggestion['Top']}\n"
        reply += f"Bottom: {outfit_suggestion['Bottom']}\n"
        reply += f"Shoes: {outfit_suggestion['Shoes']}\n"
        reply += f"Accessories: {outfit_suggestion['Accessories']}\n"
        reply += f"Outerwear: {outfit_suggestion['Outerwear']}\n"
    else:
        reply = "Sorry, I couldn't find a matching outfit for that occasion and weather."

    return reply

if __name__ == '__main__':
    app.run(debug=True)