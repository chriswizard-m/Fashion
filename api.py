import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import speech_recognition as sr
from pydub import AudioSegment
from openai import OpenAI
import os
from dotenv import load_dotenv
import openai

# Load variables from .env
load_dotenv()

# Get the key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")




# Init GPT client
client = OpenAI(api_key=openai.api_key)

# Load model
model = load_model("model/fashion_recommender_model.h5")

# Flask app
app = Flask(__name__)

# Class names for predictions
CLASS_NAMES = ["T-shirt", "Dress", "Shirt", "Jeans", "Shoes", "Hat"]  # Example

def predict_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return CLASS_NAMES[class_index]

def transcribe_mp3(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

def get_gpt_advice(user_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful fashion advisor."},
            {"role": "user", "content": user_text}
        ]
    )
    return response.choices[0].message.content

@app.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:
        file = request.files["file"]
        file_ext = file.filename.lower()
        
        if file_ext.endswith((".png", ".jpg", ".jpeg")):
            file.save("temp.jpg")
            prediction = predict_image("temp.jpg")
            advice = get_gpt_advice(f"Suggest a better outfit for someone wearing {prediction}")
            return jsonify({"type": "image", "prediction": prediction, "gpt_advice": advice})
        
        elif file_ext.endswith(".mp3"):
            file.save("temp.mp3")
            transcribed_text = transcribe_mp3("temp.mp3")
            advice = get_gpt_advice(transcribed_text)
            return jsonify({"type": "voice", "transcription": transcribed_text, "gpt_advice": advice})
    
    elif "text" in request.json:
        user_text = request.json["text"]
        advice = get_gpt_advice(user_text)
        return jsonify({"type": "text", "gpt_advice": advice})

    return jsonify({"error": "No valid input provided"}), 400

if __name__ == "__main__":
    app.run(debug=True)
