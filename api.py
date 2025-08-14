

import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_ngrok import run_with_ngrok
import speech_recognition as sr
from pydub import AudioSegment

# =====================
# CONFIG
# =====================
MODEL_PATH = '/content/drive/MyDrive/fashion_recommender_model.h5'

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# =====================
# FLASK APP
# =====================
app = Flask(__name__)
 # so Colab can host it

# Preprocessing function for images
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Dummy function to simulate fashion advice
def generate_style_suggestions(prediction):
    return [
        "Try pairing with neutral-colored shoes.",
        "Add a statement necklace for a bold look.",
        "Consider layering with a denim jacket."
    ]

# Convert audio to text
def audio_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    audio_path = "temp_audio.wav"

    # Convert to WAV if not already
    with open("temp_audio_input", "wb") as f:
        f.write(audio_bytes)
    try:
        sound = AudioSegment.from_file("temp_audio_input")
        sound.export(audio_path, format="wav")
    except:
        os.rename("temp_audio_input", audio_path)

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# =====================
# ROUTES
# =====================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fashion Recommender API is running with voice support."})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        image_file = request.files["image"]
        image = Image.open(io.BytesIO(image_file.read()))
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        suggestions = generate_style_suggestions(prediction)
        
        return jsonify({
            "status": "success",
            "input_type": "image",
            "predictions": prediction.tolist(),
            "style_suggestions": suggestions
        })
    
    elif "description" in request.form:
        description = request.form["description"]
        suggestions = [
            f"Based on '{description}', try mixing textures for added depth.",
            "Consider matching accessories to your outfit's accent color."
        ]
        return jsonify({"status": "success", "input_type": "text", "style_suggestions": suggestions})

    elif "voice" in request.files:
        voice_file = request.files["voice"].read()
        try:
            description = audio_to_text(voice_file)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

        suggestions = [
            f"Based on '{description}', try mixing textures for added depth.",
            "Consider matching accessories to your outfit's accent color."
        ]
        return jsonify({"status": "success", "input_type": "voice", "transcribed_text": description, "style_suggestions": suggestions})

    else:
        return jsonify({"status": "error", "message": "No image, text, or voice provided."})

# =====================
# START APP
# =====================
if __name__ == "__main__":
    app.run()
