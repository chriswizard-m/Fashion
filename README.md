# Fashion AI API

A Flask API for:
- Clothing recognition from images
- Voice input (MP3) transcription
- GPT-powered fashion advice

## Endpoints

### POST /predict
Accepts:
- Image (PNG, JPG, JPEG) → Predicts clothing type + GPT advice
- MP3 → Transcribes speech + GPT advice
- JSON text → GPT advice

#### Example (Image)
```bash
curl -X POST -F "file=@tshirt.jpg" https://your-render-api-url/predict
