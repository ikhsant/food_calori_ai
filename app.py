from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import base64
import requests
import markdown
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained food detection model (replace with your food model if available)
model = YOLO("yolov8s.pt")  # Replace with yolov8s-food.pt if you have one

# Gemini API config
GEMINI_API_KEY = "AIzaSyDOSC1SkNFsKJg2oKsLh5X3AtUlJ5Zm0fY"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(path)

    # Detect objects
    results = model(path)
    names = model.names
    labels = []
    for result in results:
        boxes = result.boxes
        for cls in boxes.cls:
            label = names[int(cls)]
            if label not in labels:
                labels.append(label)

    # Prompt Gemini
    prompt = f"Saya makan: {', '.join(labels)}. Berapa kira-kira total kalorinya? Apakah sehat untuk dikonsumsi?"
    response = requests.post(GEMINI_API_URL, json={
        "contents": [{"parts": [{"text": prompt}]}]
    })

    if response.status_code == 200 and 'candidates' in response.json():
        result_raw = response.json()['candidates'][0]['content']['parts'][0]['text']
        result_text = markdown.markdown(result_raw)
    else:
        result_text = "Gagal mendapatkan respons dari Gemini."

    return render_template('result.html', image_path=path, labels=labels, advice=result_text)

if __name__ == '__main__':
    app.run(debug=True)
