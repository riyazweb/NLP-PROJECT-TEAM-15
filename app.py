import os
import torch
import pdfplumber
from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
from transformers import pipeline

# --- 1. SETTINGS & PATHS ---
NGROK_TOKEN = "2UUGMJW8gaZ7Ikrl53By3xYHdLs_6b3ipRxC3rEXwy7JgQv5Y"
UPLOAD_FOLDER = 'uploads'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 2. BACKEND LOGIC (FLASK + BERT) ---
app = Flask(__name__)

# Define clean, standardized labels
LABELS = ["cited", "applied", "followed", "referred to"]
print(f"📊 Using legal outcomes: {LABELS}")

# Load AI Model (BART-Large is used for high-accuracy zero-shot)
device = 0 if torch.cuda.is_available() else -1
print("🧠 Loading BERT semantic engine...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

@app.route('/')
def home():
    # This will look for templates/index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Limit text to prevent overload
    text = text[:4000]
    
    # BERT Classification
    res = classifier(text, LABELS, multi_label=False)
    
    # Clean and format label
    label = res['labels'][0].lower().strip()
    
    return jsonify({
        'label': label,
        'score': float(res['scores'][0])
    })

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    
    extracted_text = ""
    try:
        if file.filename.lower().endswith('.pdf'):
            with pdfplumber.open(path) as pdf:
                # Extracting first 3 pages to stay within BERT token limits
                for page in pdf.pages[:3]:
                    extracted_text += (page.extract_text() or "") + " "
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read()
        
        if not extracted_text.strip():
            return jsonify({'error': 'Could not extract text from file'}), 400

        # Run BERT on the first 4000 characters
        res = classifier(extracted_text[:4000], LABELS, multi_label=False)
        
        # Clean and format label
        label = res['labels'][0].lower().strip()
        
        return jsonify({
            'label': label,
            'score': float(res['scores'][0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 3. NGROK TUNNEL & EXECUTION ---
print("🚀 Starting Ngrok Tunnel...")
ngrok.kill()
ngrok.set_auth_token(NGROK_TOKEN)
public_url = ngrok.connect(5000).public_url

print(f"\n==================================================")
print(f"🔥 LEGAL AI LIVE AT: {public_url}")
print(f"==================================================\n")

if __name__ == '__main__':
    app.run(port=5000)