import os
import torch
import pdfplumber
import pandas as pd
from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
from transformers import pipeline

# --- 1. SETTINGS & PATHS ---
NGROK_TOKEN = "2UUGMJW8gaZ7Ikrl53By3xYHdLs_6b3ipRxC3rEXwy7JgQv5Y"
UPLOAD_FOLDER = 'uploads'
CSV_PATH = '/content/NLP-PROJECT-TEAM-15/legal-text-classification-dataset/legal_text_classification.csv'  # Kaggle dataset (in same folder)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 2. LOAD DATASET FROM CSV ---
print("📂 Loading legal dataset from CSV...")
try:
    candidate_columns = ["case_outcome", "label", "category", "class", "outcome", "target"]
    all_columns = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()

    label_column = next((col for col in candidate_columns if col in all_columns), None)
    if label_column is None:
        label_column = all_columns[-1]

    # Read only label column for robustness and speed
    df_labels = pd.read_csv(
        CSV_PATH,
        usecols=[label_column],
        nrows=5000,
        engine="python",
        on_bad_lines="skip"
    )

    labels_raw = df_labels[label_column].dropna().astype(str).unique().tolist()
    LABELS = sorted([label.strip().lower() for label in labels_raw if label.strip() and label.lower() != 'nan'])

    print(f"✅ Loaded label column: {label_column}")
    print(f"📊 Number of categories: {len(LABELS)}")
    print(f"📊 Sample categories: {LABELS[:10]}")
except Exception as e:
    print(f"⚠️ Could not load CSV: {e}")
    print("🔄 Using default categories instead")
    LABELS = ["cited", "applied", "followed", "referred to"]

# --- 3. BACKEND LOGIC (FLASK + BERT) ---
app = Flask(__name__)

# Load BERT AI Model - This is the brain of our system
# It understands legal text and classifies it into categories
device = 0 if torch.cuda.is_available() else -1
print("🧠 Loading BERT AI model (this may take a minute)...")
classifier = pipeline(
    "zero-shot-classification", 
    model="facebook/bart-large-mnli", 
    device=device
)
print("✅ BERT model ready!")

def safe_text_for_model(text, max_tokens=256):
    if not text:
        return ""
    encoded = classifier.tokenizer(text, truncation=True, max_length=max_tokens)
    return classifier.tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

@app.route('/')
def home():
    # This will look for templates/index.html
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return ('', 204)

@app.route('/samples')
def samples():
    """Show sample legal texts for testing"""
    sample_texts = [
        {
            "title": "Sample 1: Case Citation",
            "text": "In the case of Smith v. Jones (2020), the court cited the precedent set in Brown v. Board of Education regarding constitutional rights and equal protection under the law."
        },
        {
            "title": "Sample 2: Legal Application",
            "text": "The court applied the doctrine of res judicata, determining that the matter had already been adjudicated in a prior proceeding and therefore could not be relitigated."
        },
        {
            "title": "Sample 3: Precedent Following",
            "text": "Following the established precedent in Miranda v. Arizona, the court ruled that the defendant's rights were violated when statements were obtained without proper warnings."
        },
        {
            "title": "Sample 4: Case Reference",
            "text": "The judgment referred to multiple international conventions and treaties, including the Geneva Convention and the Universal Declaration of Human Rights."
        }
    ]
    return jsonify({
        "message": "Here are sample legal texts you can use to test the system",
        "categories": LABELS,
        "samples": sample_texts
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    This function takes legal text and predicts what category it belongs to
    Example: If you send "The court cited the Brown case", it will identify it as "cited"
    """
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    safe_text = safe_text_for_model(text, max_tokens=256)

    try:
        # Use BERT AI to classify the legal text
        res = classifier(safe_text, LABELS, multi_label=False, truncation=True, max_length=512)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try shorter text.'}), 500
    
    # Get the predicted category (like "cited", "applied", etc.)
    label = res['labels'][0].lower().strip()
    confidence = float(res['scores'][0]) * 100  # Convert to percentage
    
    return jsonify({
        'label': label,
        'confidence': f"{confidence:.2f}%",
        'score': float(res['scores'][0]),
        'all_categories': LABELS
    })

@app.route('/upload', methods=['POST'])
def upload():
    """
    This function handles file uploads (PDF or text files)
    It extracts text from the file and then classifies it
    """
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Save the uploaded file temporarily
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    
    extracted_text = ""
    try:
        # If it's a PDF, extract text from it
        if file.filename.lower().endswith('.pdf'):
            with pdfplumber.open(path) as pdf:
                # Read first 3 pages only (to stay within BERT's limits)
                for page in pdf.pages[:3]:
                    extracted_text += (page.extract_text() or "") + " "
        else:
            # If it's a text file, just read it directly
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read()
        
        if not extracted_text.strip():
            return jsonify({'error': 'Could not extract text from file'}), 400

        safe_text = safe_text_for_model(extracted_text, max_tokens=256)

        try:
            # Now classify the extracted text using BERT
            res = classifier(safe_text, LABELS, multi_label=False, truncation=True, max_length=512)
        except Exception as e:
            print(f"Classification error: {e}")
            return jsonify({'error': 'Classification failed. Please try a smaller file.'}), 500
        
        # Get the predicted category
        label = res['labels'][0].lower().strip()
        confidence = float(res['scores'][0]) * 100
        
        return jsonify({
            'label': label,
            'confidence': f"{confidence:.2f}%",
            'score': float(res['scores'][0]),
            'text_preview': extracted_text[:200] + "...",  # Show first 200 chars
            'all_categories': LABELS
        })
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

# --- 4. START THE SERVER ---
# Ngrok creates a public URL so anyone can access your app on the internet
print("🚀 Starting Ngrok Tunnel...")
ngrok.kill()
ngrok.set_auth_token(NGROK_TOKEN)
public_url = ngrok.connect(5000).public_url

print(f"\n" + "="*60)
print(f"🔥 LEGAL AI CLASSIFICATION SYSTEM IS LIVE!")
print(f"📍 Public URL: {public_url}")
print(f"📊 Categories: {len(LABELS)}")
print(f"🧪 Test samples: {public_url}/samples")
print(f"="*60 + "\n")

if __name__ == '__main__':
    app.run(port=5000)
