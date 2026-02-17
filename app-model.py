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
MODEL_DIR = '/content/legal_case_model'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 2. OPTIONAL: LOAD DATASET LABELS FOR INFO ONLY ---
print("📂 Loading legal dataset labels (optional)...")
LABELS = []
try:
    candidate_columns = ["case_outcome", "label", "category", "class", "outcome", "target"]
    all_columns = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
    label_column = next((col for col in candidate_columns if col in all_columns), all_columns[-1])
    df_labels = pd.read_csv(CSV_PATH, usecols=[label_column], nrows=5000, engine="python", on_bad_lines="skip")
    labels_raw = df_labels[label_column].dropna().astype(str).unique().tolist()
    LABELS = sorted([label.strip().lower() for label in labels_raw if label.strip() and label.lower() != 'nan'])
    print(f"✅ Loaded label column: {label_column}")
    print(f"📊 CSV categories found: {len(LABELS)}")
except Exception as e:
    print(f"⚠️ Could not load CSV labels: {e}")

# --- 3. BACKEND LOGIC (FLASK + TRAINED MODEL) ---
app = Flask(__name__)

# Load trained model from MODEL_DIR
device = 0 if torch.cuda.is_available() else -1
print("🧠 Loading trained text-classification model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    device=device,
    truncation=True,
    max_length=256
)
print("✅ Trained model ready!")

MODEL_LABELS = []
try:
    model_id2label = getattr(classifier.model.config, "id2label", {})
    if isinstance(model_id2label, dict) and model_id2label:
        MODEL_LABELS = [str(model_id2label[k]).lower() for k in sorted(model_id2label.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
except Exception:
    MODEL_LABELS = []

if MODEL_LABELS:
    LABELS = MODEL_LABELS

def get_label_meaning(label):
    meanings = {
        'cited': 'This case is mentioned as a reference in another case.',
        'applied': 'This case rule is directly used to decide the current case.',
        'followed': 'The court follows the same reasoning used in this case.',
        'referred to': 'This case is mentioned for background context.'
    }
    return meanings.get(label, 'This label shows how this case is used in legal judgments.')

def safe_text_for_model(text, max_tokens=256):
    if not text:
        return ""
    encoded = classifier.tokenizer(text, truncation=True, max_length=max_tokens)
    return classifier.tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

def token_count(text, max_tokens=256):
    if not text:
        return 0
    encoded = classifier.tokenizer(text, truncation=True, max_length=max_tokens)
    return len(encoded.get("input_ids", []))

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
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    safe_text = safe_text_for_model(text, max_tokens=256)
    used_tokens = token_count(text, max_tokens=256)

    try:
        # Predict with trained model
        res = classifier(safe_text, top_k=1)[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try shorter text.'}), 500
    
    label = str(res.get('label', 'unknown')).lower().strip()
    score = float(res.get('score', 0.0))
    confidence = score * 100
    meaning = get_label_meaning(label)
    
    return jsonify({
        'label': label,
        'confidence': f"{confidence:.2f}%",
        'score': score,
        'all_categories': LABELS,
        'label_meaning': meaning,
        'result_summary': f"Prediction: {label.upper()} with {confidence:.2f}% confidence.",
        'result_details': {
            'input_type': 'text',
            'input_characters': len(text),
            'processed_tokens': used_tokens,
            'model_path': MODEL_DIR,
            'total_categories': len(LABELS)
        }
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
        used_tokens = token_count(extracted_text, max_tokens=256)

        try:
            # Predict with trained model
            res = classifier(safe_text, top_k=1)[0]
        except Exception as e:
            print(f"Classification error: {e}")
            return jsonify({'error': 'Classification failed. Please try a smaller file.'}), 500
        
        label = str(res.get('label', 'unknown')).lower().strip()
        score = float(res.get('score', 0.0))
        confidence = score * 100
        meaning = get_label_meaning(label)
        
        return jsonify({
            'label': label,
            'confidence': f"{confidence:.2f}%",
            'score': score,
            'text_preview': extracted_text[:200] + "...",  # Show first 200 chars
            'all_categories': LABELS,
            'label_meaning': meaning,
            'result_summary': f"Prediction: {label.upper()} with {confidence:.2f}% confidence.",
            'result_details': {
                'input_type': 'file',
                'file_name': file.filename,
                'input_characters': len(extracted_text),
                'processed_tokens': used_tokens,
                'model_path': MODEL_DIR,
                'total_categories': len(LABELS)
            }
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
