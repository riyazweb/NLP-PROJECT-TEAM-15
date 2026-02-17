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
MODEL_DIR = '/content/legal_case_model/content/legal_case_model'

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

CORE_LABELS = ["cited", "applied", "followed", "referred to"]

KEYWORD_PATTERNS = {
    "cited": [
        "cited", "citation", "citing", "relied on", "relied upon", "authority",
        "precedent cited", "binding guidance", "persuasive guidance"
    ],
    "applied": [
        "applied", "application", "apply", "doctrine", "section", "barred", "re litigation",
        "res judicata", "burden of proof", "limitation", "evidentiary", "on merits"
    ],
    "followed": [
        "followed", "following", "ratio decidendi", "binding precedent", "adherence",
        "legal certainty", "consistency", "larger bench", "same reasoning"
    ],
    "referred to": [
        "referred to", "referred", "reference", "background", "context", "policy papers",
        "committee reports", "conventions", "comparative jurisprudence", "not determinative"
    ],
}

def normalize_label(label):
    normalized = str(label or "").strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    aliases = {
        "referred": "referred to",
        "refer": "referred to",
        "referredto": "referred to",
        "referred to": "referred to",
        "referred to.": "referred to",
    }
    return aliases.get(normalized, normalized)

def select_core_prediction(predictions):
    if isinstance(predictions, dict):
        predictions = [predictions]

    best_core = None
    best_core_score = -1.0

    for item in predictions or []:
        raw = item.get("label", "")
        score = float(item.get("score", 0.0))
        canonical = normalize_label(raw)
        if canonical in CORE_LABELS and score > best_core_score:
            best_core = {
                "label": canonical,
                "score": score,
                "raw_label": str(raw)
            }
            best_core_score = score

    if best_core:
        return best_core

    top_item = (predictions or [{}])[0]
    raw = str(top_item.get("label", "unknown"))
    return {
        "label": normalize_label(raw),
        "score": float(top_item.get("score", 0.0)),
        "raw_label": raw
    }

def keyword_signal_scores(text):
    lowered = str(text or "").lower().replace("-", " ").replace("_", " ")
    scores = {label: 0.0 for label in CORE_LABELS}
    for label, patterns in KEYWORD_PATTERNS.items():
        for pattern in patterns:
            if pattern in lowered:
                scores[label] += 1.0

    total = sum(scores.values())
    if total > 0:
        return {label: value / total for label, value in scores.items()}
    return {label: 0.0 for label in CORE_LABELS}

def prediction_to_score_map(predictions):
    if isinstance(predictions, dict):
        predictions = [predictions]

    score_map = {label: 0.0 for label in CORE_LABELS}
    for item in predictions or []:
        canonical = normalize_label(item.get("label", ""))
        if canonical in score_map:
            score_map[canonical] = max(score_map[canonical], float(item.get("score", 0.0)))
    return score_map

def hybrid_core_prediction(text, model_predictions):
    model_scores = prediction_to_score_map(model_predictions)
    keyword_scores = keyword_signal_scores(text)

    alpha_model = 0.45
    alpha_keywords = 0.55

    combined = {
        label: (alpha_model * model_scores[label]) + (alpha_keywords * keyword_scores[label])
        for label in CORE_LABELS
    }

    best_label = max(combined, key=combined.get)
    best_score = float(combined[best_label])

    if best_score <= 0:
        selected = select_core_prediction(model_predictions)
        return selected

    return {
        "label": best_label,
        "score": best_score,
        "raw_label": best_label,
        "combined_scores": combined,
        "model_scores": model_scores,
        "keyword_scores": keyword_scores,
    }

LABELS = CORE_LABELS

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
        res = classifier(safe_text, top_k=None)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try shorter text.'}), 500

    selected = hybrid_core_prediction(text, res)
    label = selected['label']
    score = selected['score']
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
            res = classifier(safe_text, top_k=None)
        except Exception as e:
            print(f"Classification error: {e}")
            return jsonify({'error': 'Classification failed. Please try a smaller file.'}), 500

        selected = hybrid_core_prediction(extracted_text, res)
        label = selected['label']
        score = selected['score']
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
