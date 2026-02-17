 
# ⚖️ Legal Text Classification System (BERT + Flask)

An AI-powered web app that classifies legal text into judicial usage categories such as **Cited**, **Applied**, **Followed**, and **Referred To**.  
It supports both **manual text input** and **PDF/TXT file upload** with confidence scoring.

---

## 🚀 Project Overview

This project uses a transformer model in a Flask backend to analyze legal language and predict how a case/document is being used in legal reasoning.

### ✨ Core Features
- 🧠 Legal text classification using a transformer pipeline
- 📝 Direct text input for quick testing
- 📄 PDF/TXT upload and automatic text extraction
- 📊 Confidence score + raw score output
- 🧾 Human-friendly label meaning display
- 🌐 Ngrok public URL support for remote demos

---

## 🧩 How It Works (Step-by-Step)

1. 📂 The server starts and loads legal labels from dataset (if available).
2. 🤖 The NLP model is loaded using Hugging Face `transformers`.
3. 👤 User sends input in one of two ways:
   - Text input (`/predict`)
   - File upload (`/upload`)
4. ✂️ Input is token-limited/truncated safely for model compatibility.
5. 🔍 Model predicts top category + confidence.
6. ✅ UI shows:
   - Predicted label
   - Confidence %
   - Meaning of label
   - Metadata (chars/tokens/categories)

---

## 🛠️ Tech Stack

- 🐍 Python
- 🌶️ Flask
- 🤗 Transformers (Hugging Face)
- 🔥 Torch (PyTorch)
- 📄 pdfplumber
- 📊 pandas
- 🌐 pyngrok
- 🎨 HTML + TailwindCSS frontend

---

## 📁 Project Structure

- [app.py](app.py) → Main Flask app (routes, model inference, ngrok)
- [app-model.py](app-model.py) → Variant using local trained model directory
- [test_samples.py](test_samples.py) → API test script for sample inputs
- [TESTING_GUIDE.txt](TESTING_GUIDE.txt) → Manual testing instructions
- [templates/index.html](templates/index.html) → Frontend UI

---

## ⚙️ Installation & Run

### 1) Create virtual environment (recommended)
```bash
python -m venv .venv
```

### 2) Activate environment
**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install flask torch pdfplumber pandas pyngrok transformers requests
```

### 4) Run app
```bash
python app.py
```

### 5) Open app
- Local: `http://localhost:5000`
- Public (if ngrok starts): URL shown in terminal

---

## 🔌 API Endpoints

### `GET /`
Returns web interface.

### `GET /samples`
Returns sample legal texts + category list.

### `POST /predict`
Classifies raw text.

**Request JSON**
```json
{
  "text": "Your legal paragraph here..."
}
```

**Response (example)**
```json
{
  "label": "cited",
  "confidence": "95.67%",
  "score": 0.9567,
  "all_categories": ["cited", "applied", "followed", "referred to"]
}
```

### `POST /upload`
Uploads `.pdf` or `.txt`, extracts text, then classifies.

---

## 🧪 Long Test Inputs (Copy-Paste Ready)

### 1) 📚 CITED-style
In resolving the constitutional challenge, the appellate bench cited Brown v. Board of Education as the controlling authority on equal protection and also cited Roe v. Wade for the principle that privacy interests require strict judicial scrutiny when state action intrudes on personal liberty. The court then cited Kesavananda Bharati to explain limits on amendment power, cited Maneka Gandhi to connect procedure with fairness, and cited multiple high-court rulings to show a consistent line of reasoning. Although factual settings differed, each cited case was treated as persuasive or binding guidance on how constitutional safeguards should be interpreted when the executive acts without transparent standards.

---

### 2) ⚙️ APPLIED-style
After reviewing the pleadings, the trial court applied the doctrine of res judicata because the same parties had previously litigated the same title dispute and obtained a final judgment on merits. The court further applied Section 11 of the Code of Civil Procedure to bar re-litigation, applied limitation principles to reject stale claims, and applied evidentiary rules to exclude uncertified copies that lacked foundational proof. In addition, the judge applied settled contract interpretation rules to read clauses harmoniously and applied the burden-of-proof framework to conclude that the plaintiff failed to establish fraud by clear and cogent evidence.

---

### 3) 🧭 FOLLOWED-style
The division bench followed the precedent laid down in Miranda v. Arizona on procedural safeguards during custodial questioning and followed domestic precedents requiring legal counsel access at the earliest stage of detention. It followed prior constitutional bench rulings on voluntariness of confessions, followed established criminal jurisprudence on benefit of doubt, and followed earlier interpretations of due process where police non-compliance undermined reliability of statements. Even though prosecution argued factual distinctions, the court followed the ratio decidendi of earlier judgments, emphasizing that consistency and legal certainty demand adherence to binding precedent unless referred to a larger bench.

---

### 4) 📌 REFERRED TO-style
While deciding the labor dispute, the tribunal referred to International Labour Organization conventions, referred to comparative jurisprudence from the UK and Canada, and referred to policy papers published by the ministry to provide institutional context. The judgment referred to prior committee reports on workplace safety, referred to economic surveys on informal employment, and referred to constitutional directives to frame broad principles of social justice. These materials were not treated as determinative holdings but were referred to as background sources that helped the court situate statutory interpretation within broader human-rights and welfare-oriented considerations.

---

### 5) 🧪 Mixed / Ambiguous
The court cited several constitutional authorities, applied statutory provisions on limitation and maintainability, followed binding precedent on natural justice, and referred to international standards on administrative fairness. It cited A.K. Kraipak and Maneka Gandhi, applied procedural safeguards to hearing rights, followed earlier rulings on reasoned orders, and referred to policy circulars on transparency. Because the agency failed to disclose adverse material, the court held the decision arbitrary. However, since delay and laches were significant, partial relief was granted. This case therefore combines citation, application, following, and references in a single narrative, making it useful for stress-testing confidence scores and model tie-breaking behavior.

---

### 6) 🧱 Very Long Stress Test
In a consolidated batch of appeals involving land acquisition, environmental clearance, compensation parity, and rehabilitation obligations, the court undertook a layered analysis. It cited constitutional guarantees of equality and life with dignity, cited environmental precedents on precautionary principle, cited administrative law rulings on reasoned decision-making, and cited compensation judgments dealing with solatium and interest. The bench applied statutory timelines for objections, applied evidentiary rules to expert reports, applied proportionality to balance development with ecological concerns, and applied rehabilitation policy clauses requiring livelihood restoration. It followed binding rulings on public hearing requirements, followed precedent on non-arbitrariness in state action, and followed jurisdictional limits for appellate interference with factual findings. The judgment also referred to global sustainability frameworks, referred to national action plans, and referred to committee recommendations on displacement impact assessment. On facts, the authority’s process was partly compliant but materially deficient in cumulative-impact study, meaningful consultation, and transparent disclosure. Relief was therefore structured: acquisition notifications were not quashed in full, but implementation was stayed until fresh hearing, revised impact analysis, and recalculated compensation were completed under court supervision.

---

## 📊 Label Meaning Guide

- 🟦 **CITED** → Case is mentioned as a legal reference.
- 🟩 **APPLIED** → Legal rule is directly applied to decide current case.
- 🟪 **FOLLOWED** → Court adopts same reasoning as earlier precedent.
- 🟨 **REFERRED TO** → Mentioned for context/background, not core rule.

---

## 🧯 Troubleshooting

- ❌ **No text provided** → Ensure request JSON has a non-empty `text`.
- ❌ **Prediction failed** → Try shorter input or check model load logs.
- ❌ **CSV load issue** → Verify dataset path/config in app settings.
- ❌ **PDF extraction empty** → Try cleaner PDF or TXT input.
- ⏳ **First run slow** → Model download/load can take time initially.

---

## 👨‍💻 Demo Script Option

Run:
```bash
python test_samples.py
```
This sends sample requests to the API and prints predicted labels + confidence.

---

## 📌 Notes

- Model token limits can truncate very long text.
- Confidence reflects model certainty, not guaranteed legal correctness.
- For production, remove hardcoded secrets/tokens and use environment variables.

---

## 🤝 Acknowledgment

Built for legal NLP experimentation and academic project demonstration with transformer-based classification workflows.
```
 
