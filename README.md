 
 
# ⚖️ Legal Text Classification System (BERT + Flask)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/NLP-Transformers-orange?logo=huggingface)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered web application that classifies legal text into specific judicial usage categories. Using a fine-tuned Transformer model, the system analyzes the context of legal citations to determine if a case is being **Cited**, **Applied**, **Followed**, or **Referred To**.

---

## 🚀 Project Overview

Legal professionals often need to understand the "weight" of a precedent. This tool automates that process by analyzing the language surrounding legal citations. 

### ✨ Core Features
- 🧠 **NLP Engine:** Utilizes Hugging Face `transformers` for high-accuracy text classification.
- 📝 **Dual Input Methods:** Support for manual text snippets and bulk `.pdf` or `.txt` file uploads.
- 📊 **Confidence Scoring:** Provides a percentage-based reliability score for every prediction.
- 🧾 **Legal Dictionary:** Displays human-friendly meanings for legal labels.
- 🌐 **Remote Access:** Built-in `pyngrok` support for creating public URLs instantly.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** PyTorch, Hugging Face Transformers (BERT/Legal-BERT)
- **File Processing:** `pdfplumber` for high-fidelity PDF text extraction
- **Frontend:** HTML5, TailwindCSS (Responsive UI)
- **Deployment:** Ngrok (for local tunneling)

---

## 📁 Project Structure

```text
├── app.py              # Main Flask server & inference logic
├── app-model.py        # Local model loading variant
├── test_samples.py     # Script for automated API testing
├── TESTING_GUIDE.txt   # Step-by-step manual testing instructions
├── templates/
│   └── index.html      # Interactive web dashboard
└── static/             # (Optional) CSS/JS assets
```

 
## ⚙️ Installation & Setup

### 1. Clone & Environment
```bash
git clone https://github.com/your-username/NLP-PROJECT-TEAM-15.git
cd NLP-PROJECT-TEAM-15
python -m venv .venv
```

### 2. Activate Environment
- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 3. Install Dependencies
```bash
pip install flask torch pdfplumber pandas pyngrok transformers requests
```

### 4. Run the Application
```bash
python app.py
```
The app will be available at `http://localhost:5000`.

---

## 🔌 API Documentation

### `POST /predict`
Submit raw text for classification.
- **Payload:** `{"text": "The court cited Brown v. Board..."}`
- **Response:** Predicted label, confidence score, and category list.

### `POST /upload`
Submit a file for analysis.
- **Formats:** `.pdf`, `.txt`
- **Logic:** Extracts text, handles token truncation, and returns the primary classification.

---

## 🧪 Testing Inputs (Ready for Copy-Paste)

Use these snippets to verify the model's accuracy:

| Label | Sample Text |
| :--- | :--- |
| **CITED** | *In resolving the constitutional challenge, the appellate bench cited Brown v. Board of Education as the controlling authority on equal protection and also cited Roe v. Wade for the principle that privacy interests require strict judicial scrutiny.* |
| **APPLIED** | *The trial court applied the doctrine of res judicata because the same parties had previously litigated the same title dispute. The judge further applied Section 11 of the Code of Civil Procedure to bar re-litigation.* |
| **FOLLOWED** | *The division bench followed the precedent laid down in Miranda v. Arizona on procedural safeguards and followed domestic precedents requiring legal counsel access at the earliest stage of detention.* |
| **REFERRED TO** | *While deciding the labor dispute, the tribunal referred to International Labour Organization conventions and referred to policy papers published by the ministry to provide institutional context.* |

### 🧱 Stress Test (Long Context)
> "In a consolidated batch of appeals involving land acquisition, environmental clearance, and compensation parity, the court undertook a layered analysis. It cited constitutional guarantees of equality, applied statutory timelines for objections, followed binding rulings on public hearing requirements, and referred to global sustainability frameworks..."

---

## 📊 Label Meaning Guide

- 🟦 **CITED**: Case is mentioned as a general legal reference.
- 🟩 **APPLIED**: The legal principle is directly used to determine the outcome.
- 🟪 **FOLLOWED**: The court adopts the exact reasoning of a previous binding precedent.
- 🟨 **REFERRED TO**: Mentioned for contextual or background information only.

---

## 🤝 Acknowledgments
Developed as part of an NLP research project focused on legal domain adaptation and transformer-based classification. Special thanks to the **NLP-PROJECT-TEAM-15** contributors.
```

### How to use this:
1.  Open your project folder.
2.  Create a new file named `README.md` (or replace your current one).
3.  Paste the code above.
4.  **Optional:** In the "Clone" section, replace `your-username` with your actual GitHub username.
