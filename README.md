# MailSentinel AI

**AI-Powered Email Spam and Phishing Detection System**

MailSentinel AI is an educational cybersecurity project that uses machine learning and heuristic analysis to classify emails as **legitimate**, **spam**, or **phishing**. All application logic lives in a single Python file (`app.py`).

---

## Features

| Feature | Description |
|---|---|
| **Email Analyzer** | Classify emails with ML confidence scores and risk levels |
| **URL Analyzer** | Static analysis of embedded URLs for suspicious patterns |
| **Heuristic Rules** | Urgency language, credential harvesting, reward patterns, spoofed senders |
| **Scan History** | SQLite-backed scan log with per-scan detail view |
| **Model Performance** | Real accuracy, precision, recall, F1, confusion matrix |
| **Dark Dashboard** | Professional responsive cybersecurity UI |

---

## Quick Start

### 1. Clone / download and enter the project directory

```bash
cd MailSentinel-AI
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env and set a strong SECRET_KEY
```

### 5. Run the application

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

The app starts in **heuristic mode** if no trained model exists. It fully works — just without ML classification.

---

## Enabling ML Classification

### Step 1 — Prepare a dataset

Create `data/dataset.csv` with two columns:

| label | text |
|---|---|
| legitimate | Hi, please find the attached report... |
| spam | Congratulations! You WON $1,000,000... |
| phishing | Click here to verify your account immediately |

**Accepted label values:** `legitimate` (also `ham`, `legit`, `safe`, `0`), `spam` (also `1`), `phishing` (also `phish`, `2`).

#### Recommended public datasets

| Dataset | Labels available | Notes |
|---|---|---|
| [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/) | ham / spam | Classic email corpus |
| [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/) | ham (from inbox) | Requires preprocessing |
| [Phishing Email Dataset (Kaggle)](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) | phishing / legitimate | Ready to use |
| [TREC 2007 Spam Track](https://trec.nist.gov/data/spam.html) | spam / ham | Large corpus |

Merge datasets, add the `label` and `text` columns, and save as `data/dataset.csv`.

### Step 2 — Train the model

**Option A — Web UI:**
Go to **Model Performance → Train Model Now**

**Option B — Command line:**
```bash
python app.py --train
# or specify a custom path:
python app.py --train path/to/my_dataset.csv
```

Training output is saved to:
- `models/classifier.joblib` — trained classifier
- `models/vectorizer.joblib` — TF-IDF vectorizer
- `models/metrics.json` — evaluation metrics

### Step 3 — Restart the app

```bash
python app.py
```

The sidebar now shows **ML Model Active**.

---

## Project Structure

```
MailSentinel-AI/
├── app.py              ← All application logic (Flask + ML + templates)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── .env.example        ← Environment variable template
├── data/
│   └── dataset.csv     ← Your training data (not committed)
├── models/
│   ├── classifier.joblib
│   ├── vectorizer.joblib
│   └── metrics.json
└── instance/
    └── mailsentinel.db ← SQLite database (auto-created)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | random (insecure) | Flask session secret — **set this in production** |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `PORT` | `5000` | Port to listen on |
| `MAX_CONTENT_LENGTH_MB` | `2` | Max request body size in MB |

---

## API Endpoints

### `POST /api/analyze`
Analyze an email via JSON.

**Request:**
```json
{
  "sender": "noreply@suspicious.xyz",
  "subject": "Your account has been compromised",
  "body": "Click here immediately to verify: http://paypal.secure-login.xyz/verify"
}
```

**Response:**
```json
{
  "label": "phishing",
  "confidence": 0.92,
  "risk_level": "critical",
  "warnings": ["Urgency / pressure language detected", "Credential harvesting language detected"],
  "ml_used": true,
  "scan_id": "...",
  "details": {
    "recommendation": "Do not click any links..."
  }
}
```

### `GET /api/stats`
Returns dashboard scan statistics.

---

## Running in VS Code

1. Open the `MailSentinel-AI` folder in VS Code.
2. Select the Python interpreter from your virtual environment (`Ctrl+Shift+P` → _Python: Select Interpreter_).
3. Open a terminal (`Ctrl+\``) and run:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
4. Click the URL in the terminal output or open http://127.0.0.1:5000.

**Recommended extensions:** Python, Pylance, SQLite Viewer.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| "Model Not Loaded" in sidebar | Dataset missing or not trained yet — see ML section above |
| `LookupError: NLTK stopwords not found` | Run `python -c "import nltk; nltk.download('stopwords')"` |
| Database locked error | Stop any other running instance; only one process per SQLite file |
| Port 5000 already in use | Set `PORT=5001` in `.env` or run `PORT=5001 python app.py` |
| Training fails with "Dataset too small" | Ensure dataset has ≥50 rows per class |

---

## Security Notes

- This application is designed for **local development and educational use only**.
- Do not expose it to the public internet without adding authentication and HTTPS.
- For production deployment, use **Gunicorn** behind **Nginx** with TLS:
  ```bash
  pip install gunicorn
  gunicorn -w 2 -b 0.0.0.0:8000 app:app
  ```
- Always set a strong `SECRET_KEY` in production.
- Email content is never stored; only metadata (sender, subject, label) is saved.

---

## License

Educational use only. Not intended for production security applications.
