"""
MailSentinel AI — AI-Powered Email Spam and Phishing Detection System
All application logic is consolidated in this single file.
"""

import os
import re
import sys
import json
import math
import time
import uuid
import sqlite3
import logging
import hashlib
import datetime
import urllib.parse
from pathlib import Path
import jinja2

# ---------------------------------------------------------------------------
# Optional imports with graceful degradation
# ---------------------------------------------------------------------------
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    # Download required NLTK data silently
    for _pkg in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
        except LookupError:
            try:
                nltk.download(_pkg, quiet=True)
            except Exception:
                pass
except ImportError:
    NLTK_AVAILABLE = False

from flask import (
    Flask, request, jsonify, session,
    render_template, render_template_string, redirect, url_for, g,
)
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INSTANCE_DIR = BASE_DIR / "instance"

for _d in (DATA_DIR, MODELS_DIR, INSTANCE_DIR):
    _d.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
DB_PATH = INSTANCE_DIR / "mailsentinel.db"

SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(32).hex())
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH_MB", 2)) * 1024 * 1024
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(str(DB_PATH))
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
    return db


@app.teardown_appcontext
def close_db(exc=None):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS scans (
                id          TEXT PRIMARY KEY,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sender      TEXT,
                subject     TEXT,
                label       TEXT NOT NULL,
                confidence  REAL,
                risk_level  TEXT,
                url_count   INTEGER DEFAULT 0,
                suspicious_url_count INTEGER DEFAULT 0,
                warnings    TEXT,
                details     TEXT
            );
        """)
        db.commit()


def insert_scan(sender, subject, label, confidence, risk_level,
                url_count, suspicious_url_count, warnings, details):
    scan_id = str(uuid.uuid4())
    db = get_db()
    db.execute(
        """INSERT INTO scans
           (id, sender, subject, label, confidence, risk_level,
            url_count, suspicious_url_count, warnings, details)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            scan_id,
            _truncate(sender, 200),
            _truncate(subject, 500),
            label,
            confidence,
            risk_level,
            url_count,
            suspicious_url_count,
            json.dumps(warnings),
            json.dumps(details),
        ),
    )
    db.commit()
    return scan_id


def get_scan_history(limit=50):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_scan_by_id(scan_id):
    db = get_db()
    row = db.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return dict(row) if row else None


def delete_all_scans():
    db = get_db()
    db.execute("DELETE FROM scans")
    db.commit()


def get_dashboard_stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    spam = db.execute("SELECT COUNT(*) FROM scans WHERE label='spam'").fetchone()[0]
    phishing = db.execute("SELECT COUNT(*) FROM scans WHERE label='phishing'").fetchone()[0]
    legit = db.execute("SELECT COUNT(*) FROM scans WHERE label='legitimate'").fetchone()[0]
    recent = db.execute(
        "SELECT * FROM scans ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    return {
        "total": total,
        "spam": spam,
        "phishing": phishing,
        "legitimate": legit,
        "recent": [dict(r) for r in recent],
    }


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def _truncate(text, max_len):
    if text and len(text) > max_len:
        return text[:max_len]
    return text or ""


def preprocess_text(text):
    """Lowercase, remove HTML tags, punctuation, digits; stem and remove stopwords."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Lowercase
    text = text.lower()
    # Remove URLs (keep placeholder)
    text = re.sub(r"https?://\S+", " url ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " email ", text)
    # Remove digits
    text = re.sub(r"\d+", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Tokenize
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()
    # Remove stopwords and short tokens
    if NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words("english"))
        except Exception:
            stop_words = set()
    else:
        stop_words = {
            "the","a","an","and","or","but","in","on","at","to","for",
            "of","with","by","from","is","was","are","were","be","been",
            "this","that","it","its","i","you","he","she","we","they",
        }
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # Stemming
    if NLTK_AVAILABLE:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Machine-learning model management
# ---------------------------------------------------------------------------

# Global in-memory model state
_model_state = {
    "pipeline": None,      # sklearn Pipeline(vectorizer + classifier)
    "metrics": None,       # dict with accuracy/precision/recall/f1/confusion_matrix
    "loaded": False,
    "error": None,
}


def load_model():
    """Try to load the saved model pipeline and metrics from disk."""
    global _model_state
    if not JOBLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        _model_state["error"] = "scikit-learn or joblib not installed."
        return False
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        try:
            vectorizer = joblib.load(str(VECTORIZER_PATH))
            classifier = joblib.load(str(MODEL_PATH))
            _model_state["pipeline"] = (vectorizer, classifier)
            _model_state["loaded"] = True
            _model_state["error"] = None
            logger.info("Model loaded from disk.")
        except Exception as e:
            _model_state["error"] = f"Failed to load model: {e}"
            _model_state["loaded"] = False
            return False
    else:
        _model_state["error"] = "Model files not found. Please run training first."
        _model_state["loaded"] = False
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                _model_state["metrics"] = json.load(f)
        except Exception:
            _model_state["metrics"] = None
    return _model_state["loaded"]


def train_model(dataset_path=None):
    """
    Train the classification model.
    Expects a CSV at dataset_path with columns: label, text
    Labels must be one of: legitimate, spam, phishing
    Returns (success: bool, message: str)
    """
    if not SKLEARN_AVAILABLE or not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        return False, "scikit-learn, pandas, and numpy are required for training."
    if not JOBLIB_AVAILABLE:
        return False, "joblib is required for saving the model."

    path = Path(dataset_path) if dataset_path else DATA_DIR / "dataset.csv"
    if not path.exists():
        return False, (
            f"Dataset not found at {path}. "
            "See README.md for dataset preparation instructions."
        )

    try:
        df = pd.read_csv(str(path))
    except Exception as e:
        return False, f"Failed to read dataset: {e}"

    # Flexible column detection
    label_col = None
    text_col = None
    for c in df.columns:
        if c.lower() in ("label", "class", "category", "type"):
            label_col = c
        if c.lower() in ("text", "body", "content", "message", "email"):
            text_col = c

    if label_col is None or text_col is None:
        # Try positional: first col = label, rest = text
        if len(df.columns) >= 2:
            label_col = df.columns[0]
            text_col = df.columns[1]
        else:
            return False, "Dataset must have at least two columns: label and text."

    df = df[[label_col, text_col]].dropna()
    df.columns = ["label", "text"]

    # Normalise labels
    label_map = {}
    for raw in df["label"].unique():
        low = str(raw).lower().strip()
        if low in ("ham", "legitimate", "legit", "safe", "0"):
            label_map[raw] = "legitimate"
        elif low in ("spam", "1"):
            label_map[raw] = "spam"
        elif low in ("phishing", "phish", "2"):
            label_map[raw] = "phishing"
    if not label_map:
        return False, "Could not map dataset labels to legitimate/spam/phishing."

    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])

    if len(df) < 50:
        return False, f"Dataset too small ({len(df)} rows). Need at least 50 samples."

    df["processed"] = df["text"].apply(preprocess_text)

    X = df["processed"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    classifier = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier.fit(X_train_vec, y_train)

    y_pred = classifier.predict(X_test_vec)
    labels = sorted(set(y))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0, labels=labels)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0, labels=labels)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0, labels=labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "confusion_labels": labels,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "trained_at": datetime.datetime.utcnow().isoformat(),
    }

    # Save
    joblib.dump(vectorizer, str(VECTORIZER_PATH))
    joblib.dump(classifier, str(MODEL_PATH))
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    _model_state["pipeline"] = (vectorizer, classifier)
    _model_state["metrics"] = metrics
    _model_state["loaded"] = True
    _model_state["error"] = None

    logger.info("Model trained and saved. Accuracy=%.4f", acc)
    return True, f"Training complete. Accuracy: {acc:.2%} on {len(X_test)} test samples."


def predict_email(subject, sender, body):
    """
    Classify an email. Returns dict with label, confidence, risk_level, warnings, details.
    Falls back to rule-based heuristics if the ML model is not loaded.
    """
    combined_text = f"{subject or ''} {sender or ''} {body or ''}"

    # Always run heuristic warnings (independent of ML model)
    warnings = _heuristic_warnings(subject or "", sender or "", body or "")
    url_results = analyze_urls(combined_text)

    if _model_state["loaded"] and _model_state["pipeline"]:
        vectorizer, classifier = _model_state["pipeline"]
        processed = preprocess_text(combined_text)
        try:
            vec = vectorizer.transform([processed])
            label = classifier.predict(vec)[0]
            proba = classifier.predict_proba(vec)[0]
            classes = list(classifier.classes_)
            idx = classes.index(label)
            confidence = float(proba[idx])
        except Exception as e:
            logger.warning("Prediction error: %s", e)
            label, confidence = _rule_based_label(warnings, url_results)
    else:
        label, confidence = _rule_based_label(warnings, url_results)

    # Boost label toward phishing if strong URL signals present
    if url_results["suspicious_count"] >= 2 and label == "legitimate":
        label = "phishing"
        confidence = max(confidence, 0.65)
    elif url_results["suspicious_count"] >= 1 and label == "legitimate" and len(warnings) >= 2:
        label = "spam"
        confidence = max(confidence, 0.55)

    risk_level = _compute_risk_level(label, confidence, warnings, url_results)
    details = _build_details(label, confidence, warnings, url_results)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "risk_level": risk_level,
        "warnings": warnings,
        "url_results": url_results,
        "details": details,
        "ml_used": _model_state["loaded"],
    }


def _rule_based_label(warnings, url_results):
    score = len(warnings) + url_results["suspicious_count"] * 2
    if score >= 5:
        return "phishing", 0.70
    if score >= 2:
        return "spam", 0.60
    return "legitimate", 0.85


def _compute_risk_level(label, confidence, warnings, url_results):
    if label == "phishing":
        return "critical"
    if label == "spam":
        return "high" if (len(warnings) >= 3 or url_results["suspicious_count"] >= 1) else "medium"
    if label == "legitimate":
        if len(warnings) >= 2:
            return "low"
        return "safe"
    return "unknown"


def _build_details(label, confidence, warnings, url_results):
    return {
        "classification": label,
        "confidence_pct": f"{confidence:.0%}",
        "warning_count": len(warnings),
        "urls_found": url_results["total_count"],
        "suspicious_urls": url_results["suspicious_count"],
        "recommendation": _recommendation(label, url_results),
    }


def _recommendation(label, url_results):
    if label == "phishing":
        return (
            "Do not click any links or download attachments. "
            "Report this email to your IT/security team immediately."
        )
    if label == "spam":
        return (
            "Exercise caution. Do not provide personal information. "
            "Mark as spam and delete."
        )
    if url_results["suspicious_count"] > 0:
        return "Email appears legitimate but contains suspicious URLs. Verify before clicking."
    return "Email appears legitimate. Always remain vigilant."


# ---------------------------------------------------------------------------
# Heuristic warning rules
# ---------------------------------------------------------------------------

_URGENCY_PATTERNS = [
    r"\burgent\b", r"\bimmediate\b", r"\bact now\b", r"\blimited time\b",
    r"\bexpires?\b", r"\bdeadline\b", r"\blast chance\b", r"\bfinal notice\b",
    r"\bimmediately\b", r"\bwithin \d+ (hour|day|minute)",
]
_REWARD_PATTERNS = [
    r"\bwon\b", r"\bwinner\b", r"\bprize\b", r"\bcongratulation",
    r"\bfree\b", r"\bgift\b", r"\bcash\b", r"\blottery\b", r"\bmillion\b",
    r"\b\$\d+", r"\bunclaimed\b", r"\breward\b",
]
_CREDENTIAL_PATTERNS = [
    r"\bverify your (account|identity|email|password)",
    r"\bconfirm your (account|details|information)",
    r"\bupdate your (account|billing|payment|credit card)",
    r"\bpassword (expired?|reset|confirm)",
    r"\bclick here to (verify|confirm|update|activate)",
    r"\bsign in to (verify|confirm|secure)",
    r"\benter your (username|password|credentials|ssn|social security)",
    r"\bsuspended? (account|access|service)",
    r"\baccount (locked|suspend|deactivat|verify|limit)",
]
_FINANCIAL_PATTERNS = [
    r"\bbank (account|transfer|wire)",
    r"\bcredit card", r"\bpayment (required|needed|overdue)",
    r"\binvoice (attach|enclosed|due)",
    r"\btransfer (fund|money|amount)",
    r"\bwire transfer", r"\bbitcoin", r"\bcryptocurrency",
    r"\bwestern union", r"\bmoneygram",
]
_SENDER_PATTERNS = [
    r"no-?reply@.*\.(xyz|top|club|work|click|gq|tk|ml|cf|ga)",
    r"@\d{4,}",
    r"support@.{0,5}\.(xyz|top|club|work|click)",
    r"security.*@(?!.*\.(com|org|edu|gov|net)\b)",
]


def _heuristic_warnings(subject, sender, body):
    text = f"{subject} {sender} {body}".lower()
    found = []

    def _check(patterns, label):
        for p in patterns:
            if re.search(p, text):
                found.append(label)
                break

    _check(_urgency_patterns_list(), "Urgency / pressure language detected")
    _check(_reward_patterns_list(), "Reward or prize language detected")
    _check(_credential_patterns_list(), "Credential harvesting language detected")
    _check(_financial_patterns_list(), "Financial request or wire-transfer language detected")

    # Suspicious sender
    for p in _SENDER_PATTERNS:
        if re.search(p, sender.lower()):
            found.append("Suspicious sender address pattern")
            break

    # ALL CAPS subject
    clean_subject = re.sub(r"[^A-Za-z]", "", subject)
    if len(clean_subject) > 4 and clean_subject == clean_subject.upper():
        found.append("Subject line is entirely in uppercase")

    # Excessive punctuation
    if len(re.findall(r"[!?]", subject + body)) > 5:
        found.append("Excessive exclamation or question marks")

    # Mismatched "Dear" generic greeting
    if re.search(r"\bdear (customer|user|client|member|friend|valued)\b", text):
        found.append("Generic impersonal greeting used")

    # Attachment references + action request
    if re.search(r"\battach(ment|ed|ment)\b", text) and re.search(r"\bopen\b|\bdownload\b", text):
        found.append("Attachment open/download request")

    return found


def _urgency_patterns_list():
    return _URGENCY_PATTERNS

def _reward_patterns_list():
    return _REWARD_PATTERNS

def _credential_patterns_list():
    return _CREDENTIAL_PATTERNS

def _financial_patterns_list():
    return _FINANCIAL_PATTERNS


# ---------------------------------------------------------------------------
# URL analyzer
# ---------------------------------------------------------------------------

_URL_REGEX = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"(?<!\w)(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)"
    r"+(?:com|org|net|edu|gov|io|info|biz|co|uk|de|fr|ru|cn|xyz|top|club|work|click|tk|ml|ga|cf|gq)"
    r"(?:/\S*)?",
    re.IGNORECASE,
)

_SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".work", ".click", ".gq", ".tk", ".ml",
    ".cf", ".ga", ".pw", ".cc", ".su", ".biz.pk", ".online", ".site",
}
_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "short.link",
    "tiny.cc", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "rb.gy",
    "shorturl.at", "clck.ru",
}
_TRUSTED_DOMAINS = {
    "google.com", "microsoft.com", "apple.com", "amazon.com", "paypal.com",
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
    "github.com", "stackoverflow.com", "wikipedia.org",
}


def extract_urls(text):
    """Extract all URLs from text."""
    raw = _URL_REGEX.findall(text or "")
    # Normalise: prepend scheme if missing
    urls = []
    seen = set()
    for u in raw:
        u = u.rstrip(".,;:\"'>)")
        if not u.startswith("http"):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            urls.append(u)
    return urls


def analyze_single_url(url):
    """
    Analyse one URL for suspicious indicators.
    Returns dict with url, suspicious (bool), reasons (list).
    Never fetches the URL.
    """
    reasons = []
    url_lower = url.lower()

    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        return {"url": url, "suspicious": True, "reasons": ["Malformed URL"]}

    # IP-based URL
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname):
        reasons.append("IP address used instead of domain name")

    # Suspicious TLD
    for tld in _SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            reasons.append(f"Suspicious top-level domain ({tld})")
            break

    # URL shortener
    if any(hostname == s or hostname.endswith("." + s) for s in _URL_SHORTENERS):
        reasons.append("URL shortener service (hides true destination)")

    # Misleading brand in subdomain
    for brand in ("paypal", "apple", "microsoft", "amazon", "google", "ebay",
                  "bankofamerica", "wellsfargo", "chase", "netflix", "irs"):
        if brand in hostname and not any(hostname == t or hostname.endswith("." + t) for t in _TRUSTED_DOMAINS):
            reasons.append(f"Brand name '{brand}' in non-official domain (possible spoofing)")
            break

    # Hyphen-heavy domains (common in phishing)
    if hostname.count("-") >= 3:
        reasons.append("Domain contains many hyphens (common phishing pattern)")

    # Excessive subdomains
    parts = hostname.split(".")
    if len(parts) >= 5:
        reasons.append("Unusually long subdomain chain")

    # Long URL
    if len(url) > 200:
        reasons.append("Abnormally long URL")

    # Encoded characters (common in obfuscation)
    if "%" in query or "%2f" in url_lower or "%40" in url_lower:
        reasons.append("Percent-encoded characters in URL (possible obfuscation)")

    # @-symbol in URL (redirects to domain after @)
    if "@" in parsed.netloc:
        reasons.append("@ symbol in URL (redirects to a different host)")

    # Double slash in path
    if "//" in path[1:]:
        reasons.append("Double slash in URL path (obfuscation technique)")

    # Port numbers (non-standard)
    if parsed.port and parsed.port not in (80, 443, 8080, 8443):
        reasons.append(f"Non-standard port {parsed.port} used")

    return {"url": url, "suspicious": bool(reasons), "reasons": reasons}


def analyze_urls(text):
    """Extract and analyze all URLs in text. Returns summary dict."""
    urls = extract_urls(text)
    results = [analyze_single_url(u) for u in urls]
    suspicious = [r for r in results if r["suspicious"]]
    return {
        "total_count": len(urls),
        "suspicious_count": len(suspicious),
        "url_details": results,
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_analyze_input(data):
    errors = []
    subject = (data.get("subject") or "").strip()
    sender = (data.get("sender") or "").strip()
    body = (data.get("body") or "").strip()

    if not body and not subject:
        errors.append("Provide at least an email subject or body.")
    if len(subject) > 998:
        errors.append("Subject exceeds maximum length (998 characters).")
    if len(sender) > 320:
        errors.append("Sender address exceeds maximum length (320 characters).")
    if len(body) > 100_000:
        errors.append("Email body exceeds maximum length (100,000 characters).")
    # Reject obvious script injection
    if re.search(r"<script", body, re.IGNORECASE):
        errors.append("Email body must not contain script tags.")

    return errors, subject, sender, body


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{% block title %}MailSentinel AI{% endblock %}</title>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #21262d;
  --border: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
  --accent2: #7c5cd8;
  --danger: #f85149;
  --warn: #e3b341;
  --success: #3fb950;
  --phish: #f85149;
  --spam: #e3b341;
  --legit: #3fb950;
  --radius: 8px;
  --shadow: 0 2px 8px rgba(0,0,0,0.4);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px;scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:-apple-system,"Segoe UI",system-ui,sans-serif;line-height:1.6;min-height:100vh}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
h1,h2,h3,h4{font-weight:600;line-height:1.3}
/* Layout */
.layout{display:flex;min-height:100vh}
.sidebar{width:240px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:0;position:fixed;top:0;left:0;height:100vh;z-index:100}
.sidebar-logo{padding:20px 24px 16px;border-bottom:1px solid var(--border)}
.sidebar-logo .logo-text{font-size:1.15rem;font-weight:700;color:var(--text)}
.sidebar-logo .logo-sub{font-size:0.72rem;color:var(--muted);letter-spacing:0.05em;text-transform:uppercase;margin-top:2px}
.sidebar-logo span.badge{display:inline-block;background:var(--accent2);color:#fff;font-size:0.65rem;padding:1px 6px;border-radius:20px;margin-left:6px;vertical-align:middle}
nav{padding:16px 0;flex:1;overflow-y:auto}
nav a{display:flex;align-items:center;gap:10px;padding:10px 24px;color:var(--muted);font-size:0.9rem;border-left:3px solid transparent;transition:all .15s}
nav a:hover,nav a.active{color:var(--text);background:var(--surface2);border-left-color:var(--accent);text-decoration:none}
nav a .nav-icon{font-size:1rem;width:18px;text-align:center}
.sidebar-footer{padding:16px 24px;border-top:1px solid var(--border);font-size:0.75rem;color:var(--muted)}
.main{margin-left:240px;flex:1;display:flex;flex-direction:column;min-height:100vh}
.topbar{background:var(--surface);border-bottom:1px solid var(--border);padding:12px 32px;display:flex;align-items:center;justify-content:space-between}
.topbar-title{font-size:1rem;font-weight:600;color:var(--text)}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--success);display:inline-block;margin-right:6px}
.status-dot.red{background:var(--danger)}
.status-dot.yellow{background:var(--warn)}
.content{padding:32px;flex:1}
/* Cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px;box-shadow:var(--shadow)}
.card + .card{margin-top:20px}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid var(--border)}
.card-title{font-size:1rem;font-weight:600;color:var(--text)}
/* Stats grid */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px;margin-bottom:24px}
.stat-card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:20px}
.stat-card .stat-value{font-size:2rem;font-weight:700;color:var(--accent)}
.stat-card .stat-value.red{color:var(--danger)}
.stat-card .stat-value.yellow{color:var(--warn)}
.stat-card .stat-value.green{color:var(--success)}
.stat-card .stat-label{font-size:0.8rem;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:0.04em}
/* Forms */
.form-group{margin-bottom:16px}
label{display:block;font-size:0.85rem;color:var(--muted);margin-bottom:6px;font-weight:500}
input[type=text],input[type=email],textarea,select{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);padding:10px 14px;font-size:0.9rem;font-family:inherit;transition:border-color .15s;outline:none}
input[type=text]:focus,input[type=email]:focus,textarea:focus,select:focus{border-color:var(--accent)}
textarea{min-height:160px;resize:vertical}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 20px;border:none;border-radius:var(--radius);font-size:0.9rem;font-weight:600;cursor:pointer;transition:opacity .15s;font-family:inherit}
.btn:hover{opacity:0.85;text-decoration:none}
.btn-primary{background:var(--accent);color:#0d1117}
.btn-danger{background:var(--danger);color:#fff}
.btn-ghost{background:var(--surface2);color:var(--text);border:1px solid var(--border)}
.btn-sm{padding:6px 12px;font-size:0.8rem}
/* Badges */
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.03em}
.badge-phishing{background:rgba(248,81,73,0.15);color:var(--phish);border:1px solid rgba(248,81,73,0.3)}
.badge-spam{background:rgba(227,179,65,0.15);color:var(--spam);border:1px solid rgba(227,179,65,0.3)}
.badge-legitimate{background:rgba(63,185,80,0.15);color:var(--legit);border:1px solid rgba(63,185,80,0.3)}
.badge-critical{background:rgba(248,81,73,0.15);color:var(--phish)}
.badge-high{background:rgba(227,179,65,0.15);color:var(--warn)}
.badge-medium{background:rgba(88,166,255,0.15);color:var(--accent)}
.badge-low{background:rgba(63,185,80,0.1);color:var(--legit)}
.badge-safe{background:rgba(63,185,80,0.15);color:var(--legit)}
/* Alerts */
.alert{padding:12px 16px;border-radius:var(--radius);margin-bottom:16px;font-size:0.88rem;border:1px solid}
.alert-error{background:rgba(248,81,73,0.1);border-color:rgba(248,81,73,0.3);color:#f85149}
.alert-success{background:rgba(63,185,80,0.1);border-color:rgba(63,185,80,0.3);color:#3fb950}
.alert-warning{background:rgba(227,179,65,0.1);border-color:rgba(227,179,65,0.3);color:#e3b341}
.alert-info{background:rgba(88,166,255,0.1);border-color:rgba(88,166,255,0.3);color:#58a6ff}
/* Table */
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:0.88rem}
th,td{padding:10px 14px;border-bottom:1px solid var(--border);text-align:left}
th{color:var(--muted);font-weight:600;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;background:var(--surface2)}
tr:hover td{background:var(--surface2)}
td.mono{font-family:monospace;font-size:0.82rem}
/* Result panel */
.result-panel{border-radius:var(--radius);padding:24px;margin-top:24px;border:2px solid}
.result-panel.phishing{border-color:var(--phish);background:rgba(248,81,73,0.05)}
.result-panel.spam{border-color:var(--spam);background:rgba(227,179,65,0.05)}
.result-panel.legitimate{border-color:var(--legit);background:rgba(63,185,80,0.05)}
.result-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.result-label{font-size:1.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em}
.result-label.phishing{color:var(--phish)}
.result-label.spam{color:var(--spam)}
.result-label.legitimate{color:var(--legit)}
.confidence-bar{background:var(--surface2);border-radius:20px;height:8px;overflow:hidden;margin:8px 0}
.confidence-fill{height:100%;border-radius:20px;transition:width .4s}
.phishing .confidence-fill{background:var(--phish)}
.spam .confidence-fill{background:var(--spam)}
.legitimate .confidence-fill{background:var(--legit)}
/* Warnings list */
.warnings-list{list-style:none;padding:0}
.warnings-list li{padding:8px 12px;background:var(--surface2);border-radius:6px;margin-bottom:6px;font-size:0.87rem;display:flex;align-items:flex-start;gap:8px}
.warnings-list li::before{content:"⚠";color:var(--warn);flex-shrink:0}
/* URL analysis */
.url-item{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;margin-bottom:10px}
.url-item.suspicious{border-color:rgba(248,81,73,0.4)}
.url-text{font-family:monospace;font-size:0.8rem;color:var(--muted);word-break:break-all}
/* Confusion matrix */
.cm-table{border-collapse:separate;border-spacing:4px}
.cm-table td{width:70px;height:50px;text-align:center;border-radius:6px;font-weight:600;font-size:1rem}
.cm-label{font-size:0.72rem;color:var(--muted);font-weight:500}
/* Modal */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:200;justify-content:center;align-items:center}
.modal-overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:28px;max-width:520px;width:90%}
.modal h3{margin-bottom:12px}
.modal p{font-size:0.9rem;color:var(--muted);margin-bottom:20px}
.modal .modal-actions{display:flex;gap:10px;justify-content:flex-end}
/* Progress spinner */
.spinner{display:inline-block;width:18px;height:18px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
/* Misc */
.text-muted{color:var(--muted)}
.text-sm{font-size:0.82rem}
.mt-1{margin-top:8px}.mt-2{margin-top:16px}.mt-3{margin-top:24px}
.flex{display:flex}.items-center{align-items:center}.gap-2{gap:8px}.gap-3{gap:12px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){
  .sidebar{transform:translateX(-100%)}
  .main{margin-left:0}
  .grid-2{grid-template-columns:1fr}
  .stats-grid{grid-template-columns:1fr 1fr}
  .content{padding:20px}
}
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-text">&#128274; MailSentinel <span class="badge">AI</span></div>
      <div class="logo-sub">Threat Detection System</div>
    </div>
    <nav>
      <a href="{{ url_for('dashboard') }}" class="{{ 'active' if active_page=='dashboard' }}">
        <span class="nav-icon">&#9632;</span> Dashboard
      </a>
      <a href="{{ url_for('analyze') }}" class="{{ 'active' if active_page=='analyze' }}">
        <span class="nav-icon">&#128269;</span> Email Analyzer
      </a>
      <a href="{{ url_for('history') }}" class="{{ 'active' if active_page=='history' }}">
        <span class="nav-icon">&#128200;</span> Scan History
      </a>
      <a href="{{ url_for('model_performance') }}" class="{{ 'active' if active_page=='performance' }}">
        <span class="nav-icon">&#127942;</span> Model Performance
      </a>
      <a href="{{ url_for('about') }}" class="{{ 'active' if active_page=='about' }}">
        <span class="nav-icon">&#8505;</span> About
      </a>
    </nav>
    <div class="sidebar-footer">
      <span class="status-dot {{ 'green' if model_loaded else 'red' }}"></span>
      {{ 'ML Model Active' if model_loaded else 'Model Not Loaded' }}
    </div>
  </aside>
  <div class="main">
    <div class="topbar">
      <span class="topbar-title">{% block page_title %}MailSentinel AI{% endblock %}</span>
      <span class="text-sm text-muted">v1.0 &nbsp;|&nbsp; Python + Flask</span>
    </div>
    <div class="content">
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% for cat, msg in messages %}
          <div class="alert alert-{{ cat }}">{{ msg }}</div>
        {% endfor %}
      {% endwith %}
      {% block content %}{% endblock %}
    </div>
  </div>
</div>
{% block scripts %}{% endblock %}
</body>
</html>
"""

_DASHBOARD_TEMPLATE = """
{% extends "base.html" %}
{% block title %}Dashboard — MailSentinel AI{% endblock %}
{% block page_title %}Dashboard{% endblock %}
{% block content %}
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">{{ stats.total }}</div>
    <div class="stat-label">Total Scans</div>
  </div>
  <div class="stat-card">
    <div class="stat-value red">{{ stats.phishing }}</div>
    <div class="stat-label">Phishing Detected</div>
  </div>
  <div class="stat-card">
    <div class="stat-value yellow">{{ stats.spam }}</div>
    <div class="stat-label">Spam Detected</div>
  </div>
  <div class="stat-card">
    <div class="stat-value green">{{ stats.legitimate }}</div>
    <div class="stat-label">Legitimate</div>
  </div>
</div>

{% if not model_loaded %}
<div class="alert alert-warning">
  &#9888; ML model not loaded. Classifications will use heuristic rules.
  See <a href="{{ url_for('about') }}">About</a> for setup instructions.
</div>
{% endif %}

<div class="card">
  <div class="card-header">
    <span class="card-title">Recent Scans</span>
    <a href="{{ url_for('analyze') }}" class="btn btn-primary btn-sm">+ New Scan</a>
  </div>
  {% if stats.recent %}
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Time</th><th>Sender</th><th>Subject</th>
        <th>Label</th><th>Risk</th><th>URLs</th><th></th>
      </tr></thead>
      <tbody>
      {% for scan in stats.recent %}
        <tr>
          <td class="text-sm text-muted">{{ scan.created_at[:16] }}</td>
          <td class="mono">{{ scan.sender or '—' }}</td>
          <td>{{ scan.subject or '(no subject)' }}</td>
          <td><span class="badge badge-{{ scan.label }}">{{ scan.label }}</span></td>
          <td><span class="badge badge-{{ scan.risk_level }}">{{ scan.risk_level }}</span></td>
          <td>{{ scan.url_count }}{% if scan.suspicious_url_count > 0 %} <span class="badge badge-phishing">{{ scan.suspicious_url_count }} ⚠</span>{% endif %}</td>
          <td><a href="{{ url_for('scan_detail', scan_id=scan.id) }}" class="btn btn-ghost btn-sm">View</a></td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <p class="text-muted text-sm">No scans yet. <a href="{{ url_for('analyze') }}">Analyze an email</a> to get started.</p>
  {% endif %}
</div>
{% endblock %}
"""

_ANALYZE_TEMPLATE = """
{% extends "base.html" %}
{% block title %}Email Analyzer — MailSentinel AI{% endblock %}
{% block page_title %}Email Analyzer{% endblock %}
{% block content %}
<div class="card">
  <div class="card-header">
    <span class="card-title">&#128269; Analyze Email</span>
    {% if not model_loaded %}
    <span class="badge badge-spam">Heuristic Mode</span>
    {% else %}
    <span class="badge badge-legitimate">ML Active</span>
    {% endif %}
  </div>
  {% if errors %}
    {% for e in errors %}<div class="alert alert-error">{{ e }}</div>{% endfor %}
  {% endif %}
  <form method="POST" action="{{ url_for('analyze') }}" id="analyzeForm">
    <div class="grid-2">
      <div class="form-group">
        <label>Sender Email Address</label>
        <input type="text" name="sender" placeholder="e.g. support@example.com"
               value="{{ form.sender }}" maxlength="320" autocomplete="off">
      </div>
      <div class="form-group">
        <label>Email Subject</label>
        <input type="text" name="subject" placeholder="e.g. Your account requires attention"
               value="{{ form.subject }}" maxlength="998">
      </div>
    </div>
    <div class="form-group">
      <label>Email Body (plain text only — HTML tags will be stripped)</label>
      <textarea name="body" placeholder="Paste the email content here...">{{ form.body }}</textarea>
    </div>
    <button type="submit" class="btn btn-primary" id="submitBtn">
      <span id="btnText">&#128269; Analyze Email</span>
      <span id="btnSpinner" class="spinner" style="display:none"></span>
    </button>
  </form>
</div>

{% if result %}
<div class="result-panel {{ result.label }}">
  <div class="result-header">
    <div>
      <div class="result-label {{ result.label }}">{{ result.label }}</div>
      <div class="text-sm text-muted mt-1">
        Confidence: <strong>{{ (result.confidence * 100) | int }}%</strong> &nbsp;|&nbsp;
        Risk: <span class="badge badge-{{ result.risk_level }}">{{ result.risk_level }}</span>
        {% if result.ml_used %}
        &nbsp;|&nbsp; <span class="badge badge-legitimate">ML Classification</span>
        {% else %}
        &nbsp;|&nbsp; <span class="badge badge-spam">Heuristic Classification</span>
        {% endif %}
      </div>
    </div>
    <a href="{{ url_for('scan_detail', scan_id=scan_id) }}" class="btn btn-ghost btn-sm">View Full Report</a>
  </div>
  <div class="confidence-bar">
    <div class="confidence-fill" style="width:{{ (result.confidence * 100) | int }}%"></div>
  </div>

  {% if result.warnings %}
  <div class="mt-2">
    <div style="font-size:0.85rem;font-weight:600;color:var(--warn);margin-bottom:8px">
      &#9888; Warning Signals ({{ result.warnings | length }})
    </div>
    <ul class="warnings-list">
      {% for w in result.warnings %}<li>{{ w }}</li>{% endfor %}
    </ul>
  </div>
  {% endif %}

  <div class="card mt-2" style="background:var(--surface2);border:1px solid var(--border)">
    <div style="font-size:0.85rem;font-weight:600;margin-bottom:8px">&#128203; Recommendation</div>
    <p class="text-sm">{{ result.details.recommendation }}</p>
  </div>

  {% if result.url_results.total_count > 0 %}
  <div class="mt-2">
    <div style="font-size:0.85rem;font-weight:600;margin-bottom:8px">
      &#128279; URLs Found ({{ result.url_results.total_count }}
      {% if result.url_results.suspicious_count > 0 %}
      — <span style="color:var(--danger)">{{ result.url_results.suspicious_count }} suspicious</span>
      {% endif %})
    </div>
    {% for url_item in result.url_results.url_details %}
    <div class="url-item {{ 'suspicious' if url_item.suspicious }}">
      <div class="flex items-center gap-2" style="margin-bottom:4px">
        {% if url_item.suspicious %}
        <span class="badge badge-phishing">Suspicious</span>
        {% else %}
        <span class="badge badge-legitimate">OK</span>
        {% endif %}
        <span class="url-text">{{ url_item.url }}</span>
      </div>
      {% if url_item.reasons %}
      <ul style="list-style:none;margin-top:4px">
        {% for r in url_item.reasons %}
        <li class="text-sm" style="color:var(--warn)">&#8618; {{ r }}</li>
        {% endfor %}
      </ul>
      {% endif %}
    </div>
    {% endfor %}
  </div>
  {% endif %}
</div>
{% endif %}
{% endblock %}
{% block scripts %}
<script>
document.getElementById('analyzeForm').addEventListener('submit', function() {
  document.getElementById('btnText').style.display = 'none';
  document.getElementById('btnSpinner').style.display = 'inline-block';
});
</script>
{% endblock %}
"""

_HISTORY_TEMPLATE = """
{% extends "base.html" %}
{% block title %}Scan History — MailSentinel AI{% endblock %}
{% block page_title %}Scan History{% endblock %}
{% block content %}
<div class="card">
  <div class="card-header">
    <span class="card-title">&#128200; Scan History ({{ scans | length }})</span>
    {% if scans %}
    <button class="btn btn-danger btn-sm" onclick="document.getElementById('clearModal').classList.add('open')">
      &#128465; Clear History
    </button>
    {% endif %}
  </div>
  {% if scans %}
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Date/Time</th><th>Sender</th><th>Subject</th>
        <th>Label</th><th>Risk</th><th>Confidence</th>
        <th>URLs</th><th></th>
      </tr></thead>
      <tbody>
      {% for scan in scans %}
        <tr>
          <td class="text-sm text-muted">{{ scan.created_at[:16] }}</td>
          <td class="mono text-sm">{{ scan.sender or '—' }}</td>
          <td class="text-sm">{{ (scan.subject or '(no subject)')[:60] }}</td>
          <td><span class="badge badge-{{ scan.label }}">{{ scan.label }}</span></td>
          <td><span class="badge badge-{{ scan.risk_level }}">{{ scan.risk_level }}</span></td>
          <td class="text-sm">{{ (scan.confidence * 100) | int }}%</td>
          <td class="text-sm">{{ scan.url_count }}
            {% if scan.suspicious_url_count > 0 %}
            <span class="badge badge-phishing">{{ scan.suspicious_url_count }}⚠</span>
            {% endif %}
          </td>
          <td><a href="{{ url_for('scan_detail', scan_id=scan.id) }}" class="btn btn-ghost btn-sm">View</a></td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <p class="text-muted text-sm">No scan records found. <a href="{{ url_for('analyze') }}">Analyze an email</a> to begin.</p>
  {% endif %}
</div>

<!-- Confirm clear modal -->
<div class="modal-overlay" id="clearModal">
  <div class="modal">
    <h3>Clear All Scan History?</h3>
    <p>This will permanently delete all {{ scans | length }} scan records. This action cannot be undone.</p>
    <div class="modal-actions">
      <button class="btn btn-ghost" onclick="document.getElementById('clearModal').classList.remove('open')">Cancel</button>
      <form method="POST" action="{{ url_for('history_delete') }}" style="display:inline">
        <button type="submit" class="btn btn-danger">Delete All</button>
      </form>
    </div>
  </div>
</div>
{% endblock %}
"""

_DETAIL_TEMPLATE = """
{% extends "base.html" %}
{% block title %}Scan Detail — MailSentinel AI{% endblock %}
{% block page_title %}Scan Detail{% endblock %}
{% block content %}
{% if scan %}
<div class="flex items-center gap-3" style="margin-bottom:24px">
  <a href="{{ url_for('history') }}" class="btn btn-ghost btn-sm">&#8592; Back</a>
  <span class="badge badge-{{ scan.label }}">{{ scan.label }}</span>
  <span class="badge badge-{{ scan.risk_level }}">{{ scan.risk_level }}</span>
  <span class="text-sm text-muted">Scanned {{ scan.created_at[:16] }}</span>
</div>

<div class="grid-2">
  <div class="card">
    <div class="card-header"><span class="card-title">Email Metadata</span></div>
    <table style="font-size:0.88rem"><tbody>
      <tr><th>Sender</th><td>{{ scan.sender or '—' }}</td></tr>
      <tr><th>Subject</th><td>{{ scan.subject or '(no subject)' }}</td></tr>
      <tr><th>Scan ID</th><td class="mono text-sm">{{ scan.id }}</td></tr>
    </tbody></table>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Classification Result</span></div>
    <div class="result-label {{ scan.label }}" style="font-size:1.4rem;text-transform:uppercase;margin-bottom:10px">{{ scan.label }}</div>
    <div class="confidence-bar">
      <div class="confidence-fill {{ scan.label }}" style="width:{{ (scan.confidence * 100) | int }}%"></div>
    </div>
    <p class="text-sm text-muted mt-1">Confidence: {{ (scan.confidence * 100) | int }}%</p>
    {% if details.recommendation %}
    <p class="text-sm mt-2">{{ details.recommendation }}</p>
    {% endif %}
  </div>
</div>

{% if warnings %}
<div class="card mt-2">
  <div class="card-header"><span class="card-title">&#9888; Warning Signals</span></div>
  <ul class="warnings-list">
    {% for w in warnings %}<li>{{ w }}</li>{% endfor %}
  </ul>
</div>
{% endif %}

{% if url_details %}
<div class="card mt-2">
  <div class="card-header">
    <span class="card-title">&#128279; URL Analysis ({{ url_details | length }})</span>
    <span class="text-sm text-muted">{{ scan.suspicious_url_count }} suspicious</span>
  </div>
  {% for url_item in url_details %}
  <div class="url-item {{ 'suspicious' if url_item.suspicious }}" style="margin-bottom:10px">
    <div class="flex items-center gap-2" style="margin-bottom:4px">
      {% if url_item.suspicious %}
      <span class="badge badge-phishing">Suspicious</span>
      {% else %}
      <span class="badge badge-legitimate">Clean</span>
      {% endif %}
      <span class="url-text">{{ url_item.url }}</span>
    </div>
    {% if url_item.reasons %}
    <ul style="list-style:none;margin-top:4px">
      {% for r in url_item.reasons %}
      <li class="text-sm" style="color:var(--warn)">&#8618; {{ r }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  </div>
  {% endfor %}
</div>
{% endif %}

{% else %}
<div class="alert alert-error">Scan record not found.</div>
<a href="{{ url_for('history') }}" class="btn btn-ghost">Back to History</a>
{% endif %}
{% endblock %}
"""

_PERFORMANCE_TEMPLATE = """
{% extends "base.html" %}
{% block title %}Model Performance — MailSentinel AI{% endblock %}
{% block page_title %}Model Performance{% endblock %}
{% block content %}
{% if not model_loaded %}
<div class="alert alert-warning">
  &#9888; ML model is not loaded. Performance metrics are unavailable.
  See <a href="{{ url_for('about') }}">About</a> for training instructions.
</div>
{% endif %}

{% if metrics %}
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value green">{{ (metrics.accuracy * 100) | round(1) }}%</div>
    <div class="stat-label">Accuracy</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{{ (metrics.precision * 100) | round(1) }}%</div>
    <div class="stat-label">Precision</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{{ (metrics.recall * 100) | round(1) }}%</div>
    <div class="stat-label">Recall</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{{ (metrics.f1_score * 100) | round(1) }}%</div>
    <div class="stat-label">F1 Score</div>
  </div>
</div>

<div class="grid-2">
  <div class="card">
    <div class="card-header"><span class="card-title">Metric Details</span></div>
    <table style="font-size:0.9rem"><tbody>
      <tr><td>Train samples</td><td><strong>{{ metrics.train_size }}</strong></td></tr>
      <tr><td>Test samples</td><td><strong>{{ metrics.test_size }}</strong></td></tr>
      <tr><td>Accuracy</td><td><strong>{{ (metrics.accuracy * 100) | round(2) }}%</strong></td></tr>
      <tr><td>Precision (weighted)</td><td><strong>{{ (metrics.precision * 100) | round(2) }}%</strong></td></tr>
      <tr><td>Recall (weighted)</td><td><strong>{{ (metrics.recall * 100) | round(2) }}%</strong></td></tr>
      <tr><td>F1 Score (weighted)</td><td><strong>{{ (metrics.f1_score * 100) | round(2) }}%</strong></td></tr>
      <tr><td>Trained at</td><td class="text-sm text-muted">{{ metrics.trained_at[:16] }}</td></tr>
    </tbody></table>
  </div>

  <div class="card">
    <div class="card-header"><span class="card-title">Confusion Matrix</span></div>
    <div style="overflow-x:auto">
      <table class="cm-table">
        <thead>
          <tr>
            <th></th>
            {% for lbl in metrics.confusion_labels %}
            <th class="cm-label" style="text-align:center">Pred {{ lbl }}</th>
            {% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for i, row in enumerate(metrics.confusion_matrix) %}
          <tr>
            <th class="cm-label">True {{ metrics.confusion_labels[i] }}</th>
            {% for j, val in enumerate(row) %}
            {% set is_diag = (i == j) %}
            <td style="background: {{ 'rgba(63,185,80,0.25)' if is_diag else 'rgba(248,81,73,0.10)' }};
                        color: {{ '#3fb950' if is_diag else '#8b949e' }}">{{ val }}</td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    <p class="text-sm text-muted mt-1">Diagonal = correct predictions (green)</p>
  </div>
</div>

{% else %}
<div class="card">
  <div class="card-header"><span class="card-title">No Metrics Available</span></div>
  <p class="text-muted text-sm">
    Train the model first to see performance metrics.
    See the <a href="{{ url_for('about') }}">About page</a> for instructions.
  </p>
</div>
{% endif %}

<div class="card mt-2">
  <div class="card-header"><span class="card-title">&#127775; Train / Retrain Model</span></div>
  <p class="text-sm text-muted" style="margin-bottom:16px">
    Place your dataset CSV at <code>data/dataset.csv</code> with columns <code>label</code> and <code>text</code>,
    then click the button below. Labels: <code>legitimate</code>, <code>spam</code>, <code>phishing</code>.
  </p>
  <form method="POST" action="{{ url_for('train_route') }}" id="trainForm">
    <button type="submit" class="btn btn-primary" onclick="this.disabled=true;this.innerHTML='Training… (may take a moment)'">
      &#127775; Train Model Now
    </button>
  </form>
  {% if train_message %}
  <div class="alert {{ 'alert-success' if train_success else 'alert-error' }} mt-2">{{ train_message }}</div>
  {% endif %}
</div>
{% endblock %}
"""

_ABOUT_TEMPLATE = """
{% extends "base.html" %}
{% block title %}About — MailSentinel AI{% endblock %}
{% block page_title %}About MailSentinel AI{% endblock %}
{% block content %}
<div class="card">
  <div class="card-header"><span class="card-title">&#128274; Project Overview</span></div>
  <p style="font-size:0.92rem;margin-bottom:12px">
    <strong>MailSentinel AI</strong> is an AI-powered email threat detection system that uses machine learning
    and heuristic analysis to classify emails as <em>legitimate</em>, <em>spam</em>, or <em>phishing</em>.
    It is designed as an educational cybersecurity project.
  </p>
  <p class="text-sm text-muted">
    All processing is performed locally. The system never sends email content externally and never
    automatically visits URLs found in scanned emails.
  </p>
</div>

<div class="grid-2 mt-2">
  <div class="card">
    <div class="card-header"><span class="card-title">&#129520; Technology Stack</span></div>
    <ul style="list-style:none;font-size:0.88rem">
      <li style="padding:6px 0;border-bottom:1px solid var(--border)">&#128013; Python 3.10+</li>
      <li style="padding:6px 0;border-bottom:1px solid var(--border)">&#9889; Flask (web framework)</li>
      <li style="padding:6px 0;border-bottom:1px solid var(--border)">&#129302; scikit-learn (ML, TF-IDF)</li>
      <li style="padding:6px 0;border-bottom:1px solid var(--border)">&#128202; pandas / NumPy (data)</li>
      <li style="padding:6px 0;border-bottom:1px solid var(--border)">&#128172; NLTK (NLP preprocessing)</li>
      <li style="padding:6px 0;border-bottom:1px solid var(--border)">&#128451; SQLite (scan history)</li>
      <li style="padding:6px 0">&#128196; Joblib (model serialization)</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">&#127775; ML Pipeline</span></div>
    <ol style="font-size:0.88rem;padding-left:18px">
      <li style="padding:5px 0">Text preprocessing (lowercase, de-HTML, stop-word removal, stemming)</li>
      <li style="padding:5px 0">TF-IDF vectorization (bigrams, 15,000 features)</li>
      <li style="padding:5px 0">Logistic Regression classifier (balanced class weights)</li>
      <li style="padding:5px 0">80/20 stratified train/test split</li>
      <li style="padding:5px 0">Model and metrics saved with Joblib</li>
    </ol>
  </div>
</div>

<div class="card mt-2">
  <div class="card-header"><span class="card-title">&#128203; Dataset Setup</span></div>
  <p class="text-sm" style="margin-bottom:12px">Place a CSV file at <code>data/dataset.csv</code> with two columns:</p>
  <table style="font-size:0.88rem;margin-bottom:12px"><thead><tr>
    <th>label</th><th>text</th>
  </tr></thead><tbody>
    <tr><td><code>legitimate</code></td><td>Hi, please find the report attached...</td></tr>
    <tr><td><code>spam</code></td><td>Congratulations! You have WON $1,000,000...</td></tr>
    <tr><td><code>phishing</code></td><td>Click here to verify your account immediately...</td></tr>
  </tbody></table>
  <p class="text-sm text-muted">
    Recommended datasets: <strong>SpamAssassin Public Corpus</strong>, <strong>Enron Email Dataset</strong>,
    <strong>UCI SMS Spam Collection</strong> (converted to email format), or any labeled phishing dataset.
    After placing the CSV, go to <a href="{{ url_for('model_performance') }}">Model Performance</a> and click "Train Model Now".
  </p>
</div>

<div class="card mt-2">
  <div class="card-header"><span class="card-title">&#9888; Limitations</span></div>
  <ul style="font-size:0.88rem;padding-left:18px">
    <li style="padding:4px 0">Classification accuracy depends entirely on training data quality and size.</li>
    <li style="padding:4px 0">Without a trained model, only heuristic rules are applied.</li>
    <li style="padding:4px 0">URL analysis is static (no DNS resolution or page fetching).</li>
    <li style="padding:4px 0">Adversarial phishing emails crafted to evade keyword detection may be missed.</li>
    <li style="padding:4px 0">This system is for educational purposes and is not a production security product.</li>
  </ul>
</div>

<div class="card mt-2">
  <div class="card-header"><span class="card-title">&#128274; Security Controls</span></div>
  <ul style="font-size:0.88rem;padding-left:18px">
    <li style="padding:4px 0">All user input is escaped before rendering — no XSS vectors.</li>
    <li style="padding:4px 0">SQLite queries use parameterized statements — no SQL injection.</li>
    <li style="padding:4px 0">Email HTML tags are stripped before processing.</li>
    <li style="padding:4px 0">Full email bodies are not stored in the database.</li>
    <li style="padding:4px 0">Suspicious URLs are never automatically visited.</li>
    <li style="padding:4px 0">Request body size capped at 2 MB.</li>
  </ul>
</div>
{% endblock %}
"""

# ---------------------------------------------------------------------------
# Template registry — register all templates in a DictLoader so that
# {% extends "base.html" %} works correctly with in-memory templates.
# ---------------------------------------------------------------------------

_ERROR_TEMPLATE = """{% extends "base.html" %}
{% block title %}Error — MailSentinel AI{% endblock %}
{% block page_title %}Error{% endblock %}
{% block content %}
<div class="alert alert-error">{{ error_message | safe }}</div>
<a href="{{ url_for('dashboard') }}" class="btn btn-ghost">Go to Dashboard</a>
{% endblock %}
"""

_TEMPLATE_DICT = {
    "base.html": _BASE_TEMPLATE,
    "dashboard.html": _DASHBOARD_TEMPLATE,
    "analyze.html": _ANALYZE_TEMPLATE,
    "history.html": _HISTORY_TEMPLATE,
    "detail.html": _DETAIL_TEMPLATE,
    "performance.html": _PERFORMANCE_TEMPLATE,
    "about.html": _ABOUT_TEMPLATE,
    "error.html": _ERROR_TEMPLATE,
}

# Patch Flask's Jinja2 loader to include our in-memory templates
_dict_loader = jinja2.DictLoader(_TEMPLATE_DICT)
app.jinja_loader = jinja2.ChoiceLoader([_dict_loader])


def _render(template_name, **kwargs):
    """Render a named in-memory template with common context variables injected."""
    kwargs.setdefault("model_loaded", _model_state["loaded"])
    kwargs.setdefault("enumerate", enumerate)
    return render_template(template_name, **kwargs)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    stats = get_dashboard_stats()
    return _render("dashboard.html", active_page="dashboard", stats=stats)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    result = None
    scan_id = None
    errors = []
    form = {"sender": "", "subject": "", "body": ""}

    if request.method == "POST":
        raw = request.form
        errors, subject, sender, body = validate_analyze_input(raw)
        form = {"sender": sender, "subject": subject, "body": body}

        if not errors:
            try:
                result = predict_email(subject, sender, body)
                scan_id = insert_scan(
                    sender=sender,
                    subject=subject,
                    label=result["label"],
                    confidence=result["confidence"],
                    risk_level=result["risk_level"],
                    url_count=result["url_results"]["total_count"],
                    suspicious_url_count=result["url_results"]["suspicious_count"],
                    warnings=result["warnings"],
                    details=result["details"],
                )
            except Exception as e:
                logger.exception("Analysis error")
                errors.append("An internal error occurred. Please try again.")
                result = None

    return _render(
        "analyze.html",
        active_page="analyze",
        result=result,
        scan_id=scan_id,
        errors=errors,
        form=form,
    )


@app.route("/history")
def history():
    scans = get_scan_history(limit=200)
    return _render("history.html", active_page="history", scans=scans)


@app.route("/history/delete", methods=["POST"])
def history_delete():
    delete_all_scans()
    return redirect(url_for("history"))


@app.route("/scan/<scan_id>")
def scan_detail(scan_id):
    # Validate scan_id format to prevent injection
    if not re.match(r"^[0-9a-f\-]{36}$", scan_id):
        return redirect(url_for("history"))
    scan = get_scan_by_id(scan_id)
    warnings = []
    url_details = []
    details = {}
    if scan:
        try:
            warnings = json.loads(scan.get("warnings") or "[]")
        except Exception:
            pass
        try:
            raw_details = json.loads(scan.get("details") or "{}")
            details = raw_details
            # url_details may be stored; if not, re-extract from details
        except Exception:
            pass
        # url_details are not stored (privacy); just show counts
    return _render(
        "detail.html",
        active_page="history",
        scan=scan,
        warnings=warnings,
        url_details=url_details,
        details=details,
    )


@app.route("/performance", methods=["GET"])
def model_performance():
    return _render(
        "performance.html",
        active_page="performance",
        metrics=_model_state["metrics"],
        train_message=None,
        train_success=False,
    )


@app.route("/train", methods=["POST"])
def train_route():
    success, message = train_model()
    return _render(
        "performance.html",
        active_page="performance",
        metrics=_model_state["metrics"],
        train_message=message,
        train_success=success,
    )


@app.route("/about")
def about():
    return _render("about.html", active_page="about")


# ---------------------------------------------------------------------------
# JSON API endpoints (for optional JS integration)
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(force=False, silent=True) or {}
    errors, subject, sender, body = validate_analyze_input(data)
    if errors:
        return jsonify({"errors": errors}), 400
    try:
        result = predict_email(subject, sender, body)
        scan_id = insert_scan(
            sender=sender,
            subject=subject,
            label=result["label"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            url_count=result["url_results"]["total_count"],
            suspicious_url_count=result["url_results"]["suspicious_count"],
            warnings=result["warnings"],
            details=result["details"],
        )
        result["scan_id"] = scan_id
        return jsonify(result), 200
    except Exception as e:
        logger.exception("API analysis error")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    return jsonify(get_dashboard_stats()), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

def _render_error(message, status, active_page=""):
    return _render("error.html", active_page=active_page, error_message=message), status


@app.errorhandler(404)
def not_found(e):
    return _render_error("404 — Page not found.", 404)


@app.errorhandler(413)
def too_large(e):
    return _render_error("413 — Request too large. Maximum allowed size is 2 MB.", 413)


@app.errorhandler(500)
def server_error(e):
    logger.error("500: %s", e)
    return _render_error("An internal error occurred. Please try again.", 500)


# ---------------------------------------------------------------------------
# CLI training helper (python app.py --train)
# ---------------------------------------------------------------------------

def cli_train():
    """Run training from the command line."""
    dataset = None
    if len(sys.argv) >= 3:
        dataset = sys.argv[2]
    with app.app_context():
        success, message = train_model(dataset)
    print(("[OK]" if success else "[FAIL]"), message)
    sys.exit(0 if success else 1)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--train":
        cli_train()

    init_db()
    load_model()

    if not _model_state["loaded"]:
        logger.warning(
            "ML model not found. Starting in heuristic mode. "
            "Add data/dataset.csv and run: python app.py --train"
        )

    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting MailSentinel AI on http://127.0.0.1:%d", port)
    app.run(
        host="0.0.0.0",
        port=port,
        debug=DEBUG_MODE,
        use_reloader=False,
    )
