"""Quick integration tests for MailSentinel AI."""
import sys
import os
import json

os.environ.setdefault("SECRET_KEY", "test-key-12345")

import importlib.util
spec = importlib.util.spec_from_file_location("app", "app.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

app = mod.app
mod.init_db()
client = app.test_client()

PASS = 0
FAIL = 0


def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))
        FAIL += 1


# ---- Basic page routes ----
print("\n== Page Routes ==")
for path, expected in [
    ("/", 302),
    ("/dashboard", 200),
    ("/analyze", 200),
    ("/history", 200),
    ("/performance", 200),
    ("/about", 200),
    ("/nonexistent", 404),
]:
    r = client.get(path)
    check(f"GET {path}", r.status_code == expected, f"got {r.status_code}")

# ---- Analyze (POST) ----
print("\n== Email Analysis ==")

r = client.post("/analyze", data={
    "sender": "fake@suspicious.xyz",
    "subject": "URGENT verify your account",
    "body": "Click here to verify http://paypal-verify.tk/login or lose access forever",
})
check("POST /analyze returns 200", r.status_code == 200)
body_lower = r.data.lower()
check("Phishing or spam detected", b"phishing" in body_lower or b"spam" in body_lower)
check("Warning signal shown", b"warning" in body_lower or b"urgency" in body_lower or b"credential" in body_lower)

# Legitimate email
r = client.post("/analyze", data={
    "sender": "colleague@company.com",
    "subject": "Meeting notes",
    "body": "Hi, please find attached the notes from today meeting. Best regards.",
})
check("POST /analyze legitimate email 200", r.status_code == 200)
check("Legitimate detected", b"legitimate" in r.data.lower())

# Validation: empty body+subject
r = client.post("/analyze", data={"sender": "", "subject": "", "body": ""})
check("Empty input returns 200 with error", r.status_code == 200)
check("Error message shown", b"provide" in r.data.lower() or b"subject" in r.data.lower() or b"body" in r.data.lower())

# ---- URL analysis ----
print("\n== URL Analysis ==")
urls = mod.extract_urls("Visit http://192.168.1.1/phish and http://bit.ly/abc and https://google.com")
check("Extract 3 URLs", len(urls) == 3, f"got {len(urls)}")

r1 = mod.analyze_single_url("http://192.168.1.1/login")
check("IP URL is suspicious", r1["suspicious"])
check("IP reason present", any("IP address" in x for x in r1["reasons"]))

r2 = mod.analyze_single_url("http://bit.ly/abc123")
check("Shortener URL is suspicious", r2["suspicious"])

r3 = mod.analyze_single_url("https://google.com/search?q=test")
check("Google URL is clean", not r3["suspicious"])

# ---- Heuristic warnings ----
print("\n== Heuristic Rules ==")
w = mod._heuristic_warnings(
    "URGENT ACTION REQUIRED",
    "noreply@secure.xyz",
    "You must verify your account immediately or it will be suspended",
)
check("Urgency warning", any("urgency" in x.lower() for x in w))
check("Credential warning", any("credential" in x.lower() for x in w))

w2 = mod._heuristic_warnings("Meeting notes", "boss@company.com", "See attached agenda for tomorrow")
check("Legit email has few warnings", len(w2) <= 1)

# ---- Database operations ----
print("\n== Database ==")
with app.app_context():
    mod.init_db()
    scan_id = mod.insert_scan(
        sender="test@example.com",
        subject="Test subject",
        label="spam",
        confidence=0.85,
        risk_level="high",
        url_count=2,
        suspicious_url_count=1,
        warnings=["Test warning"],
        details={"recommendation": "Delete this email"},
    )
    check("Insert scan returns ID", bool(scan_id) and len(scan_id) == 36)

    retrieved = mod.get_scan_by_id(scan_id)
    check("Retrieve scan by ID", retrieved is not None)
    check("Scan label correct", retrieved["label"] == "spam")
    check("Scan sender correct", retrieved["sender"] == "test@example.com")

    stats = mod.get_dashboard_stats()
    check("Dashboard stats has totals", "total" in stats and stats["total"] >= 1)

# ---- Input validation ----
print("\n== Input Validation ==")
errors, s, sn, b = mod.validate_analyze_input({"subject": "", "sender": "", "body": ""})
check("Empty input flagged", len(errors) > 0)

errors2, s2, sn2, b2 = mod.validate_analyze_input({
    "subject": "Hello",
    "sender": "test@test.com",
    "body": "Normal email content"
})
check("Valid input passes", len(errors2) == 0)

errors3, *_ = mod.validate_analyze_input({
    "subject": "Test",
    "sender": "x",
    "body": "<script>alert(1)</script> test content"
})
check("Script tag rejected", len(errors3) > 0)

# ---- JSON API ----
print("\n== JSON API ==")
r = client.post("/api/analyze",
    data=json.dumps({
        "sender": "win@lottery.tk",
        "subject": "You Won a Prize!",
        "body": "Congratulations! Click here to claim your reward now"
    }),
    content_type="application/json",
)
check("POST /api/analyze returns 200", r.status_code == 200)
d = json.loads(r.data)
check("JSON has label field", "label" in d)
check("JSON has confidence field", "confidence" in d)
check("JSON has risk_level field", "risk_level" in d)
check("JSON has scan_id field", "scan_id" in d)

# Bad JSON API request
r = client.post("/api/analyze",
    data=json.dumps({"subject": "", "body": ""}),
    content_type="application/json",
)
check("Empty API request returns 400", r.status_code == 400)

# ---- Model state ----
print("\n== Model State ==")
with app.app_context():
    mod.load_model()  # triggers the "not found" error message
check("Model not loaded (no dataset yet)", not mod._model_state["loaded"])
check("ML error message set after load_model()", bool(mod._model_state["error"]))

# ---- Summary ----
print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
