# AI-SOC Platform — Setup & Run Guide

## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Start the server
python backend/app.py

# 3. Open browser
# Go to: http://localhost:5000
```

---

## Project Structure

```
ai_soc_platform/
├── backend/
│   ├── app.py              ← Flask API (run this)
│   └── requirements.txt    ← Python packages
├── frontend/
│   └── index.html          ← Dashboard (served by Flask)
└── data/
    ├── phase2_gnn_lateral.csv       ← GNN lateral movement suspects
    ├── phase2_ueba_threats.csv      ← UEBA insider threat scores
    ├── phase2_rl_playbooks.csv      ← RL-SOAR optimal playbooks
    ├── phase2_metrics.json          ← Phase 2 metrics
    ├── phase3_malware_families.csv  ← DBSCAN malware families
    ├── phase3_metrics.json          ← Phase 3 metrics
    ├── phase3_zero_day_threats.csv  ← Zero-day threat records
    └── phase5_compliance_report.json← Compliance report
```

---

## How It Works

**Backend (Flask):**
- Reads ALL data from CSV/JSON files in `data/` folder
- ZERO hardcoded values — everything from real datasets
- Copilot routes questions to relevant data and returns live answers

**Frontend (HTML/JS):**
- Fetches ALL data from Flask API on page load
- Every table, chart, and metric is pulled from the API
- Copilot sends POST to `/api/copilot` and displays real dataset answers

---

## API Endpoints

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /api/metrics` | All KPIs from real data files |
| `GET /api/gnn` | GNN lateral movement suspects (from CSV) |
| `GET /api/ueba` | UEBA insider threat scores (from CSV) |
| `GET /api/soar` | RL-SOAR playbooks (from CSV) |
| `GET /api/clustering` | Malware families (from CSV) |
| `GET /api/adversarial` | Adversarial ML results (from JSON) |
| `GET /api/compliance` | Compliance report (from JSON) |
| `GET /api/mitre` | MITRE ATT&CK kill-chain |
| `POST /api/copilot` | Copilot answer from real datasets |

---

## VS Code Steps

1. Open the project folder in VS Code
2. Open Terminal (Ctrl + `)
3. Run:
   ```bash
   pip install -r backend/requirements.txt
   python backend/app.py
   ```
4. Open Chrome/Edge → go to `http://localhost:5000`

---

## Copilot Examples

Type any of these in the Copilot chat:

- `Show me last 2 hours data`
- `Which machines have lateral movement?`
- `Are there insider threats right now?`
- `What malware families are active?`
- `Which incidents need SOAR escalation?`
- `Have any phishing attacks been detected?`
- `ISO 27001 compliance status`

All answers come from the real CSV/JSON files in `data/`.
