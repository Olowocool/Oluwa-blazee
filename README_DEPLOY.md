# NBA AI Betting System — Deployment & Operating Guide

This guide is for deploying and operating the current working NBA AI betting system without restructuring the project.

---

## 1. Required Files Before Deployment

Make sure these files exist in the GitHub repository root:

```text
frontend.py
requirements.txt
runtime.txt
render.yaml
app.py
automation_runner.py
train_ensemble_model.py
model_evaluation.py
model_health.py
model_versioning.py
model_rollback.py
confidence_engine.py
uncertainty_engine.py
auto_learning.py
auto_update_results.py
historical_backfill_engine.py
historical_data_engine.py
odds_snapshot_engine.py
market_intelligence_engine.py
advanced_features.py
injury_rest_engine.py
ensemble_consensus.py
feature_engineering.py
team_map.json
```

Also make sure this folder exists:

```text
models/
```

And contains at least one base model such as:

```text
basketball_xgb_calibrated.joblib
basketball_xgb_calibrated_v2.joblib
basketball_xgb_calibrated_v3.joblib
```

---

## 2. requirements.txt Must Include

Your `requirements.txt` should include at least:

```text
streamlit
pandas
numpy
requests
scikit-learn
joblib
xgboost
lxml
html5lib
beautifulsoup4
fastapi
uvicorn
nba_api
pyarrow
```

If Render says `streamlit: command not found`, it means `streamlit` is missing from `requirements.txt`.

---

## 3. Render Settings

Create a new Render Web Service.

### Runtime

```text
Python 3
```

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
```

### Branch

```text
main
```

---

## 4. Environment Variables

Add this in Render if you use live odds:

```text
ODDS_API_KEY=your_api_key_here
```

If no key is added, the app will still run, but live odds may not load.

---

## 5. First Actions After Fresh Deployment

A fresh Render deployment may not have runtime CSV files yet.

After opening the app, do this exact order:

1. Scroll to **Historical Backfill Engine**.
2. Set rows to `500`.
3. Click **Generate Historical Backfill Data**.
4. Click **Train Ensemble Model**.
5. Click **Run Full Daily Automation**.
6. Refresh the page.

Expected dashboard values after this:

```text
Training Rows: 500+
Win Rate: non-zero
ROI: non-zero
Saved Model Versions: at least 1
```

---

## 6. Daily Operating Workflow

Each day, use this order:

1. Click **Load Daily Predictions**.
2. Review predictions, confidence, injury impact, and ensemble consensus.
3. Save candidate/bet picks if needed.
4. Later, click **Run Auto Result Sync**.
5. Click **Run Full Daily Automation**.
6. Check:
   - Model Health Dashboard
   - Saved Model Versions
   - Bet Performance Dashboard
   - Feature Importance

---

## 7. If Load Daily Predictions Times Out

This usually means the backend Render API is sleeping.

Do this:

1. Open the backend URL directly, for example:

```text
https://oluwa-blazee.onrender.com/docs
```

2. Wait until it loads.
3. Return to the Streamlit app.
4. Click **Load Daily Predictions** again.

If it still times out, increase timeout values in `frontend.py` from:

```python
timeout=60
```

to:

```python
timeout=120
```

---

## 8. If Training Rows Shows 0

The app is probably missing `bet_history.csv` on the new deployment.

Fix:

1. Generate Historical Backfill Data.
2. Train Ensemble Model.
3. Run Full Daily Automation.
4. Refresh the page.

The health dashboard should then show real training rows.

---

## 9. If No Saved Model Versions Appear

Run:

```text
Run Full Daily Automation
```

Then refresh the app.

Saved versions should appear as:

```text
ensemble_model_YYYYMMDD_HHMMSS.joblib
```

If they do not appear, check that `model_versioning.py` saves to:

```text
models/ensemble_model_YYYYMMDD_HHMMSS.joblib
```

---

## 10. Model Rollback

If a new model performs badly:

1. Go to **Saved Model Versions**.
2. Select an older model.
3. Click **Restore Selected Model**.

This copies the selected version back to:

```text
models/ensemble_model.joblib
models/current_model.joblib
```

---

## 11. Important Render Storage Warning

Render free services may use temporary storage. Runtime-created files may disappear after redeploys/restarts, including:

```text
bet_history.csv
odds_snapshots.csv
learning_dataset.csv
models/ensemble_model_*.joblib
models/current_model.joblib
```

To protect your work, regularly download or commit backups of:

```text
bet_history.csv
odds_snapshots.csv
models/
```

Future upgrade: move these to PostgreSQL or cloud storage.

---

## 12. Current Stable Baseline

Latest known working baseline:

```text
Training Rows: 500
Win Rate: around 54–56%
ROI: positive
Automation Runner: working
Model Versioning: working
Model Rollback: working
Model Health Dashboard: working
Confidence Engine: working
Ensemble Training: working
```

---

## 13. Next High-Impact Upgrade

The next useful feature is **Real Market Intelligence**:

```text
odds_snapshots.csv
line_movement_diff
steam_move
reverse_line_movement
sharp_support_pct
closing line value
```

Do not add more dashboards until market data quality improves.

