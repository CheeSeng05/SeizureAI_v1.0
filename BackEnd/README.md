# SeizureAI dashboard

Frontend and backend for viewing VSViG inference results.

```
seizureai/
  backend/                  FastAPI - reads results/, serves JSON
    app/
      main.py               app factory, CORS, routers
      config.py             SEIZUREAI_RESULTS_DIR
      dependencies.py       DI wiring
      models.py             pydantic schemas = the API contract
      routers/
        recordings.py       GET  metadata, timeline, provenance
        analysis.py         POST scoring, GET test matrix
      services/
        results_store.py    the only module that touches the filesystem
        scoring.py          threshold/persistence metrics
    tests/
      test_scoring.py       regression against the recorded run
  frontend/                 React - renders, never computes metrics
    src/
      api/client.js         the only module that knows the API exists
      hooks/useApi.js       fetching + debounced scoring
      components/ pages/
```

## Why this split

The model repo runs inference and writes `results/`. This backend only reads
those files, so the two stay independently testable and the dashboard cannot
silently disagree with `results/video_only/*_metrics.json`.

**All metric computation lives in `scoring.py`.** The frontend renders numbers
it is given and calculates none of its own. That matters because a threshold
slider is a clinical decision: the value shown on screen must be the value the
backend would put in a report. `tests/test_scoring.py` pins the output against
the recorded run, so a refactor that changes any metric fails the build.

## Run

```powershell
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SEIZUREAI_RESULTS_DIR = "C:\...\FYPModels\results"
uvicorn app.main:app --reload
```

Check `http://localhost:8000/api/health` — it reports which recordings were
found. Interactive docs at `/docs`.

```powershell
cd frontend
npm install
npm run dev
```

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | results dir, recordings found |
| GET | `/api/recordings` | all recordings with metadata |
| GET | `/api/recordings/{id}` | one recording |
| GET | `/api/recordings/{id}/timeline` | per-window probabilities |
| GET | `/api/recordings/{id}/provenance` | checkpoints, test mode, training flags |
| POST | `/api/recordings/{id}/score` | metrics at a threshold and persistence |
| GET | `/api/test-matrix` | what ran, what was blocked, and why |

`/api/test-matrix` is deliberately first-class. A dashboard that shows only the
completed test implies the others succeeded.

## Adding recordings

Rerun inference in the model repo:

```powershell
python -m seizure_multimodal test --mode video
```

New files appear under `results/video_only/` and the API serves them on restart.
No frontend change, nothing to re-embed.

## Tests

```powershell
cd backend && python -m pytest -q
```

Covers metric reproduction against the recorded run, early-onset detection,
persistence suppression, transition-window exclusion, and the guarantee that a
seizure offset is never invented when the annotation lacks one.
