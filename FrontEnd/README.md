# SeizureAI — React dashboard

Vite + React + Tailwind frontend for the FastAPI analysis service.

## Layout

```
frontend/
  src/
    api/client.js          the only module that knows the API exists
    hooks/useApi.js        fetching + debounced scoring
    components/
      Layout.jsx           sidebar, header, router outlet
      Timeline.jsx         per-window probability bars
      ui.jsx               Card, Notice, Stat, Loading, formatters
    pages/
      Dashboard.jsx        overview + test matrix
      Upload.jsx           queue a recording
      Historical.jsx       past analyses, filterable
      Results.jsx          timeline + live threshold controls
      Metrics.jsx          evaluation metrics per recording
      Settings.jsx         provenance and configuration
      Login.jsx            placeholder gate
    App.jsx                routes
    main.jsx               entry
  vite.config.js           dev proxy for /api
```

## Where the arithmetic lives

**The frontend computes no metrics.** Every number comes from the analysis API,
which holds the tested implementation in `scoring.py`. The browser decides which
timeline bars to colour and nothing else.

That boundary matters because a threshold is a clinical decision: the value on
screen must be the value the backend would put in a report. A second
implementation in JavaScript could silently drift from the first.

## Running

```powershell
# terminal 1 — analysis API
cd backend
$env:SEIZUREAI_RESULTS_DIR = "..\FYPModels\results"
uvicorn app.main:app --reload

# terminal 2 — dashboard
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

Vite proxies `/api` to `http://localhost:8000` (see `vite.config.js`), so the
browser only ever talks to the Vite origin. No CORS grant is needed and the API
can stay bound to localhost. Override the target with `VITE_API_URL`.

## The Results page

The two sliders call `POST /api/recordings/{id}/score`. Requests are debounced
by 120 ms and tagged with a sequence number in `useScoring`, so a slow response
cannot overwrite a newer one mid-drag — without that, the metrics on screen
could disagree with the slider position.

## Empty states are deliberate

A statistic with no value renders as an em-dash, never as `0.000`. On a clinical
dashboard "we measured zero" and "we did not measure this" must not look the
same. `Stat` takes an explicit `unavailable` prop for that reason, and the
Dashboard's test matrix reports blocked and skipped tests rather than showing
only the one that completed.

## Build

```powershell
npm run build      # -> dist/
npm run preview    # serve the build locally
```

Note that `preview` does not proxy `/api`; serve `dist/` behind the same origin
as the API, or put both behind one reverse proxy.
