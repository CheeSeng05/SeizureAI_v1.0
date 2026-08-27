from __future__ import annotations

import csv
import mimetypes
import tempfile
import threading
import uuid
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

from sqlalchemy import select
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from fyp_data_platform.config.settings import Settings, load_settings
from fyp_data_platform.db.models import ModelRun, StoredFile
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.db.session import create_engine_and_session
from fyp_data_platform.integrations.seizure_model import SeizureModelIntegration
from fyp_data_platform.storage.factory import create_storage
from fyp_data_platform.storage.integrity import sha256_file
from fyp_data_platform.storage.keys import safe_entity, source_key


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SeizureAI Local</title>
  <style>
    :root {
      --bg: #0f1318;
      --panel: #171d24;
      --panel-2: #111820;
      --line: #2a3440;
      --text: #e6edf3;
      --muted: #8fa1b3;
      --accent: #4f8cff;
      --accent-2: #19b37a;
      --danger: #ef5b5b;
      --warn: #d8a31a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
      line-height: 1.45;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #10161d;
    }
    h1 {
      font-size: 18px;
      margin: 0;
      font-weight: 650;
      letter-spacing: 0;
    }
    .sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 0.9fr) minmax(520px, 1.4fr);
      gap: 18px;
      padding: 18px;
      max-width: 1420px;
      margin: 0 auto;
    }
    section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      overflow: hidden;
    }
    .section-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .section-title { font-size: 13px; font-weight: 650; }
    .section-body { padding: 16px; }
    .wide { grid-column: 1 / -1; }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    input {
      width: 100%;
      min-height: 38px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      outline: none;
    }
    input:focus { border-color: var(--accent); }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .field { margin-bottom: 12px; }
    .drop-zone {
      min-height: 190px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      border: 1px dashed #506171;
      background: #101820;
      border-radius: 8px;
      padding: 20px;
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease;
    }
    .drop-zone.dragging {
      border-color: var(--accent);
      background: #132033;
    }
    .drop-title { font-size: 15px; font-weight: 650; }
    .drop-meta { color: var(--muted); font-size: 12px; margin-top: 8px; }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    button {
      min-height: 36px;
      border: 1px solid var(--line);
      background: #202a35;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .status {
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      color: var(--muted);
      background: #111820;
      min-height: 42px;
      white-space: pre-wrap;
    }
    .outcome-grid {
      display: grid;
      grid-template-columns: minmax(260px, 0.7fr) minmax(420px, 1.3fr);
      gap: 16px;
    }
    .outcome-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 14px;
      min-width: 0;
    }
    .outcome-title {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .outcome-value {
      font-size: 24px;
      font-weight: 750;
      letter-spacing: 0;
    }
    .outcome-value.detected { color: #ff8d8d; }
    .outcome-value.clear { color: #72dfad; }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #101820;
      min-width: 0;
    }
    .stat-label {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 14px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .timeline {
      display: flex;
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #0e141b;
    }
    .timeline-segment {
      min-width: 2px;
      background: #263343;
      border-right: 1px solid rgba(255,255,255,0.05);
    }
    .timeline-segment.active { background: var(--danger); }
    .timeline-segment.hot { background: var(--warn); }
    .timeline-image {
      width: 100%;
      max-height: 260px;
      object-fit: contain;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-top: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 12px;
    }
    th {
      color: var(--muted);
      font-weight: 650;
      background: #121922;
      position: sticky;
      top: 0;
    }
    .table-wrap {
      max-height: 360px;
      overflow: auto;
    }
    .key {
      color: var(--muted);
      max-width: 340px;
      overflow-wrap: anywhere;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #253140;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.ok { background: #10291f; color: #72dfad; }
    .pill.warn { background: #302711; color: #f2c35c; }
    .pill.bad { background: #321919; color: #ff8d8d; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .summary {
      color: var(--muted);
      max-width: 300px;
    }
    .run-row { cursor: pointer; }
    .run-row.selected { background: #142136; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .grid-2 { grid-template-columns: 1fr; }
      .outcome-grid { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SeizureAI Local</h1>
      <div class="sub">MinIO/PostgreSQL upload console</div>
    </div>
    <button onclick="refresh()">Refresh</button>
  </header>

  <main>
    <section>
      <div class="section-head">
        <div class="section-title">Upload Videos</div>
        <span class="pill">Local only</span>
      </div>
      <div class="section-body">
        <div class="grid-2">
          <div class="field">
            <label for="datasetId">Dataset ID</label>
            <input id="datasetId" value="ui-videos">
          </div>
          <div class="field">
            <label for="datasetName">Dataset name</label>
            <input id="datasetName" value="UI uploaded videos">
          </div>
          <div class="field">
            <label for="patientId">Patient ID</label>
            <input id="patientId" value="local-patient-001">
          </div>
          <div class="field">
            <label for="recordingId">Recording ID</label>
            <input id="recordingId" placeholder="Auto from filename">
          </div>
        </div>

        <input id="fileInput" type="file" multiple accept=".mp4,.avi,.mkv,.mov" hidden>
        <div id="dropZone" class="drop-zone">
          <div>
            <div class="drop-title">Drop MP4 videos here</div>
            <div class="drop-meta">Files upload immediately into MinIO and PostgreSQL</div>
          </div>
        </div>
        <div class="actions">
          <button class="primary" onclick="document.getElementById('fileInput').click()">Choose files</button>
          <button onclick="clearStatus()">Clear log</button>
        </div>
        <div id="status" class="status">Ready.</div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div class="section-title">Registered Videos</div>
        <span id="videoCount" class="pill">0</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Recording</th>
              <th>File</th>
              <th>Size</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="videosBody"></tbody>
        </table>
      </div>
    </section>

    <section class="wide">
      <div class="section-head">
        <div class="section-title">Selected Result</div>
        <span id="selectedRunPill" class="pill">None</span>
      </div>
      <div id="resultBody" class="section-body">
        <div class="key">Run a video or select a completed run to view timestamps and probabilities.</div>
      </div>
    </section>

    <section class="wide">
      <div class="section-head">
        <div class="section-title">Model Runs</div>
        <span id="runCount" class="pill">0</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Recording</th>
              <th>Mode</th>
              <th>Run ID</th>
              <th>Results</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody id="runsBody"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const statusBox = document.getElementById('status');
    const jobs = new Set();
    let selectedRunId = null;

    function appendStatus(text) {
      statusBox.textContent = (statusBox.textContent === 'Ready.' ? '' : statusBox.textContent + '\n') + text;
    }

    function clearStatus() {
      statusBox.textContent = 'Ready.';
    }

    function formValue(id) {
      return document.getElementById(id).value.trim();
    }

    function sizeLabel(bytes) {
      if (!bytes) return '-';
      const units = ['B', 'KB', 'MB', 'GB'];
      let value = Number(bytes);
      let index = 0;
      while (value >= 1024 && index < units.length - 1) {
        value /= 1024;
        index += 1;
      }
      return value.toFixed(index === 0 ? 0 : 1) + ' ' + units[index];
    }

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[char]));
    }

    function pct(value) {
      if (value == null || Number.isNaN(Number(value))) return '-';
      return (Number(value) * 100).toFixed(1) + '%';
    }

    function secRange(start, end) {
      return Number(start).toFixed(1) + 's - ' + Number(end).toFixed(1) + 's';
    }

    async function uploadFiles(files) {
      if (!files.length) return;
      const data = new FormData();
      data.append('dataset_id', formValue('datasetId') || 'ui-videos');
      data.append('dataset_name', formValue('datasetName') || 'UI uploaded videos');
      data.append('patient_id', formValue('patientId') || 'local-patient-001');
      data.append('recording_id', formValue('recordingId'));
      for (const file of files) data.append('videos', file);

      appendStatus('Uploading ' + files.length + ' file(s)...');
      const response = await fetch('/api/upload', { method: 'POST', body: data });
      const payload = await response.json();
      if (!response.ok) {
        appendStatus('Upload failed: ' + (payload.error || response.statusText));
        return;
      }
      for (const item of payload.uploaded) {
        appendStatus('Stored ' + item.filename + ' as ' + item.dataset_id + '/' + item.recording_id);
      }
      await refresh();
    }

    async function startRun(datasetId, recordingId) {
      appendStatus('Queued video run for ' + datasetId + '/' + recordingId);
      const data = new FormData();
      data.append('dataset_id', datasetId);
      data.append('recording_id', recordingId);
      data.append('require_labels', 'false');
      const response = await fetch('/api/run', { method: 'POST', body: data });
      const payload = await response.json();
      if (!response.ok) {
        appendStatus('Run failed to start: ' + (payload.error || response.statusText));
        return;
      }
      jobs.add(payload.job_id);
      appendStatus('Started job ' + payload.job_id);
      pollJob(payload.job_id);
      await refresh();
    }

    async function pollJob(jobId) {
      const response = await fetch('/api/jobs/' + jobId);
      const payload = await response.json();
      if (payload.status === 'completed') {
        appendStatus('Job completed: ' + payload.result.run_id);
        selectedRunId = payload.result.run_id;
        jobs.delete(jobId);
        await refresh();
        await loadResult(selectedRunId);
        return;
      }
      if (payload.status === 'failed') {
        appendStatus('Job failed: ' + payload.error);
        jobs.delete(jobId);
        await refresh();
        return;
      }
      setTimeout(() => pollJob(jobId), 4000);
    }

    function statusPill(status) {
      const cls = status === 'completed' ? 'ok' : status === 'failed' || status === 'blocked' ? 'bad' : 'warn';
      return '<span class="pill ' + cls + '">' + status + '</span>';
    }

    function metricsSummary(item) {
      if (item.error) return 'Error: ' + item.error;
      if (!item.outcome) return 'No metrics yet';
      if (item.outcome.detected) return 'Seizure detected, events: ' + item.outcome.event_count;
      return 'No seizure detected';
    }

    function outcomePill(outcome) {
      if (!outcome) return '<span class="pill">Pending</span>';
      if (outcome.detected) return '<span class="pill bad">Detected</span>';
      return '<span class="pill ok">No seizure</span>';
    }

    async function loadResult(runId) {
      selectedRunId = runId;
      const response = await fetch('/api/runs/' + runId + '/results');
      const payload = await response.json();
      if (!response.ok) {
        document.getElementById('resultBody').innerHTML = '<div class="key">' + esc(payload.error || response.statusText) + '</div>';
        document.getElementById('selectedRunPill').textContent = 'Error';
        return;
      }
      renderResult(payload);
      await refresh();
    }

    function renderResult(result) {
      const outcome = result.outcome || {};
      const detected = Boolean(outcome.detected);
      const title = detected ? 'Seizure detected' : 'No seizure detected';
      const valueClass = detected ? 'detected' : 'clear';
      document.getElementById('selectedRunPill').innerHTML = outcomePill(outcome);

      const eventRows = (outcome.events || []).map(event => `
        <tr>
          <td>${secRange(event.start_sec, event.end_sec)}</td>
          <td>${pct(event.score)}</td>
        </tr>
      `).join('') || '<tr><td colspan="2" class="key">No positive event windows.</td></tr>';

      const windowRows = (result.windows || []).slice(0, 20).map(row => `
        <tr>
          <td>${secRange(row.start_sec, row.end_sec)}</td>
          <td>${pct(row.probability)}</td>
          <td>${row.prediction ? '<span class="pill bad">1</span>' : '<span class="pill ok">0</span>'}</td>
        </tr>
      `).join('') || '<tr><td colspan="3" class="key">No probability windows found.</td></tr>';

      const timelineSegments = (result.windows || []).map(row => {
        const width = Math.max(2, Number(row.duration_sec || 1));
        const cls = row.prediction ? 'active' : Number(row.probability) >= 0.35 ? 'hot' : '';
        return '<div class="timeline-segment ' + cls + '" style="flex:' + width + '" title="' + esc(secRange(row.start_sec, row.end_sec) + ' ' + pct(row.probability)) + '"></div>';
      }).join('');

      const image = result.timeline_image
        ? '<img class="timeline-image" src="' + esc(result.timeline_image.url) + '" alt="Prediction timeline plot">'
        : '';

      document.getElementById('resultBody').innerHTML = `
        <div class="outcome-grid">
          <div class="outcome-card">
            <div class="outcome-title">${esc(result.dataset_id || '-')} / ${esc(result.recording_id || '-')}</div>
            <div class="outcome-value ${valueClass}">${title}</div>
            <div class="stats">
              <div class="stat"><div class="stat-label">Events</div><div class="stat-value">${outcome.event_count ?? 0}</div></div>
              <div class="stat"><div class="stat-label">Max probability</div><div class="stat-value">${pct(outcome.max_probability)}</div></div>
              <div class="stat"><div class="stat-label">Duration</div><div class="stat-value">${result.duration_sec ? Number(result.duration_sec).toFixed(1) + 's' : '-'}</div></div>
            </div>
            <div class="status">${esc(outcome.note || '')}</div>
          </div>
          <div class="outcome-card">
            <div class="outcome-title">Probability timeline</div>
            <div class="timeline">${timelineSegments || ''}</div>
            ${image}
          </div>
        </div>
        <div class="grid-2" style="margin-top:16px">
          <div class="outcome-card">
            <div class="outcome-title">Detected timestamps</div>
            <div class="table-wrap">
              <table><thead><tr><th>Time</th><th>Score</th></tr></thead><tbody>${eventRows}</tbody></table>
            </div>
          </div>
          <div class="outcome-card">
            <div class="outcome-title">Prediction windows</div>
            <div class="table-wrap">
              <table><thead><tr><th>Time</th><th>Probability</th><th>Prediction</th></tr></thead><tbody>${windowRows}</tbody></table>
            </div>
          </div>
        </div>
      `;
    }

    async function refresh() {
      const response = await fetch('/api/catalog');
      const payload = await response.json();
      document.getElementById('videoCount').textContent = String(payload.videos.length);
      document.getElementById('runCount').textContent = String(payload.runs.length);

      const videosBody = document.getElementById('videosBody');
      videosBody.innerHTML = payload.videos.map(item => `
        <tr>
          <td class="mono">${esc(item.dataset_id)}</td>
          <td class="mono">${esc(item.recording_source_id)}</td>
          <td><div>${esc(item.filename)}</div><div class="key">${esc(item.storage_key)}</div></td>
          <td>${sizeLabel(item.size_bytes)}</td>
          <td><button onclick="startRun('${esc(item.dataset_id)}', '${esc(item.recording_source_id)}')">Run</button></td>
        </tr>
      `).join('') || '<tr><td colspan="5" class="key">No videos registered yet.</td></tr>';

      const runsBody = document.getElementById('runsBody');
      runsBody.innerHTML = payload.runs.map(item => `
        <tr class="run-row ${item.id === selectedRunId ? 'selected' : ''}" onclick="loadResult('${esc(item.id)}')">
          <td>${statusPill(item.status)}</td>
          <td><div class="mono">${esc(item.dataset_id || '-')}</div><div class="mono key">${esc(item.recording_id || '-')}</div></td>
          <td>${esc(item.modality || '-')}</td>
          <td class="mono key">${esc(item.id)}</td>
          <td><div>${outcomePill(item.outcome)} ${item.artifact_count} artifact(s)</div><div class="key">${esc(item.output_prefix || '-')}</div><div class="summary">${esc(metricsSummary(item))}</div></td>
          <td>${esc(item.started_at || '-')}</td>
        </tr>
      `).join('') || '<tr><td colspan="6" class="key">No model runs yet.</td></tr>';

      if (!selectedRunId && payload.runs.length && payload.runs[0].status === 'completed') {
        loadResult(payload.runs[0].id);
      }
    }

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', event => {
      event.preventDefault();
      dropZone.classList.add('dragging');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
    dropZone.addEventListener('drop', event => {
      event.preventDefault();
      dropZone.classList.remove('dragging');
      uploadFiles(event.dataTransfer.files);
    });
    fileInput.addEventListener('change', event => uploadFiles(event.target.files));

    refresh();
    setInterval(refresh, 10000);
  </script>
</body>
</html>
"""


def _recording_source_id(recording_id: str, file_stem: str, count: int) -> str:
    stem = safe_entity(file_stem)
    if recording_id and count == 1:
        return safe_entity(recording_id)
    if recording_id:
        return f"{safe_entity(recording_id)}-{stem}"
    return stem


def _register_video(
    *,
    settings: Settings,
    storage: Any,
    session_factory: Any,
    upload: FileStorage,
    dataset_id: str,
    dataset_name: str,
    patient_id: str,
    recording_id: str,
    file_count: int,
) -> dict[str, Any]:
    filename = secure_filename(upload.filename or "")
    if not filename:
        raise ValueError("Uploaded file is missing a filename.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video file extension: {suffix}")

    dataset_id = safe_entity(dataset_id or "ui-videos")
    dataset_name = dataset_name or "UI uploaded videos"
    patient_id = safe_entity(patient_id or "local-patient-001")
    source_recording_id = _recording_source_id(recording_id, Path(filename).stem, file_count)
    relative = f"manual/{patient_id}/{source_recording_id}/{filename}"
    key = source_key(dataset_id, relative)

    with tempfile.TemporaryDirectory(prefix="fyp-ui-upload-") as directory:
        temporary = Path(directory) / filename
        upload.save(temporary)
        checksum = sha256_file(temporary)
        metadata = storage.put_file(temporary, key, sha256=checksum)

    with session_factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset(
            dataset_id,
            name=dataset_name,
            version=None,
            description="Local UI uploaded videos",
            source_url=None,
            license=None,
            metadata_json={"uploaded_via": "local_ui"},
        )
        patient, _ = repo.patient(dataset_id, patient_id, metadata_json={"uploaded_via": "local_ui"})
        recording, _ = repo.recording(
            dataset_id,
            source_recording_id,
            patient_id=patient.id,
            session_id=None,
            task=None,
            modality="video",
            duration_sec=None,
            video_fps=None,
            start_datetime=None,
            metadata_json={"uploaded_via": "local_ui"},
        )
        file_row, _ = repo.stored_file(
            dataset_id,
            relative,
            recording_id=recording.id,
            role="source-media",
            modality="video",
            storage_backend=settings.storage_backend,
            storage_key=key,
            payload_state="stored",
            size_bytes=metadata.size_bytes,
            sha256=metadata.sha256,
            media_type=None,
            metadata_json={"uploaded_via": "local_ui", "original_filename": filename},
        )

    return {
        "file_id": file_row.id,
        "dataset_id": dataset_id,
        "patient_id": patient_id,
        "recording_id": source_recording_id,
        "filename": filename,
        "storage_key": key,
        "sha256": metadata.sha256,
        "size_bytes": metadata.size_bytes,
    }


def _metric_documents(row: ModelRun) -> list[dict[str, Any]]:
    payload = row.metrics_json or {}
    documents = []
    for value in (payload.get("metrics") or {}).values():
        if isinstance(value, dict):
            documents.append(value)
    return documents


def _metric_body(document: dict[str, Any]) -> dict[str, Any]:
    nested = document.get("metrics")
    return nested if isinstance(nested, dict) else document


def _best_metric_document(row: ModelRun) -> dict[str, Any]:
    documents = _metric_documents(row)
    for document in documents:
        body = _metric_body(document)
        if "predicted_event_count" in body or "predicted_events" in body:
            return document
    return documents[0] if documents else {}


def _artifacts(row: ModelRun) -> list[dict[str, Any]]:
    payload = row.metrics_json or {}
    return [item for item in payload.get("artifacts", []) if isinstance(item, dict)]


def _find_artifact(row: ModelRun, *suffixes: str) -> dict[str, Any] | None:
    artifacts = _artifacts(row)
    for suffix in suffixes:
        for artifact in artifacts:
            relative = str(artifact.get("relative_path", ""))
            if relative == suffix or relative.endswith(f"/{suffix}"):
                return artifact
    return None


def _download_artifact_bytes(storage: Any, artifact: dict[str, Any]) -> bytes:
    key = str(artifact.get("key", ""))
    if not key:
        raise LookupError("Artifact does not have a storage key.")
    suffix = Path(str(artifact.get("relative_path", "artifact"))).suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="fyp-ui-artifact-") as directory:
        destination = Path(directory) / f"artifact{suffix}"
        storage.download_file(key, destination)
        return destination.read_bytes()


def _read_artifact_text(storage: Any, artifact: dict[str, Any]) -> str:
    return _download_artifact_bytes(storage, artifact).decode("utf-8-sig")


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _prediction_windows(storage: Any, row: ModelRun) -> list[dict[str, Any]]:
    artifact = _find_artifact(row, "video_only/predictions.csv", "predictions.csv")
    if artifact is None:
        artifact = _find_artifact(row, "video_only/probability_timeline.csv", "probability_timeline.csv")
    if artifact is None:
        return []

    text = _read_artifact_text(storage, artifact)
    windows = []
    for csv_row in csv.DictReader(StringIO(text)):
        start_sec = _float_or_none(csv_row.get("start_sec"))
        end_sec = _float_or_none(csv_row.get("end_sec"))
        probability = _float_or_none(
            csv_row.get("video_probability")
            or csv_row.get("probability")
            or csv_row.get("seizure_probability")
        )
        prediction = _int_or_zero(
            csv_row.get("video_prediction")
            or csv_row.get("prediction")
            or csv_row.get("predicted_label")
        )
        if start_sec is None or end_sec is None:
            continue
        windows.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": max(end_sec - start_sec, 0.0),
                "probability": probability,
                "prediction": prediction,
            }
        )
    return windows


def _events_from_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in windows:
        if not row.get("prediction"):
            if current is not None:
                events.append(current)
                current = None
            continue
        probability = row.get("probability") or 0.0
        if current is None:
            current = {
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "score": probability,
            }
        else:
            current["end_sec"] = row["end_sec"]
            current["score"] = max(float(current.get("score") or 0.0), float(probability))
    if current is not None:
        events.append(current)
    return events


def _outcome(row: ModelRun, windows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    if row.status != "completed":
        return None
    windows = windows or []
    document = _best_metric_document(row)
    body = _metric_body(document)
    metric_events = body.get("predicted_events") if isinstance(body.get("predicted_events"), list) else []
    events = [
        {
            "start_sec": _float_or_none(event.get("start_sec")) or 0.0,
            "end_sec": _float_or_none(event.get("end_sec")) or 0.0,
            "score": _float_or_none(event.get("score")) or 0.0,
        }
        for event in metric_events
        if isinstance(event, dict)
    ]
    if not events:
        events = _events_from_windows(windows)
    event_count = _int_or_zero(body.get("predicted_event_count")) if "predicted_event_count" in body else len(events)
    detected = event_count > 0 or any(row.get("prediction") for row in windows)
    max_probability = max(
        (float(row["probability"]) for row in windows if row.get("probability") is not None),
        default=None,
    )
    prediction_only = bool(body.get("prediction_only"))
    note = (
        "Prediction-only run. Accuracy is not calculated until a human label file is supplied."
        if prediction_only
        else "Evaluation run. Accuracy metrics are available because labels were supplied."
    )
    return {
        "detected": detected,
        "event_count": event_count if event_count else len(events),
        "events": events,
        "max_probability": max_probability,
        "prediction_only": prediction_only,
        "note": note,
    }


def _run_result(storage: Any, row: ModelRun) -> dict[str, Any]:
    windows = _prediction_windows(storage, row)
    document = _best_metric_document(row)
    run_info = document.get("run", {}) if isinstance(document.get("run"), dict) else {}
    video_info = run_info.get("video", {}) if isinstance(run_info.get("video"), dict) else {}
    timeline = _find_artifact(row, "video_only/timeline.png", "timeline.png")
    return {
        "run_id": row.id,
        "status": row.status,
        "modality": row.modality,
        "dataset_id": (row.config_json or {}).get("dataset_id"),
        "recording_id": (row.config_json or {}).get("recording_id"),
        "duration_sec": video_info.get("duration_sec") or (
            max((item["end_sec"] for item in windows), default=None)
        ),
        "outcome": _outcome(row, windows),
        "windows": windows,
        "timeline_image": (
            {
                "relative_path": timeline.get("relative_path"),
                "url": f"/api/runs/{row.id}/artifact/{timeline.get('relative_path')}",
            }
            if timeline
            else None
        ),
    }


def create_app(config_path: str | None = None) -> Any:
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as error:
        raise RuntimeError(
            "The local UI requires Flask. Install it with: python -m pip install -e \".[ui]\""
        ) from error

    settings = load_settings(config_path)
    engine, session_factory = create_engine_and_session(settings.database_url)
    storage = create_storage(settings)
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024

    @app.get("/")
    def index() -> Any:
        return Response(HTML, mimetype="text/html")

    @app.get("/api/catalog")
    def catalog() -> Any:
        with session_factory() as session:
            videos = session.scalars(
                select(StoredFile)
                .where(StoredFile.modality == "video", StoredFile.payload_state == "stored")
                .order_by(StoredFile.created_at.desc())
                .limit(100)
            ).all()
            runs = session.scalars(
                select(ModelRun).order_by(ModelRun.started_at.desc()).limit(50)
            ).all()
        video_rows = []
        for row in videos:
            recording_source_id = ""
            if row.recording_id and "::" in row.recording_id:
                recording_source_id = row.recording_id.split("::", 1)[1]
            video_rows.append(
                {
                    "id": row.id,
                    "dataset_id": row.dataset_id,
                    "recording_id": row.recording_id,
                    "recording_source_id": recording_source_id,
                    "filename": Path(row.source_relative_path).name,
                    "storage_key": row.storage_key,
                    "size_bytes": row.size_bytes,
                    "sha256": row.sha256,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            )
        run_rows = []
        for row in runs:
            config_json = row.config_json or {}
            metrics_json = row.metrics_json or {}
            artifacts = metrics_json.get("artifacts") or []
            run_rows.append(
                {
                    "id": row.id,
                    "status": row.status,
                    "modality": row.modality,
                    "dataset_id": config_json.get("dataset_id"),
                    "recording_id": config_json.get("recording_id"),
                    "output_prefix": str(artifacts[0]["key"]).rsplit("/", 1)[0] if artifacts else None,
                    "artifact_count": len(artifacts),
                    "outcome": _outcome(row),
                    "error": metrics_json.get("error"),
                    "started_at": row.started_at.isoformat() if row.started_at else None,
                    "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                }
            )
        return jsonify({"videos": video_rows, "runs": run_rows})

    @app.get("/api/runs/<run_id>/results")
    def run_results(run_id: str) -> Any:
        with session_factory() as session:
            row = session.get(ModelRun, run_id)
            if row is None:
                return jsonify({"error": "Unknown model run."}), 404
            try:
                return jsonify(_run_result(storage, row))
            except Exception as error:
                return jsonify({"error": str(error)}), 500

    @app.get("/api/runs/<run_id>/artifact/<path:relative_path>")
    def run_artifact(run_id: str, relative_path: str) -> Any:
        with session_factory() as session:
            row = session.get(ModelRun, run_id)
            if row is None:
                return jsonify({"error": "Unknown model run."}), 404
            artifact = next(
                (
                    item
                    for item in _artifacts(row)
                    if str(item.get("relative_path")) == relative_path
                ),
                None,
            )
            if artifact is None:
                return jsonify({"error": "Unknown artifact."}), 404
            try:
                body = _download_artifact_bytes(storage, artifact)
            except Exception as error:
                return jsonify({"error": str(error)}), 500
        mimetype = mimetypes.guess_type(relative_path)[0] or "application/octet-stream"
        return Response(body, mimetype=mimetype)

    @app.post("/api/upload")
    def upload() -> Any:
        files = [item for item in request.files.getlist("videos") if item.filename]
        if not files:
            return jsonify({"error": "No video files were uploaded."}), 400
        try:
            uploaded = [
                _register_video(
                    settings=settings,
                    storage=storage,
                    session_factory=session_factory,
                    upload=file_item,
                    dataset_id=request.form.get("dataset_id", "ui-videos"),
                    dataset_name=request.form.get("dataset_name", "UI uploaded videos"),
                    patient_id=request.form.get("patient_id", "local-patient-001"),
                    recording_id=request.form.get("recording_id", ""),
                    file_count=len(files),
                )
                for file_item in files
            ]
        except Exception as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({"uploaded": uploaded})

    @app.post("/api/run")
    def run_model() -> Any:
        dataset_id = request.form.get("dataset_id", "").strip()
        recording_id = request.form.get("recording_id", "").strip()
        require_labels = request.form.get("require_labels", "false").lower() == "true"
        if not dataset_id or not recording_id:
            return jsonify({"error": "dataset_id and recording_id are required."}), 400
        job_id = str(uuid.uuid4())
        with jobs_lock:
            jobs[job_id] = {"status": "running", "dataset_id": dataset_id, "recording_id": recording_id}

        def worker() -> None:
            try:
                result = SeizureModelIntegration(settings, storage, session_factory).run(
                    dataset_id,
                    recording_id,
                    mode="video",
                    require_labels=require_labels,
                )
                payload = asdict(result)
                with jobs_lock:
                    jobs[job_id] = {"status": "completed", "result": payload}
            except Exception as error:
                with jobs_lock:
                    jobs[job_id] = {"status": "failed", "error": str(error)}

        threading.Thread(target=worker, name=f"model-run-{job_id}", daemon=True).start()
        return jsonify({"job_id": job_id, "status": "running"})

    @app.get("/api/jobs/<job_id>")
    def job(job_id: str) -> Any:
        with jobs_lock:
            payload = jobs.get(job_id)
        if payload is None:
            return jsonify({"error": "Unknown job."}), 404
        return jsonify(payload)

    @app.teardown_appcontext
    def _shutdown(_: BaseException | None = None) -> None:
        # The engine is intentionally kept open for the process lifetime.
        _ = engine

    return app


def run_ui(*, config_path: str | None, host: str, port: int, debug: bool) -> None:
    app = create_app(config_path)
    print(f"Local UI running at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
