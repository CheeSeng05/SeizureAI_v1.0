export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError(0, 'The SeizureAI service is unreachable.');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(response.status, body.error || response.statusText);
  return body;
}

export const api = {
  health: () => request('/api/health'),
  catalog: () => request('/api/catalog'),
  run: (id) => request(`/api/runs/${id}`),
  startRun: (datasetId, recordingId, mode = 'video', requireLabels = false) => request('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, recording_id: recordingId, mode, require_labels: requireLabels }),
  }),
  job: (id) => request(`/api/jobs/${id}`),
  upload: (mode, files, eeg, fields, labels = null) => {
    const form = new FormData();
    form.append('mode', mode);
    files.forEach((file) => form.append('videos', file));
    if (eeg) form.append('eeg', eeg);
    Object.entries(fields).forEach(([key, value]) => form.append(key, value));
    if (labels) form.append('labels', labels);
    return request('/api/uploads', { method: 'POST', body: form });
  },
};
