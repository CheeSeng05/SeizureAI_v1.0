// The only module that knows the API exists. Everything else calls these
// functions, so changing transport or adding auth touches one file.
//
// Paths are relative: Vite proxies /api in dev (vite.config.js), and in
// production the app sits behind the same origin or a reverse proxy. Either
// way the browser needs no CORS grant and the API can stay on localhost.
const BASE = import.meta.env.VITE_API_BASE ?? '';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch {
    throw new ApiError(0, 'Analysis API unreachable. Is uvicorn running?');
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json();
}

export const api = {
  health: () => request('/api/health'),
  recordings: () => request('/api/recordings'),
  recording: (id) => request(`/api/recordings/${id}`),
  timeline: (id) => request(`/api/recordings/${id}/timeline`),
  provenance: (id) => request(`/api/recordings/${id}/provenance`),
  testMatrix: () => request('/api/test-matrix'),
  score: (id, threshold, minConsecutive) =>
    request(`/api/recordings/${id}/score`, {
      method: 'POST',
      body: JSON.stringify({ threshold, min_consecutive: minConsecutive }),
    }),

  uploads: () => request('/api/uploads'),
  deleteUpload: (id) => fetch(`${BASE}/api/uploads/${id}`, { method: 'DELETE' }),
  analyse: (id) => request(`/api/uploads/${id}/analyse`, { method: 'POST' }),
  job: (id) => request(`/api/jobs/${id}`),
  jobs: () => request('/api/jobs'),
  cancelJob: (id) => request(`/api/jobs/${id}/cancel`, { method: 'POST' }),

  /**
   * Upload with progress. Uses XMLHttpRequest rather than fetch because fetch
   * still cannot report upload progress — and these files are gigabytes, so a
   * progress bar is not decoration.
   */
  upload(file, fields, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append('file', file);
      Object.entries(fields).forEach(([k, v]) => {
        if (v !== '' && v !== null && v !== undefined) form.append(k, v);
      });

      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${BASE}/api/uploads`);
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) onProgress?.(e.loaded / e.total, e.loaded, e.total);
      });
      xhr.addEventListener('load', () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new ApiError(xhr.status, body.detail ?? xhr.statusText));
      });
      xhr.addEventListener('error', () =>
        reject(new ApiError(0, 'Upload failed — is the analysis API running?')));
      xhr.addEventListener('abort', () => reject(new ApiError(0, 'Upload cancelled')));
      xhr.send(form);
      resolve.xhr = xhr;
    });
  },
};