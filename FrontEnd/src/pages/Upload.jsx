import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle2, FileVideo, Play, Trash2, Upload as UploadIcon, X } from 'lucide-react';
import { api } from '../api/client';
import { Card, Notice } from '../components/ui';
import JobProgress from '../components/JobProgress';

const GB = 1024 ** 3;
const fmtSize = (b) => (b >= GB ? `${(b / GB).toFixed(2)} GB` : `${(b / 1024 ** 2).toFixed(0)} MB`);

export default function Upload() {
  const navigate = useNavigate();
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [meta, setMeta] = useState({
    patient_id: '', seizure_id: '', seizure_type: '',
    eeg_onset_sec: '', clinical_onset_sec: '',
  });
  const [progress, setProgress] = useState(null);
  const [uploaded, setUploaded] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);

  // Poll while a job is in flight. Cleared as soon as it settles, so a
  // finished run stops hitting the API.
  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return undefined;
    const t = setInterval(() => {
      api.job(job.job_id).then(setJob).catch((e) => setError(e));
    }, 2000);
    return () => clearInterval(t);
  }, [job]);

  const pick = (f) => {
    setError(null);
    setUploaded(null);
    setJob(null);
    setProgress(null);
    setFile(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) pick(f);
  };

  const doUpload = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const rec = await api.upload(file, meta, (frac) => setProgress(frac));
      setUploaded(rec);
      setProgress(1);
    } catch (e) {
      setError(e);
      setProgress(null);
    }
  };

  const doAnalyse = async () => {
    setError(null);
    try {
      setJob(await api.analyse(uploaded.upload_id));
    } catch (e) {
      setError(e);
    }
  };

  const reset = () => {
    setFile(null); setUploaded(null); setJob(null); setProgress(null); setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const scorable = meta.eeg_onset_sec !== '' && meta.clinical_onset_sec !== '';
  const busy = progress !== null && progress < 1;
  const running = job && ['queued', 'running'].includes(job.status);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">Upload recording</h2>
        <p className="mt-1 text-sm text-slate-500">
          Stage a recording, then run VSViG inference on it
        </p>
      </div>

      {error && (
        <Notice tone="red" title="Something went wrong">{error.message}</Notice>
      )}

      {/* -- drop zone ------------------------------------------------- */}
      <Card>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`cursor-pointer rounded-xl border-2 border-dashed py-12 text-center transition ${
            dragging ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:border-slate-300'
          }`}
        >
          <input
            ref={inputRef} type="file" className="hidden"
            accept=".mp4,.avi,.mkv,.mov,.mpg,.mpeg,.wmv"
            onChange={(e) => e.target.files?.[0] && pick(e.target.files[0])}
          />
          {file ? (
            <div className="flex items-center justify-center gap-3">
              <FileVideo className="h-8 w-8 text-blue-500" />
              <div className="text-left">
                <div className="font-medium text-slate-800">{file.name}</div>
                <div className="text-xs text-slate-500">{fmtSize(file.size)}</div>
              </div>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); reset(); }}
                className="ml-2 text-slate-400 hover:text-slate-600"
                aria-label="Clear selection"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <>
              <UploadIcon className="mx-auto mb-3 h-10 w-10 text-sky-400" />
              <div className="font-medium text-slate-700">Drop a recording here, or click to browse</div>
              <div className="mt-2 text-xs text-slate-400">
                .mp4, .avi, .mkv · faces must already be masked
              </div>
            </>
          )}
        </div>

        {progress !== null && (
          <div className="mt-4">
            <div className="mb-1.5 flex justify-between text-xs text-slate-600">
              <span>{progress < 1 ? 'Uploading' : 'Uploaded'}</span>
              <span className="font-mono">{(progress * 100).toFixed(0)}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-full rounded-full transition-all ${progress < 1 ? 'bg-blue-500' : 'bg-green-500'}`}
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>
        )}
      </Card>

      {/* -- metadata + actions ---------------------------------------- */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card label="Recording details">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {[
                ['patient_id', 'Patient ID', 'e.g. Pat01'],
                ['seizure_id', 'Seizure ID', 'e.g. Sz2'],
              ].map(([k, label, ph]) => (
                <div key={k}>
                  <label htmlFor={k} className="mb-1.5 block text-sm text-slate-600">{label}</label>
                  <input
                    id={k} value={meta[k]} placeholder={ph} disabled={!!uploaded}
                    onChange={(e) => setMeta({ ...meta, [k]: e.target.value })}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm disabled:bg-slate-50"
                  />
                </div>
              ))}
            </div>

            <div>
              <label htmlFor="seizure_type" className="mb-1.5 block text-sm text-slate-600">
                Seizure type
              </label>
              <select
                id="seizure_type" value={meta.seizure_type} disabled={!!uploaded}
                onChange={(e) => setMeta({ ...meta, seizure_type: e.target.value })}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm disabled:bg-slate-50"
              >
                <option value="">Not specified</option>
                <option value="P">P — focal</option>
                <option value="PG">PG — partial-to-generalised</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                ['eeg_onset_sec', 'EEG onset (s)', '624'],
                ['clinical_onset_sec', 'Clinical onset (s)', '640'],
              ].map(([k, label, ph]) => (
                <div key={k}>
                  <label htmlFor={k} className="mb-1.5 block text-sm text-slate-600">{label}</label>
                  <input
                    id={k} type="number" step="0.1" value={meta[k]} placeholder={ph}
                    disabled={!!uploaded}
                    onChange={(e) => setMeta({ ...meta, [k]: e.target.value })}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm disabled:bg-slate-50"
                  />
                </div>
              ))}
            </div>

            {!scorable && (
              <Notice tone="amber">
                Without both onsets the run produces probabilities but no metrics — there is nothing
                to score predictions against. Take them from Label.xlsx.
              </Notice>
            )}
          </div>
        </Card>

        <Card label="Analysis">
          <div className="space-y-4">
            <div>
              <span className="mb-1.5 block text-sm text-slate-600">Model</span>
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                VSViG-base · frozen, inference only
              </div>
            </div>

            <div>
              <span className="mb-2 block text-sm text-slate-600">Modalities</span>
              <div className="flex flex-wrap gap-2">
                <span className="rounded-md bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700">
                  ✓ Body pose
                </span>
                <span className="rounded-md bg-slate-100 px-2.5 py-1 text-xs text-slate-400">
                  Facial AUs — faces masked
                </span>
                <span className="rounded-md bg-slate-100 px-2.5 py-1 text-xs text-slate-400">
                  EEG — checkpoint missing
                </span>
              </div>
            </div>

            {!uploaded ? (
              <button
                type="button" onClick={doUpload} disabled={!file || busy}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 py-2.5 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-500"
              >
                <UploadIcon className="h-4 w-4" />
                {busy ? 'Uploading…' : 'Upload'}
              </button>
            ) : (
              <>
                <div className="flex items-start gap-2 rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-800">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
                  <div>
                    <div className="font-medium">Staged</div>
                    <div className="mt-0.5 font-mono text-[11px] break-all">
                      {uploaded.sha256.slice(0, 24)}…
                    </div>
                  </div>
                </div>

                <button
                  type="button" onClick={doAnalyse} disabled={running}
                  className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 py-2.5 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-500"
                >
                  <Play className="h-4 w-4" />
                  {running ? 'Analysing…' : 'Run inference'}
                </button>
              </>
            )}
          </div>
        </Card>
      </div>

      {job && (
        <Card label="Inference">
          <JobProgress
            job={job}
            onCancel={() => api.cancelJob(job.job_id)
              .then(() => api.job(job.job_id).then(setJob))
              .catch(setError)}
          />
          {job.status === 'succeeded' && (
            <button
              type="button" onClick={() => navigate('/historical')}
              className="mt-4 w-full rounded-lg bg-blue-600 py-2.5 text-sm font-medium text-white hover:bg-blue-700"
            >
              View results
            </button>
          )}
        </Card>
      )}

      <Notice tone="slate">
        Inference is invoked in the model repository rather than reimplemented here, so a run through
        this page produces exactly what the documented command would.
      </Notice>
    </div>
  );
}