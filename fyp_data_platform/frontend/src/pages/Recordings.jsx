import { useState } from 'react';
import { Play, RefreshCw, Search } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Card, Empty, Loading, Notice, formatBytes, formatDate, formatDuration } from '../components';

export default function Recordings() {
  const catalog = useResource(api.catalog, []); const [query, setQuery] = useState('');
  const [job, setJob] = useState(null); const [error, setError] = useState(null);
  if (catalog.loading) return <Loading label="Loading recordings" />;
  if (catalog.error) return <Notice tone="danger" title="Could not load recordings"><p>{catalog.error.message}</p></Notice>;
  const rows = catalog.data.recordings.filter((row) => `${row.dataset_id} ${row.recording_source_id} ${row.video?.filename || ''} ${row.eeg?.filename || ''}`.toLowerCase().includes(query.toLowerCase()));
  const run = async (row, mode, requireLabels = false) => {
    setError(null); setJob({ key: `${row.id}:${mode}:${requireLabels}`, progress: { stage_label: 'Starting model run', percent: 0 } });
    try {
      const started = await api.startRun(row.dataset_id, row.recording_source_id, mode, requireLabels);
      const poll = async () => { const current = await api.job(started.id); if (current.status === 'running') { setJob((active) => active ? { ...active, progress: current.progress } : active); window.setTimeout(poll, 3000); return; } setJob(null); catalog.refresh(); if (current.status === 'failed') setError(new Error(current.error)); };
      poll();
    } catch (caught) { setJob(null); setError(caught); }
  };
  return <div className="page">
    <div className="page-title"><div><p>Data catalog</p><h1>Registered recordings</h1><span>Run video and EEG models separately using their registered inputs.</span></div><button className="button" onClick={catalog.refresh}><RefreshCw size={16} /> Refresh</button></div>
    {error && <Notice tone="danger" title="Run could not start"><p>{error.message}</p></Notice>}
    {job && <Card title="Current model run"><div className="job-progress"><div><strong>{job.progress?.stage_label || 'Running model'}</strong><span>{Math.round(job.progress?.percent || 0)}%</span></div><div className="progress-track"><i style={{ width: `${Math.max(0, Math.min(100, job.progress?.percent || 0))}%` }} /></div><div className="progress-meta"><span>{job.progress?.completed_units !== undefined && job.progress?.total_units ? `${Math.round(job.progress.completed_units)} of ${Math.round(job.progress.total_units)} completed` : 'Preparing stage'}</span><span>Elapsed {formatDuration(job.progress?.elapsed_sec)}{job.progress?.estimated_remaining_sec ? ` - about ${formatDuration(job.progress.estimated_remaining_sec)} remaining` : ''}</span></div></div></Card>}
    <Card title={`${rows.length} recording${rows.length === 1 ? '' : 's'}`} action={<label className="search"><Search /><input placeholder="Search catalog" value={query} onChange={(event) => setQuery(event.target.value)} /></label>}>
      {!rows.length ? <Empty title="No recordings found">Upload a video, EEG CSV, or both.</Empty> : <div className="table-wrap"><table><thead><tr><th>Recording</th><th>Dataset</th><th>Inputs</th><th>Ground truth</th><th>Stored</th><th /></tr></thead><tbody>{rows.map((row) => <tr key={row.id}>
        <td><strong>{row.recording_source_id}</strong></td><td>{row.dataset_id}</td>
        <td><div className="modality-files">{row.video && <span><b>Video</b>{row.video.filename}<small>{formatBytes(row.video.size_bytes)}</small></span>}{row.eeg && <span><b>EEG</b>{row.eeg.filename}<small>{formatBytes(row.eeg.size_bytes)}</small></span>}</div></td>
        <td>{row.label ? <><strong>{row.label.filename}</strong><small>{row.label.identity?.patient_id} - {row.label.identity?.seizure_id}</small></> : row.eeg ? <small>EEG labels from column y</small> : <small>Prediction only</small>}</td><td>{formatDate(row.created_at)}</td>
        <td className="right"><div className="run-actions">{row.video && <button className="button small" disabled={job !== null} onClick={() => run(row, 'video')}><Play size={14} />Video</button>}{row.video && row.label && <button className="button primary small" disabled={job !== null} onClick={() => run(row, 'video', true)}><Play size={14} />Evaluate video</button>}{row.eeg && <button className="button small" disabled={job !== null} onClick={() => run(row, 'eeg')}><Play size={14} />EEG</button>}{row.video && row.eeg && <button className="button primary small" disabled={job !== null} onClick={() => run(row, 'unimodal')}><Play size={14} />Both independently</button>}</div></td>
      </tr>)}</tbody></table></div>}
    </Card>
  </div>;
}
