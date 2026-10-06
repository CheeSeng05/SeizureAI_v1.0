import { useEffect, useState } from 'react';
import { Play, RefreshCw, Search, Square } from 'lucide-react';
import { api } from '../api';
import ArchiveButton from '../ArchiveButton';
import { useResource } from '../hooks';
import { useActiveRun } from '../useActiveRun';
import { Card, Empty, Loading, Notice, formatBytes, formatDate, formatDuration } from '../components';

export default function Recordings() {
  const catalog = useResource(api.catalog, []); const [query, setQuery] = useState('');
  const { job, error, checking, completedVersion, stopping, notice, startRun: run, stopRun } = useActiveRun();
  const busy = checking || job !== null;
  useEffect(() => { if (completedVersion) catalog.refresh(); }, [completedVersion, catalog.refresh]);
  if (catalog.loading) return <Loading label="Loading recordings" />;
  if (catalog.error) return <Notice tone="danger" title="Could not load recordings"><p>{catalog.error.message}</p></Notice>;
  const rows = catalog.data.recordings.filter((row) => `${row.dataset_id} ${row.recording_source_id} ${(row.videos || []).map((video) => video.filename).join(' ')} ${row.video?.filename || ''} ${row.eeg?.filename || ''}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="page">
    <div className="page-title"><div><p>Data catalog</p><h1>Registered recordings</h1><span>Run individual models or verified synchronized video–EEG fusion.</span></div><button className="button" onClick={catalog.refresh}><RefreshCw size={16} /> Refresh</button></div>
    {error && <Notice tone="danger" title={job ? 'Progress update unavailable' : 'Model run update'}><p>{error.message}{job ? ' Retrying automatically; the last known progress is shown below.' : ''}</p></Notice>}
    {checking && !job && <Notice title="Checking for an active model run"><p>Recovering the current progress from the server.</p></Notice>}
    {notice && <Notice title="Model run"><p>{notice}</p></Notice>}
    {job && <div className="title-actions"><button className="button" type="button" disabled={stopping || job.status === 'starting'} onClick={stopRun}><Square size={16} />{stopping ? 'Stopping…' : 'Stop model run'}</button>{stopping && <span>Waiting for processing to stop. Uploaded files are kept.</span>}</div>}
    {job && <Card title="Current model run"><div className="job-progress"><div><strong>{job.progress?.stage_label || 'Running model'}</strong><span>{Math.round(job.progress?.percent || 0)}%</span></div><div className="progress-track"><i style={{ width: `${Math.max(0, Math.min(100, job.progress?.percent || 0))}%` }} /></div><div className="progress-meta"><span>{job.progress?.completed_units !== undefined && job.progress?.total_units ? `${Math.round(job.progress.completed_units)} of ${Math.round(job.progress.total_units)} completed` : 'Preparing stage'}</span><span>Elapsed {formatDuration(job.progress?.elapsed_sec)}{job.progress?.estimated_remaining_sec ? ` - about ${formatDuration(job.progress.estimated_remaining_sec)} remaining` : ''}</span></div></div></Card>}
    <Card title={`${rows.length} recording${rows.length === 1 ? '' : 's'}`} action={<label className="search"><Search /><input placeholder="Search catalog" value={query} onChange={(event) => setQuery(event.target.value)} /></label>}>
      {!rows.length ? <Empty title="No recordings found">Upload a video, EEG CSV, or both.</Empty> : <div className="table-wrap"><table className="recordings-table"><colgroup><col className="recording-column" /><col className="dataset-column" /><col /><col className="labels-column" /><col className="stored-column" /><col className="actions-column" /></colgroup><thead><tr><th>Recording</th><th>Dataset</th><th>Inputs</th><th>Ground truth</th><th>Stored</th><th>Actions</th></tr></thead><tbody>{rows.map((row) => <tr key={row.id}>
        <td className="recording-identity"><strong>{row.recording_source_id}</strong><ArchiveButton kind="recordings" id={row.id} refresh={catalog.refresh} disabled={busy} /></td><td>{row.dataset_id}</td>
        <td><div className="modality-files">{row.video && <span><b>Video{(row.videos || []).length > 1 ? ` (${row.videos.length} segments)` : ''}</b>{(row.videos || [row.video]).map((video) => video.filename).join(', ')}<small>{formatBytes((row.videos || [row.video]).reduce((total, video) => total + Number(video.size_bytes || 0), 0))}</small></span>}{row.eeg && <span><b>EEG</b>{row.eeg.filename}<small>{formatBytes(row.eeg.size_bytes)}</small></span>}{row.video && row.eeg && <small>{row.sync_verified ? `Synchronized · ${row.sync_source}` : 'Synchronization unavailable'}</small>}</div></td>
        <td>{row.label ? <><strong>{row.label.filename}</strong><small>{row.label.identity?.patient_id} - {row.label.identity?.seizure_id}</small></> : row.eeg ? <small>EEG labels from column y</small> : <small>Prediction only</small>}</td><td>{formatDate(row.created_at)}</td>
        <td className="right"><div className="run-actions">
          {row.video && row.eeg && <button className="button primary small" title={row.sync_verified ? 'Run synchronized weighted late fusion' : 'Video/EEG synchronization could not be verified'} disabled={busy || !row.sync_verified} onClick={() => run(row, 'fusion')}><Play size={14} />Run Multimodal Fusion</button>}
          {row.video && <button className="button small" disabled={busy} onClick={() => run(row, 'video')}><Play size={14} />{row.eeg ? 'Run Video Only' : 'Run Video Model'}</button>}
          {row.video && row.label && <button className="button small" disabled={busy} onClick={() => run(row, 'video', true)}><Play size={14} />Evaluate video</button>}
          {row.eeg && <button className="button small" disabled={busy} onClick={() => run(row, 'eeg')}><Play size={14} />{row.video ? 'Run EEG Only' : 'Run EEG Model'}</button>}
          {row.video && row.eeg && !row.sync_verified && <small>Fusion disabled: synchronization could not be verified.</small>}
        </div></td>
      </tr>)}</tbody></table></div>}
    </Card>
  </div>;
}
