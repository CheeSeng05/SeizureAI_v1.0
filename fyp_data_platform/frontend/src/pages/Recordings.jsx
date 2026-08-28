import { useState } from 'react';
import { Play, RefreshCw, Search } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Card, Empty, Loading, Notice, formatBytes, formatDate } from '../components';

export default function Recordings() {
  const catalog = useResource(api.catalog, []);
  const [query, setQuery] = useState('');
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  if (catalog.loading) return <Loading label="Loading recordings" />;
  if (catalog.error) return <Notice tone="danger" title="Could not load recordings"><p>{catalog.error.message}</p></Notice>;
  const rows = catalog.data.videos.filter((video) => `${video.dataset_id} ${video.recording_source_id} ${video.filename}`.toLowerCase().includes(query.toLowerCase()));
  const run = async (video) => {
    setError(null); setJob(video.id);
    try {
      const started = await api.startRun(video.dataset_id, video.recording_source_id, false);
      const poll = async () => {
        const current = await api.job(started.id);
        if (current.status === 'running') { window.setTimeout(poll, 3000); return; }
        setJob(null); catalog.refresh();
        if (current.status === 'failed') setError(new Error(current.error));
      };
      poll();
    } catch (caught) { setJob(null); setError(caught); }
  };
  return <div className="page">
    <div className="page-title"><div><p>Data catalog</p><h1>Registered recordings</h1><span>Videos available to the authoritative inference pipeline.</span></div><button className="button" onClick={catalog.refresh}><RefreshCw size={16} /> Refresh</button></div>
    {error && <Notice tone="danger" title="Run could not start"><p>{error.message}</p></Notice>}
    <Card title={`${rows.length} recording${rows.length === 1 ? '' : 's'}`} action={<label className="search"><Search /><input placeholder="Search catalog" value={query} onChange={(event) => setQuery(event.target.value)} /></label>}>
      {!rows.length ? <Empty title="No recordings found">Upload a video to register the first recording.</Empty> : <div className="table-wrap"><table><thead><tr><th>Recording</th><th>Dataset</th><th>File</th><th>Stored</th><th /></tr></thead><tbody>
        {rows.map((video) => <tr key={video.id}><td><strong>{video.recording_source_id}</strong><small className="hash">{video.sha256?.slice(0, 16)}…</small></td><td>{video.dataset_id}</td><td>{video.filename}<small>{formatBytes(video.size_bytes)}</small></td><td>{formatDate(video.created_at)}</td><td className="right"><button className="button primary small" disabled={job !== null} onClick={() => run(video)}><Play size={14} />{job === video.id ? 'Running…' : 'Run'}</button></td></tr>)}
      </tbody></table></div>}
    </Card>
  </div>;
}
