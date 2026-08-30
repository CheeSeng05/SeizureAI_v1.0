import { Link } from 'react-router-dom';
import { ArrowRight, RefreshCw } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Badge, Card, Empty, Loading, Notice, formatDate, formatDuration } from '../components';

export default function Runs() {
  const catalog = useResource(api.catalog, []);
  if (catalog.loading) return <Loading label="Loading model runs" />;
  if (catalog.error) return <Notice tone="danger" title="Could not load runs"><p>{catalog.error.message}</p></Notice>;
  const runs = catalog.data.runs;
  const running = runs.filter((run) => run.status === 'running').length; const failed = runs.filter((run) => ['failed', 'blocked'].includes(run.status)).length;
  return <div className="page">
    <div className="page-title"><div><p>Inference history</p><h1>Model runs</h1><span>Durable statuses and isolated output artifacts from PostgreSQL.</span></div><button className="button" onClick={catalog.refresh}><RefreshCw size={16} /> Refresh</button></div>
    {(running > 0 || failed > 0) && <Notice tone={failed > 0 ? 'danger' : 'info'} title="Recent run activity"><p>{running} currently running · {failed} failed or blocked. Use Refresh to retrieve the latest server status.</p></Notice>}
    <Card title={`${runs.length} recent run${runs.length === 1 ? '' : 's'}`}>
      {!runs.length ? <Empty title="No runs recorded">Start prediction from a registered recording.</Empty> : <div className="run-list">{runs.map((run) => <Link key={run.id} to={`/runs/${run.id}`} className="run-row">
        <Badge status={run.status} /><div className="grow"><strong>{run.recording_id || 'Unknown recording'}</strong><span>{run.dataset_id} · {run.modality} · {run.require_labels ? 'evaluation' : 'prediction only'} · {run.compute?.mode?.toUpperCase() || 'UNKNOWN DEVICE'}</span></div><div className="run-meta"><span>{formatDate(run.started_at)}</span><small>{formatDuration(run.runtime?.timings?.total_sec)} · {run.artifact_count} artifacts</small></div><ArrowRight />
      </Link>)}</div>}
    </Card>
  </div>;
}
