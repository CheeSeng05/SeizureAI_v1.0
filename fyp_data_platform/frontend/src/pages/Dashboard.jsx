import { Link } from 'react-router-dom';
import { ArrowRight, Database, FileVideo, PlayCircle, ShieldCheck } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Badge, Card, Empty, Loading, Notice, Stat, formatDate } from '../components';

export default function Dashboard() {
  const health = useResource(api.health, []);
  const catalog = useResource(api.catalog, []);
  if (health.loading || catalog.loading) return <Loading label="Opening workspace" />;
  if (health.error || catalog.error) return <Notice tone="danger" title="Platform unavailable"><p>{(health.error || catalog.error).message}</p></Notice>;
  const counts = health.data.counts;
  const latest = catalog.data.runs[0];
  return <div className="page">
    <div className="page-title"><div><p>Overview</p><h1>Research workspace</h1><span>Your stored recordings and authoritative model runs.</span></div><Link className="button primary" to="/upload">Upload recording <ArrowRight size={16} /></Link></div>
    <Notice title="Pipeline-backed results"><p>The dashboard displays artifacts saved by FYPModels. It does not recalculate model metrics in the browser.</p></Notice>
    <div className="stats-grid">
      <Stat label="Registered recordings" value={counts.recordings} detail="PostgreSQL catalog" />
      <Stat label="Stored files" value={counts.files} detail="MinIO object storage" />
      <Stat label="Model runs" value={counts.runs} detail="All recorded statuses" />
      <Stat label="Storage" value={health.data.storage_backend.toUpperCase()} detail={health.data.environment} />
    </div>
    <div className="two-column">
      <Card title="Latest model run" action={<Link to="/runs">View all</Link>}>
        {!latest ? <Empty title="No model runs yet">Upload a video and start prediction-only inference.</Empty> : <div className="latest-run">
          <div className="latest-icon"><PlayCircle /></div><div className="grow"><div className="row"><strong>{latest.recording_id}</strong><Badge status={latest.status} /></div><p>{latest.dataset_id} · {latest.modality}</p><small>{formatDate(latest.started_at)}</small></div>
          <Link className="icon-button" to={`/runs/${latest.id}`}><ArrowRight /></Link>
        </div>}
      </Card>
      <Card title="Platform guarantees">
        <ul className="guarantees">
          <li><Database /><div><strong>Catalogued inputs</strong><span>Dataset, patient and recording identity remain linked.</span></div></li>
          <li><ShieldCheck /><div><strong>Verified storage</strong><span>Uploaded objects retain SHA-256 checksums.</span></div></li>
          <li><FileVideo /><div><strong>Isolated outputs</strong><span>Each model run has its own artifact prefix.</span></div></li>
        </ul>
      </Card>
    </div>
  </div>;
}
