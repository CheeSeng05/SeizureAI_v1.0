import { Database, HardDrive, Layers3, Server } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Card, Loading, Notice } from '../components';

export default function System() {
  const health = useResource(api.health, []);
  if (health.loading) return <Loading label="Checking services" />;
  if (health.error) return <Notice tone="danger" title="System check failed"><p>{health.error.message}</p></Notice>;
  const data = health.data;
  return <div className="page narrow"><div className="page-title"><div><p>Configuration</p><h1>System status</h1><span>Runtime services used by the research platform.</span></div></div>
    <div className="system-grid">
      <Card><div className="system-item"><Server /><div><span>Application API</span><strong>Healthy</strong><small>Flask · {data.environment}</small></div></div></Card>
      <Card><div className="system-item"><Database /><div><span>Metadata catalog</span><strong>PostgreSQL</strong><small>{data.counts.recordings} recordings · {data.counts.runs} runs</small></div></div></Card>
      <Card><div className="system-item"><HardDrive /><div><span>Object storage</span><strong>{data.storage_backend.toUpperCase()}</strong><small>{data.counts.files} registered files</small></div></div></Card>
      <Card><div className="system-item"><Layers3 /><div><span>Inference</span><strong>FYPModels</strong><small>Authoritative seizure_multimodal pipeline</small></div></div></Card>
    </div>
    <Notice><p>This is a research prototype and is not intended for clinical diagnosis or patient care.</p></Notice>
  </div>;
}
