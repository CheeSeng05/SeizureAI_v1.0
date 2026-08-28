import { Cpu, Database, HardDrive, Layers3, Server } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Card, Loading, Notice } from '../components';

export default function System() {
  const health = useResource(api.health, []);
  if (health.loading) return <Loading label="Checking services" />;
  if (health.error) return <Notice tone="danger" title="System check failed"><p>{health.error.message}</p></Notice>;
  const data = health.data;
  const compute = data.compute;
  const primaryDevice = compute.devices[0];
  const memory = primaryDevice ? `${(primaryDevice.memory_bytes / (1024 ** 3)).toFixed(1)} GB VRAM` : `${compute.logical_cpu_count || 'Unknown'} logical CPU cores`;
  return <div className="page narrow"><div className="page-title"><div><p>Configuration</p><h1>System status</h1><span>Runtime services used by the research platform.</span></div></div>
    <div className="system-grid">
      <Card><div className="system-item"><Server /><div><span>Application API</span><strong>Healthy</strong><small>Flask · {data.environment}</small></div></div></Card>
      <Card><div className="system-item"><Database /><div><span>Metadata catalog</span><strong>PostgreSQL</strong><small>{data.counts.recordings} recordings · {data.counts.runs} runs</small></div></div></Card>
      <Card><div className="system-item"><HardDrive /><div><span>Object storage</span><strong>{data.storage_backend.toUpperCase()}</strong><small>{data.counts.files} registered files</small></div></div></Card>
      <Card><div className="system-item"><Cpu /><div><span>Active processing</span><strong>{compute.mode === 'cuda' ? 'NVIDIA CUDA' : 'CPU'}</strong><small>{primaryDevice?.name || compute.cpu} · {memory}</small></div></div></Card>
      <Card><div className="system-item"><Layers3 /><div><span>PyTorch runtime</span><strong>{compute.pytorch_version || 'Unavailable'}</strong><small>{compute.cuda_available ? `CUDA ${compute.cuda_runtime_version} · cuDNN ${compute.cudnn_version || 'unknown'}` : 'CUDA is not available to this container'}</small></div></div></Card>
    </div>
    {compute.profile === 'nvidia-cuda' && !compute.cuda_available && <Notice tone="danger" title="CUDA configuration problem"><p>The GPU profile was requested, but PyTorch cannot access an NVIDIA GPU. Check Docker Desktop and the NVIDIA driver.</p></Notice>}
    {compute.profile !== 'nvidia-cuda' && <Notice><p>Universal CPU mode is active. On a configured NVIDIA computer, start with the GPU Compose override to enable CUDA.</p></Notice>}
    <Notice><p>This is a research prototype and is not intended for clinical diagnosis or patient care.</p></Notice>
  </div>;
}
