import { useParams } from 'react-router-dom';
import { Download, FileOutput } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Badge, Card, Empty, Loading, Notice, Stat, formatBytes, formatTime, percent } from '../components';

function Timeline({ windows }) {
  if (!windows.length) return <Empty title="No prediction timeline">This run did not produce video probability windows.</Empty>;
  return <div><div className="timeline-chart">{windows.map((window, index) => <i key={index} title={`${window.start_sec.toFixed(1)}–${window.end_sec.toFixed(1)}s · ${percent(window.probability)}`} className={window.prediction ? 'positive' : ''} style={{ height: `${Math.max(3, (window.probability || 0) * 100)}%` }} />)}</div><div className="timeline-axis"><span>0:00</span><span>{formatTime(windows.at(-1).end_sec)}</span></div></div>;
}

export default function RunResult() {
  const { id } = useParams();
  const result = useResource(() => api.run(id), [id]);
  if (result.loading) return <Loading label="Loading stored results" />;
  if (result.error) return <Notice tone="danger" title="Could not load result"><p>{result.error.message}</p></Notice>;
  const data = result.data; const outcome = data.outcome; const metrics = data.metrics || {};
  return <div className="page">
    <div className="page-title"><div><p>{data.dataset_id}</p><h1>{data.recording_id}</h1><span>Run {data.id}</span></div><Badge status={data.status} /></div>
    {data.error && <Notice tone="danger" title="Pipeline error"><p>{data.error}</p></Notice>}
    {outcome && <Notice title={outcome.prediction_only ? 'Prediction-only result' : 'Label-backed evaluation'}><p>{outcome.note}</p></Notice>}
    <div className="stats-grid">
      <Stat label="Outcome" value={!outcome ? 'Unavailable' : outcome.detected ? 'Detected' : 'No detection'} />
      <Stat label="Predicted events" value={outcome?.event_count} />
      <Stat label="Maximum probability" value={percent(outcome?.max_probability)} />
      <Stat label="Video duration" value={formatTime(data.duration_sec)} />
    </div>
    <Card title="Seizure probability timeline"><Timeline windows={data.windows || []} /></Card>
    <div className="two-column">
      <Card title="Detected events">{!outcome?.events?.length ? <Empty title="No positive events">No stored event crossed the model decision threshold.</Empty> : <div className="event-list">{outcome.events.map((event, index) => <div key={index}><span>{formatTime(event.start_sec)} – {formatTime(event.end_sec)}</span><strong>{percent(event.score)}</strong></div>)}</div>}</Card>
      <Card title="Evaluation metrics">{metrics.prediction_only || !Object.keys(metrics).length ? <Empty title="Metrics unavailable">Register matching labels and run evaluation to calculate accuracy metrics.</Empty> : <dl className="metric-list">{[['Accuracy', metrics.accuracy], ['Sensitivity', metrics.sensitivity], ['Specificity', metrics.specificity], ['Precision', metrics.precision], ['F1', metrics.f1], ['AUROC', metrics.auroc], ["Cohen's κ", metrics.cohen_kappa], ['False positives/hour', metrics.false_positives_per_hour]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{typeof value === 'number' ? value.toFixed(3) : value ?? '—'}</dd></div>)}</dl>}</Card>
    </div>
    <Card title="Stored artifacts">{!data.artifacts.length ? <Empty title="No artifacts">The pipeline did not upload output files for this run.</Empty> : <div className="artifact-list">{data.artifacts.map((artifact) => <a key={artifact.relative_path} href={artifact.url} target="_blank" rel="noreferrer"><FileOutput /><div><strong>{artifact.relative_path}</strong><span>{formatBytes(artifact.size_bytes)} · SHA-256 {artifact.sha256?.slice(0, 12)}…</span></div><Download /></a>)}</div>}</Card>
  </div>;
}
