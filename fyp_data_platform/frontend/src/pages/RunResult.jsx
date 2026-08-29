import { useParams } from 'react-router-dom';
import { Download, FileOutput } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Badge, Card, Empty, Loading, Notice, Stat, formatBytes, formatDuration, formatTime, percent } from '../components';

const metricFields = [['Accuracy', 'accuracy'], ['Balanced accuracy', 'balanced_accuracy'], ['Sensitivity', 'sensitivity'], ['Specificity', 'specificity'], ['Precision', 'precision'], ['F1', 'f1'], ['AUROC', 'auroc'], ['Evaluated samples/windows', 'evaluated_windows'], ['Excluded windows', 'excluded_windows'], ['False positives/hour', 'false_positives_per_hour'], ['Onset absolute error (s)', 'onset_mean_absolute_error_sec']];
function MetricList({ metrics }) {
  if (!metrics || metrics.prediction_only) return <Empty title="Metrics unavailable">This workflow did not produce label-backed evaluation metrics.</Empty>;
  return <dl className="metric-list">{metricFields.filter(([, key]) => metrics[key] !== undefined).map(([label, key]) => <div key={key}><dt>{label}</dt><dd>{typeof metrics[key] === 'number' ? metrics[key].toFixed(3) : metrics[key] ?? '—'}</dd></div>)}</dl>;
}
function Timeline({ windows }) {
  if (!windows.length) return <Empty title="No video timeline">This run did not produce video probability windows.</Empty>;
  return <div><div className="timeline-chart">{windows.map((window, index) => <i key={index} title={`${window.start_sec.toFixed(1)}-${window.end_sec.toFixed(1)}s - ${percent(window.probability)}`} className={window.prediction ? 'positive' : ''} style={{ height: `${Math.max(3, (window.probability || 0) * 100)}%` }} />)}</div><div className="timeline-axis"><span>0:00</span><span>{formatTime(windows.at(-1).end_sec)}</span></div></div>;
}

export default function RunResult() {
  const { id } = useParams(); const result = useResource(() => api.run(id), [id]);
  if (result.loading) return <Loading label="Loading stored results" />;
  if (result.error) return <Notice tone="danger" title="Could not load result"><p>{result.error.message}</p></Notice>;
  const data = result.data; const outcome = data.outcome; const runtime = data.runtime || {}; const timings = runtime.timings || {}; const compute = runtime.compute || data.compute || {};
  const workflows = data.workflow_metrics || {}; const isVideo = data.modality === 'video'; const isEEG = data.modality === 'eeg'; const isBoth = data.modality === 'unimodal';
  return <div className="page">
    <div className="page-title"><div><p>{data.dataset_id}</p><h1>{data.recording_id}</h1><span>{isBoth ? 'Video and EEG run independently' : `${data.modality?.toUpperCase()} model run`} - {data.id}</span></div><Badge status={data.status} /></div>
    {isBoth && <Notice title="Not a fused prediction"><p>The video and EEG models ran separately. Their probabilities and metrics were not combined.</p></Notice>}
    {data.error && <Notice tone="danger" title="Pipeline error"><p>{data.error}</p></Notice>}
    <div className="stats-grid"><Stat label="Workflow" value={isBoth ? 'Both separate' : isEEG ? 'EEG' : 'Video'} /><Stat label="Status" value={data.status} /><Stat label="Total run time" value={formatDuration(timings.total_sec)} /><Stat label="Processing device" value={compute.mode === 'cuda' ? compute.devices?.[0]?.name || 'NVIDIA CUDA' : compute.cpu || 'CPU'} /></div>
    <Card title="Processing performance"><dl className="metric-list"><div><dt>Model pipeline</dt><dd>{formatDuration(timings.model_pipeline_sec)}</dd></div><div><dt>Input staging</dt><dd>{formatDuration(timings.input_staging_sec)}</dd></div><div><dt>Configuration</dt><dd>{formatDuration(timings.configuration_sec)}</dd></div><div><dt>Artifact persistence</dt><dd>{formatDuration(timings.artifact_persistence_sec)}</dd></div></dl></Card>
    {(isVideo || isBoth) && <><div className="stats-grid"><Stat label="Video outcome" value={!outcome ? 'Unavailable' : outcome.detected ? 'Detected' : 'No detection'} /><Stat label="Predicted events" value={outcome?.event_count} /><Stat label="Maximum probability" value={percent(outcome?.max_probability)} /><Stat label="Video duration" value={formatTime(data.duration_sec)} /></div><Card title="Video seizure probability timeline"><Timeline windows={data.windows || []} /></Card></>}
    <div className={isBoth ? 'two-column' : ''}>{(isVideo || isBoth) && <Card title="Video evaluation metrics"><MetricList metrics={workflows.video?.metrics || (isVideo ? data.metrics : null)} /></Card>}{(isEEG || isBoth) && <Card title="EEG evaluation metrics"><MetricList metrics={workflows.eeg?.metrics || (isEEG ? data.metrics : null)} />{workflows.eeg?.error && <Notice tone="danger"><p>{workflows.eeg.error}</p></Notice>}</Card>}</div>
    <Card title="Stored artifacts">{!data.artifacts.length ? <Empty title="No artifacts">The pipeline did not upload output files for this run.</Empty> : <div className="artifact-list">{data.artifacts.map((artifact) => <a key={artifact.relative_path} href={artifact.url} target="_blank" rel="noreferrer"><FileOutput /><div><strong>{artifact.relative_path}</strong><span>{formatBytes(artifact.size_bytes)} - SHA-256 {artifact.sha256?.slice(0, 12)}...</span></div><Download /></a>)}</div>}</Card>
  </div>;
}
