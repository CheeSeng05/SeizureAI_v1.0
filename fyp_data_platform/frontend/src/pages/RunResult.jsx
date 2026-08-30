import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { Download, FileOutput } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Badge, Card, Empty, Loading, Notice, Stat, formatBytes, formatDuration, formatTime, percent } from '../components';
import VideoTimelinePlayer from '../VideoTimelinePlayer';

const metricFields = [['Accuracy', 'accuracy'], ['Balanced accuracy', 'balanced_accuracy'], ['Sensitivity', 'sensitivity'], ['Specificity', 'specificity'], ['Precision', 'precision'], ['F1', 'f1'], ['AUROC', 'auroc'], ['Evaluated samples/windows', 'evaluated_windows'], ['Excluded windows', 'excluded_windows'], ['False positives/hour', 'false_positives_per_hour'], ['Onset absolute error (s)', 'onset_mean_absolute_error_sec']];
function MetricList({ metrics }) {
  if (!metrics || metrics.prediction_only) return <Empty title="Metrics unavailable">This workflow did not produce label-backed evaluation metrics.</Empty>;
  return <dl className="metric-list">{metricFields.filter(([, key]) => metrics[key] !== undefined).map(([label, key]) => <div key={key}><dt>{label}</dt><dd>{typeof metrics[key] === 'number' ? metrics[key].toFixed(3) : metrics[key] ?? '—'}</dd></div>)}</dl>;
}

function eventsFromWindows(windows) {
  const events = []; let current = null;
  windows.forEach((window) => {
    if (!window.prediction) { if (current) events.push(current); current = null; return; }
    if (!current) current = { start_sec: window.start_sec, end_sec: window.end_sec, score: window.probability || 0 };
    else { current.end_sec = window.end_sec; current.score = Math.max(current.score, window.probability || 0); }
  });
  if (current) events.push(current);
  return events;
}

function divide(top, bottom) { return bottom ? top / bottom : 'N/A'; }

function metricsAtThreshold(original, windows, events, groundTruth) {
  if (!original || original.prediction_only) return original;
  const evaluated = windows.filter((window) => window.reference_label === 0 || window.reference_label === 1);
  if (!evaluated.length) return original;
  const tp = evaluated.filter((window) => window.prediction && window.reference_label === 1).length;
  const tn = evaluated.filter((window) => !window.prediction && window.reference_label === 0).length;
  const fp = evaluated.filter((window) => window.prediction && window.reference_label === 0).length;
  const fn = evaluated.filter((window) => !window.prediction && window.reference_label === 1).length;
  const sensitivity = divide(tp, tp + fn); const specificity = divide(tn, tn + fp); const precision = divide(tp, tp + fp);
  const f1 = typeof precision === 'number' && typeof sensitivity === 'number' ? divide(2 * precision * sensitivity, precision + sensitivity) : 'N/A';
  const eegOnset = groundTruth.eeg_onset_sec; const clinicalOnset = groundTruth.clinical_onset_sec;
  const candidates = eegOnset == null ? [] : events.filter((event) => event.end_sec > eegOnset);
  return {
    ...original,
    accuracy: divide(tp + tn, evaluated.length),
    balanced_accuracy: typeof sensitivity === 'number' && typeof specificity === 'number' ? (sensitivity + specificity) / 2 : 'N/A',
    sensitivity, specificity, precision, f1,
    evaluated_windows: evaluated.length,
    excluded_windows: windows.length - evaluated.length,
    false_positives_per_hour: eegOnset > 0 ? events.filter((event) => event.end_sec <= eegOnset).length / (eegOnset / 3600) : 'N/A',
    onset_mean_absolute_error_sec: candidates.length && clinicalOnset != null ? Math.abs(candidates[0].start_sec - clinicalOnset) : 'N/A',
  };
}
export default function RunResult() {
  const { id } = useParams(); const [threshold, setThreshold] = useState(0.5); const result = useResource(() => api.run(id), [id]);
  if (result.loading) return <Loading label="Loading stored results" />;
  if (result.error) return <Notice tone="danger" title="Could not load result"><p>{result.error.message}</p></Notice>;
  const data = result.data; const originalOutcome = data.outcome; const runtime = data.runtime || {}; const timings = runtime.timings || {}; const compute = runtime.compute || data.compute || {};
  const workflows = data.workflow_metrics || {}; const isVideo = data.modality === 'video'; const isEEG = data.modality === 'eeg'; const isBoth = data.modality === 'unimodal';
  const thresholdWindows = (data.windows || []).map((window) => ({ ...window, prediction: window.probability != null && Number(window.probability) >= threshold ? 1 : 0 }));
  const thresholdEvents = eventsFromWindows(thresholdWindows);
  const outcome = originalOutcome ? { ...originalOutcome, detected: thresholdEvents.length > 0, event_count: thresholdEvents.length, events: thresholdEvents } : null;
  const originalVideoMetrics = workflows.video?.metrics || (isVideo ? data.metrics : null);
  const videoMetrics = metricsAtThreshold(originalVideoMetrics, thresholdWindows, thresholdEvents, data.ground_truth || {});
  return <div className="page">
    <div className="page-title"><div><p>{data.dataset_id}</p><h1>{data.recording_id}</h1><span>{isBoth ? 'Video and EEG run independently' : `${data.modality?.toUpperCase()} model run`} - {data.id}</span></div><Badge status={data.status} /></div>
    {isBoth && <Notice title="Not a fused prediction"><p>The video and EEG models ran separately. Their probabilities and metrics were not combined.</p></Notice>}
    {data.error && <Notice tone="danger" title="Pipeline error"><p>{data.error}</p></Notice>}
    <div className="stats-grid"><Stat label="Workflow" value={isBoth ? 'Both separate' : isEEG ? 'EEG' : 'Video'} /><Stat label="Status" value={data.status} /><Stat label="Total run time" value={formatDuration(timings.total_sec)} /><Stat label="Processing device" value={compute.mode === 'cuda' ? compute.devices?.[0]?.name || 'NVIDIA CUDA' : compute.cpu || 'CPU'} /></div>
    <Card title="Processing performance"><dl className="metric-list"><div><dt>Model pipeline</dt><dd>{formatDuration(timings.model_pipeline_sec)}</dd></div><div><dt>Input staging</dt><dd>{formatDuration(timings.input_staging_sec)}</dd></div><div><dt>Configuration</dt><dd>{formatDuration(timings.configuration_sec)}</dd></div><div><dt>Artifact persistence</dt><dd>{formatDuration(timings.artifact_persistence_sec)}</dd></div></dl></Card>
    {(isVideo || isBoth) && <><Card title="Detection threshold"><div className="threshold-control"><div><strong>Analysis threshold: {threshold.toFixed(2)}</strong><span>Changes this page only. The model is not rerun and the stored result remains unchanged.</span></div><input type="range" min="0" max="1" step="0.01" value={threshold} aria-label="Detection threshold" onChange={(event) => setThreshold(Number(event.target.value))} /><div className="threshold-scale"><span>0.00 · more sensitive</span><button type="button" onClick={() => setThreshold(0.5)} disabled={threshold === 0.5}>Reset to 0.50</button><span>1.00 · more strict</span></div></div></Card><div className="stats-grid"><Stat label="Video outcome" value={!outcome ? 'Unavailable' : outcome.detected ? 'Detected' : 'No detection'} /><Stat label="Predicted events" value={outcome?.event_count} /><Stat label="Maximum probability" value={percent(outcome?.max_probability)} /><Stat label="Video duration" value={formatTime(data.duration_sec)} /></div><Card title="Video and seizure timeline"><VideoTimelinePlayer source={data.source_video} windows={thresholdWindows} events={outcome?.events || []} groundTruth={data.ground_truth || {}} duration={data.duration_sec} /></Card></>}
    <div className={isBoth ? 'two-column' : ''}>{(isVideo || isBoth) && <Card title="Video evaluation metrics"><Notice title="Exploratory threshold metrics"><p>Threshold-dependent values below use {threshold.toFixed(2)}. AUROC is unchanged because it evaluates the probability ranking across thresholds.</p></Notice><MetricList metrics={videoMetrics} /></Card>}{(isEEG || isBoth) && <Card title="EEG evaluation metrics"><MetricList metrics={workflows.eeg?.metrics || (isEEG ? data.metrics : null)} />{workflows.eeg?.error && <Notice tone="danger"><p>{workflows.eeg.error}</p></Notice>}</Card>}</div>
    <Card title="Stored artifacts">{!data.artifacts.length ? <Empty title="No artifacts">The pipeline did not upload output files for this run.</Empty> : <div className="artifact-list">{data.artifacts.map((artifact) => <a key={artifact.relative_path} href={artifact.url} target="_blank" rel="noreferrer"><FileOutput /><div><strong>{artifact.relative_path}</strong><span>{formatBytes(artifact.size_bytes)} - SHA-256 {artifact.sha256?.slice(0, 12)}...</span></div><Download /></a>)}</div>}</Card>
  </div>;
}
