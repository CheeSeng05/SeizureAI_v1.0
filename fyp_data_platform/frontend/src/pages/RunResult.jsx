import FusionTimeline from '../FusionTimeline';
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Download, FileDown, FileOutput } from 'lucide-react';
import { api } from '../api';
import { useResource } from '../hooks';
import { Badge, Card, Empty, Loading, Notice, Stat, formatBytes, formatDuration, formatTime, percent } from '../components';
import VideoTimelinePlayer from '../VideoTimelinePlayer';
import { applyThreshold, eventsFromWindows, metricsAtThreshold } from '../analysis';
import { downloadRunPdf } from '../pdfReport';
import { elapsedSeconds } from '../runTiming';
import StopRunButton from '../StopRunButton';
import PatientSummary from '../PatientSummary';

const metricFields = [['Accuracy', 'accuracy', 'Share of evaluated windows classified correctly.'], ['Balanced accuracy', 'balanced_accuracy', 'Average of sensitivity and specificity; useful with imbalanced labels.'], ['Sensitivity', 'sensitivity', 'Share of labelled seizure windows detected by the model.'], ['Specificity', 'specificity', 'Share of labelled non-seizure windows rejected correctly.'], ['Precision', 'precision', 'Share of model-positive windows that are labelled seizure windows.'], ['F1', 'f1', 'Balance between precision and sensitivity.'], ['AUROC', 'auroc', 'Probability-ranking quality across all thresholds; the slider does not change it.'], ['True positives', 'true_positives', 'Seizure windows correctly detected.'], ['True negatives', 'true_negatives', 'Non-seizure windows correctly rejected.'], ['False positives', 'false_positives', 'Non-seizure windows incorrectly flagged.'], ['False negatives', 'false_negatives', 'Seizure windows missed by the model.'], ['Labelled seizure windows', 'labelled_seizure_windows', 'Evaluated windows labelled as seizure.'], ['Labelled non-seizure windows', 'labelled_non_seizure_windows', 'Evaluated windows labelled as non-seizure.'], ['Predicted seizure windows', 'predicted_seizure_windows', 'Evaluated windows above the selected threshold.'], ['Predicted non-seizure windows', 'predicted_non_seizure_windows', 'Evaluated windows below the selected threshold.'], ['Evaluated samples/windows', 'evaluated_windows', 'Windows with usable ground-truth labels.'], ['Excluded windows', 'excluded_windows', 'Windows omitted because no binary label was available.'], ['False positives/hour', 'false_positives_per_hour', 'Model events before labelled onset, normalized per monitoring hour.'], ['Onset absolute error (s)', 'onset_mean_absolute_error_sec', 'Absolute difference between model event start and labelled clinical onset.']];
function MetricList({ metrics }) {
  if (!metrics || metrics.prediction_only) return <Empty title="Metrics unavailable">This workflow did not produce label-backed evaluation metrics.</Empty>;
  return <dl className="metric-list">{metricFields.filter(([, key]) => metrics[key] !== undefined).map(([label, key, help]) => <div key={key}><dt title={help}>{label}<small className="metric-help" aria-label={help}>?</small></dt><dd>{typeof metrics[key] === 'number' ? metrics[key].toFixed(3) : metrics[key] ?? '—'}</dd></div>)}</dl>;
}

export default function RunResult() {
  const { id } = useParams(); const [threshold, setThreshold] = useState(0.5); const result = useResource(() => api.run(id), [id], { pollInterval: 3000, shouldPoll: (data) => data.status === 'running' });
  const [now, setNow] = useState(Date.now());
  useEffect(() => { if (result.data?.status !== 'running') return; const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, [result.data?.status]);
  if (result.loading) return <Loading label="Loading stored results" />;
  if (result.error) return <Notice tone="danger" title="Could not load result"><p>{result.error.message}</p></Notice>;
  const data = result.data; const running = data.status === 'running'; const complete = data.status === 'completed'; const pending = !complete; const progress = data.progress; const originalOutcome = data.outcome; const runtime = data.runtime || {}; const timings = runtime.timings || {}; const compute = runtime.compute || data.compute || {};
  const workflows = data.workflow_metrics || {}; const isVideo = data.modality === 'video'; const isEEG = data.modality === 'eeg'; const isBoth = data.modality === 'unimodal'; const isFusion = data.modality === 'fusion';
  const isMultiSegmentVideo = isVideo && Number(data.metrics?.video_segment_count || 0) > 1;
  const fusionWindows = data.fusion_windows || []; const fusionAvailable = fusionWindows.filter((window) => window.fusion_available); const fusionPositive = fusionAvailable.filter((window) => Number(window.fused_prediction) === 1);
  const eegWindows = (data.eeg_windows?.length ? data.eeg_windows : isEEG ? data.windows : []) || [];
  const eegWorkflow = workflows.eeg || {};
  const eegMetrics = eegWorkflow.metrics || (isEEG ? data.metrics : {}) || {};
  const eegMetadata = eegWorkflow.metadata || {};
  const isRecordingPrediction = eegWindows.length > 0 && Boolean(eegMetrics.prediction_only);
  const eegProbabilities = eegWindows.map((window) => Number(window.probability)).filter(Number.isFinite);
  const eegMaximum = eegProbabilities.length ? Math.max(...eegProbabilities) : null;
  const eegThreshold = Number.isFinite(eegMetrics.threshold) ? eegMetrics.threshold : 0.5;
  const detectedEegWindows = eegWindows.filter((window) => window.probability != null && Number.isFinite(Number(window.probability)) && Number(window.probability) > eegThreshold);
  const eegDuration = eegMetadata.duration_sec || (eegWindows.length ? Math.max(...eegWindows.map((window) => Number(window.end_sec) || 0)) : null);
  const thresholdWindows = applyThreshold(data.windows || [], threshold);
  const thresholdEvents = eventsFromWindows(thresholdWindows);
  const probabilities = thresholdWindows.map((window) => Number(window.probability)).filter(Number.isFinite);
  const maxProbability = probabilities.length ? Math.max(...probabilities) : null;
  const outcome = originalOutcome ? { ...originalOutcome, detected: thresholdEvents.length > 0, event_count: thresholdEvents.length, events: thresholdEvents } : null;
  const originalVideoMetrics = workflows.video?.metrics || (isVideo ? data.metrics : null);
  const videoMetrics = metricsAtThreshold(originalVideoMetrics, thresholdWindows, thresholdEvents, data.ground_truth || {});
  const defaultWindows = applyThreshold(data.windows || [], 0.5);
  const defaultEvents = eventsFromWindows(defaultWindows);
  const defaultMetrics = metricsAtThreshold(originalVideoMetrics, defaultWindows, defaultEvents, data.ground_truth || {});
  const truthAvailable = data.ground_truth && Object.values(data.ground_truth).some((value) => value != null);
  const matchedEvents = thresholdEvents.filter((event) => { const start = data.ground_truth?.clinical_onset_sec ?? data.ground_truth?.eeg_onset_sec; const end = data.ground_truth?.seizure_end_sec; return start != null && (end != null ? event.end_sec > start && event.start_sec < end : event.start_sec <= start && event.end_sec >= start); }).length;
  const exportReport = () => {
    downloadRunPdf({ data, threshold, windows: thresholdWindows, events: thresholdEvents, matchedEvents, maxProbability, metrics: videoMetrics, compute, runtime });
  };
  return <div className="page">
    <div className="page-title"><div><p>{data.dataset_id}</p><h1>{data.recording_id}</h1><span>{isBoth ? 'Video and EEG run independently' : isFusion ? 'Synchronized weighted late fusion' : `${data.modality?.toUpperCase()} model run`} - {data.id}</span></div><div className="title-actions"><button className="button" type="button" onClick={exportReport} disabled={!complete}><FileDown size={16} /> Export report</button><Badge status={data.status} /></div></div>
    <PatientSummary key={id} runId={id} patient={data.patient} />
    {isBoth && <Notice title="Not a fused prediction"><p>The video and EEG models ran separately. Their probabilities and metrics were not combined.</p></Notice>}
    {data.error && <Notice tone="danger" title="Pipeline error"><p>{data.error}</p></Notice>}
    {running && <StopRunButton id={id} requested={data.stop_requested} />}
    {data.status === 'stopped' && <Notice title="Run stopped"><p>This run was stopped before completion. Uploaded files are unchanged. Start a new run from Recordings when ready.</p></Notice>}

    {running && <Notice title="Model is processing"><p>{progress?.stage_label || 'The model pipeline is running locally.'}{progress?.total_units ? ` · ${progress.completed_units || 0} of ${progress.total_units} clips/batches processed` : ''} Results and final stage timings appear when processing finishes. This page refreshes automatically.</p></Notice>}

    {isFusion && complete && <>
      <Notice title="Verified multimodal result"><p>EEG is the master timeline. Video probabilities were mapped from their original segments and combined by temporal overlap. The default 80% EEG / 20% video weights are an engineering baseline, not clinically optimized.</p></Notice>
      <div className="stats-grid"><Stat label="EEG windows" value={fusionWindows.length} /><Stat label="Fusion-eligible" value={fusionAvailable.length} /><Stat label="Without video coverage" value={fusionWindows.length - fusionAvailable.length} /><Stat label="Fused positive windows" value={fusionPositive.length} /></div>
      <FusionTimeline key={id} runId={id} windows={fusionWindows} />
    </>}

    {(isVideo || isBoth) && <>{complete && <><Notice title="Result summary"><p>The model found <strong>{thresholdEvents.length}</strong> possible seizure event{thresholdEvents.length === 1 ? '' : 's'} at threshold {threshold.toFixed(2)}.{truthAvailable ? ` ${matchedEvents} matched the workbook timing and ${thresholdEvents.length - matchedEvents} were model-only.` : ' No workbook was supplied, so these predictions cannot be checked against ground truth.'}</p></Notice><Card title="Detection threshold"><div className="threshold-control"><div><strong>Analysis threshold</strong><span>Changes this page only. The model is not rerun and the stored result remains unchanged.</span></div><div className="threshold-input"><input type="range" min="0" max="1" step="0.01" value={threshold} aria-label="Detection threshold slider" onChange={(event) => setThreshold(Number(event.target.value))} /><input type="number" min="0" max="1" step="0.01" value={threshold} aria-label="Type detection threshold" onChange={(event) => { const value = Number(event.target.value); if (value >= 0 && value <= 1) setThreshold(value); }} /></div><div className="threshold-scale"><span>0.00 · more sensitive</span><button type="button" onClick={() => setThreshold(0.5)} disabled={threshold === 0.5}>Reset to 0.50</button><span>1.00 · more strict</span></div></div></Card><div className="stats-grid"><Stat label="Video outcome" value={!outcome ? 'Unavailable' : outcome.detected ? 'Detected' : 'No detection'} /><Stat label="Predicted events" value={outcome?.event_count} /><Stat label="Maximum probability" value={percent(maxProbability)} /><Stat label="Video duration" value={formatTime(data.duration_sec)} /></div></>}{isMultiSegmentVideo ? <Card title="Synchronized video-segment timeline"><p>All registered video segments were processed independently and mapped to the recording clock. Playback is not combined because the original files remain separate.</p><div className="comparison-table"><div><strong>Time</strong><strong>Segment</strong><strong>Probability</strong></div>{thresholdWindows.slice(0, 500).map((window, index) => <div key={`${window.start_sec}-${index}`}><span>{formatTime(window.start_sec)}–{formatTime(window.end_sec)}</span><b>{window.video_segment_id || '—'}</b><b>{window.probability == null ? '—' : Number(window.probability).toFixed(4)}</b></div>)}</div>{thresholdWindows.length > 500 && <p>Showing the first 500 windows. Download video_only/predictions.csv for the complete timeline.</p>}</Card> : <Card title="Video and seizure timeline"><VideoTimelinePlayer key={id} pending={pending} source={data.source_video} windows={thresholdWindows} events={outcome?.events || []} groundTruth={data.ground_truth || {}} duration={data.duration_sec} threshold={threshold} /></Card>}</>}
    {(isEEG || isBoth) && isRecordingPrediction && <>
      <Notice title="EEG recording processed"><p>Showing only windows with probability above {eegThreshold.toFixed(2)}. These experimental predictions are not a clinical diagnosis.</p></Notice>
      <div className="stats-grid"><Stat label="EEG windows analysed" value={eegWindows.length} /><Stat label="Detected windows" value={detectedEegWindows.length} /><Stat label="Maximum probability" value={percent(eegMaximum)} /><Stat label="Recording duration" value={formatTime(eegDuration)} /></div>
      <Card title="Detected EEG windows">
        <p>{detectedEegWindows.length} of {eegWindows.length} windows have probability &gt; {eegThreshold.toFixed(2)}. Download predictions.csv for the complete timeline.</p>
        {!detectedEegWindows.length ? <Empty title="No detected windows">No EEG windows exceed the {eegThreshold.toFixed(2)} threshold.</Empty> : <div className="comparison-table">
          <div><strong>Time</strong><strong>Probability</strong><strong>Detection</strong></div>
          {detectedEegWindows.map((window, index) => <div key={`${window.start_sec}-${index}`}><span>{formatTime(window.start_sec)}–{formatTime(window.end_sec)}</span><b>{Number(window.probability).toFixed(4)}</b><b>Possible seizure</b></div>)}
        </div>}
      </Card>
    </>}
    <details className="card result-details"><summary>Evaluation details</summary><div className="card-body"><p>Evaluation against supplied labels for this run. These values do not describe overall model performance.</p>
    <div className={isBoth ? 'two-column' : ''}>{complete && (isVideo || isBoth) && <Card title="Video evaluation metrics"><Notice title="Exploratory threshold metrics"><p>Threshold-dependent values below use {threshold.toFixed(2)}. AUROC is unchanged because it evaluates the probability ranking across thresholds.</p></Notice><MetricList metrics={videoMetrics} /></Card>}{(isEEG || isBoth) && <Card title="EEG evaluation metrics"><MetricList metrics={workflows.eeg?.metrics || (isEEG ? data.metrics : null)} />{workflows.eeg?.error && <Notice tone="danger"><p>{workflows.eeg.error}</p></Notice>}</Card>}</div>
    {complete && (isVideo || isBoth) && <Card title="Default versus selected threshold"><div className="comparison-table"><div><strong>Measure</strong><strong>Default 0.50</strong><strong>Selected {threshold.toFixed(2)}</strong></div>{[['Detected events', defaultEvents.length, thresholdEvents.length], ...metricFields.map(([label, key]) => [label, defaultMetrics?.[key], videoMetrics?.[key]])].map(([label, original, selected]) => <div key={label}><span>{label}</span><b>{typeof original === 'number' ? original.toFixed(3) : original ?? '—'}</b><b>{typeof selected === 'number' ? selected.toFixed(3) : selected ?? '—'}</b></div>)}</div></Card>}
    </div></details>
    <details className="card result-details"><summary>Processing details and downloads</summary><div className="card-body">
    <div className="stats-grid"><Stat label="Workflow" value={isFusion ? 'Multimodal fusion' : isBoth ? 'Both separate' : isEEG ? 'EEG' : 'Video'} /><Stat label="Status" value={data.status} /><Stat label={running ? "Elapsed run time" : "Total run time"} value={formatDuration(elapsedSeconds(data, now))} /><Stat label="Processing device" value={compute.mode === 'cuda' ? compute.devices?.[0]?.name || 'NVIDIA CUDA' : `CPU (${compute.cpu || 'local'})`} /></div>
    <Card title="Processing performance"><dl className="metric-list"><div><dt>Model pipeline</dt><dd>{running ? 'In progress' : formatDuration(timings.model_pipeline_sec)}</dd></div><div><dt>Input staging</dt><dd>{running ? 'Available when finished' : formatDuration(timings.input_staging_sec)}</dd></div><div><dt>Configuration</dt><dd>{running ? 'Available when finished' : formatDuration(timings.configuration_sec)}</dd></div><div><dt>Artifact persistence</dt><dd>{running ? 'Available when finished' : formatDuration(timings.artifact_persistence_sec)}</dd></div></dl></Card>
    <Card title="Stored artifacts">{!data.artifacts.length ? <Empty title={running ? "Outputs pending" : "No artifacts"}>{running ? "Outputs are saved when the model finishes." : "The pipeline did not upload output files for this run."}</Empty> : <div className="artifact-list">{data.artifacts.map((artifact) => <a key={artifact.relative_path} href={artifact.url} target="_blank" rel="noreferrer"><FileOutput /><div><strong>{artifact.relative_path}</strong><span>{formatBytes(artifact.size_bytes)} - SHA-256 {artifact.sha256?.slice(0, 12)}...</span></div><Download /></a>)}</div>}</Card>
    </div></details>
  </div>;
}
