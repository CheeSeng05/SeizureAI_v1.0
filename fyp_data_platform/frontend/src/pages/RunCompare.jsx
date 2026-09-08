import { useState } from 'react';
import { Link } from 'react-router-dom';
import { FileDown } from 'lucide-react';
import { api } from '../api';
import { videoAnalysis } from '../analysis';
import { useResource } from '../hooks';
import { Card, Empty, Loading, Notice } from '../components';
import { downloadComparisonPdf } from '../pdfReport';

const sections = [
  { title: 'Detection overview', rows: [
    { label: 'Threshold', value: (_, threshold) => threshold, digits: 2, direction: 'neutral' },
    { label: 'Detected events', value: (analysis) => analysis.events.length, digits: 0, direction: 'neutral' },
    { label: 'Maximum probability', value: (analysis) => analysis.maxProbability, direction: 'neutral' },
    { label: 'Video duration (s)', value: (_, __, detail) => detail?.duration_sec, direction: 'neutral' },
  ] },
  { title: 'Performance metrics', rows: [
    { label: 'Accuracy', value: (analysis) => analysis.metrics?.accuracy, direction: 'higher' },
    { label: 'Balanced accuracy', value: (analysis) => analysis.metrics?.balanced_accuracy, direction: 'higher' },
    { label: 'Sensitivity / recall', value: (analysis) => analysis.metrics?.sensitivity, direction: 'higher' },
    { label: 'Specificity', value: (analysis) => analysis.metrics?.specificity, direction: 'higher' },
    { label: 'Precision', value: (analysis) => analysis.metrics?.precision, direction: 'higher' },
    { label: 'F1', value: (analysis) => analysis.metrics?.f1, direction: 'higher' },
    { label: 'AUROC', value: (analysis) => analysis.metrics?.auroc, direction: 'neutral' },
  ] },
  { title: 'Confusion matrix and window counts', rows: [
    { label: 'True positives', value: (analysis) => analysis.metrics?.true_positives, digits: 0, direction: 'higher' },
    { label: 'True negatives', value: (analysis) => analysis.metrics?.true_negatives, digits: 0, direction: 'higher' },
    { label: 'False positives', value: (analysis) => analysis.metrics?.false_positives, digits: 0, direction: 'lower' },
    { label: 'False negatives', value: (analysis) => analysis.metrics?.false_negatives, digits: 0, direction: 'lower' },
    { label: 'Labelled seizure windows', value: (analysis) => analysis.metrics?.labelled_seizure_windows, digits: 0, direction: 'neutral' },
    { label: 'Labelled non-seizure windows', value: (analysis) => analysis.metrics?.labelled_non_seizure_windows, digits: 0, direction: 'neutral' },
    { label: 'Predicted seizure windows', value: (analysis) => analysis.metrics?.predicted_seizure_windows, digits: 0, direction: 'neutral' },
    { label: 'Predicted non-seizure windows', value: (analysis) => analysis.metrics?.predicted_non_seizure_windows, digits: 0, direction: 'neutral' },
    { label: 'Evaluated windows', value: (analysis) => analysis.metrics?.evaluated_windows, digits: 0, direction: 'neutral' },
    { label: 'Excluded windows', value: (analysis) => analysis.metrics?.excluded_windows, digits: 0, direction: 'neutral' },
  ] },
  { title: 'Detection quality', rows: [
    { label: 'False positives/hour', value: (analysis) => analysis.metrics?.false_positives_per_hour, direction: 'lower' },
    { label: 'Onset absolute error (s)', value: (analysis) => analysis.metrics?.onset_mean_absolute_error_sec, direction: 'lower' },
  ] },
  { title: 'Runtime', rows: [
    { label: 'Total runtime (s)', value: (_, __, detail) => detail?.runtime?.timings?.total_sec, direction: 'lower' },
    { label: 'Model pipeline (s)', value: (_, __, detail) => detail?.runtime?.timings?.model_pipeline_sec, direction: 'lower' },
    { label: 'Input staging (s)', value: (_, __, detail) => detail?.runtime?.timings?.input_staging_sec, direction: 'lower' },
    { label: 'Configuration (s)', value: (_, __, detail) => detail?.runtime?.timings?.configuration_sec, direction: 'lower' },
    { label: 'Artifact persistence (s)', value: (_, __, detail) => detail?.runtime?.timings?.artifact_persistence_sec, direction: 'lower' },
  ] },
];

function numeric(value) { return typeof value === 'number' && Number.isFinite(value) ? value : null; }
function display(value, digits = 3) { return numeric(value) == null ? value ?? 'N/A' : value.toFixed(digits); }
function difference(left, right, direction, digits = 3) {
  const first = numeric(left); const second = numeric(right); if (first == null || second == null) return { text: 'N/A', tone: 'neutral' };
  const delta = first - second; const improved = direction === 'higher' ? delta > 0 : direction === 'lower' ? delta < 0 : false; const worse = direction === 'higher' ? delta < 0 : direction === 'lower' ? delta > 0 : false;
  return { text: `${delta > 0 ? '+' : ''}${delta.toFixed(digits)}`, tone: improved ? 'improved' : worse ? 'worse' : 'neutral' };
}

function comparisonWarnings(left, right) {
  const warnings = [];
  if (left.dataset_id !== right.dataset_id || left.recording_id !== right.recording_id) warnings.push('The runs use different recordings. Detection and runtime differences may not be directly comparable.');
  if (left.model_type !== right.model_type || left.model_version !== right.model_version) warnings.push('The runs use different model types or versions.');
  if (left.modality !== right.modality) warnings.push('The runs use different modalities.');
  if (Boolean(left.require_labels) !== Boolean(right.require_labels)) warnings.push('Only one run used registered ground-truth labels, so label-backed metrics are not equivalent.');
  return warnings;
}

export default function RunCompare() {
  const catalog = useResource(api.catalog, []); const [leftId, setLeftId] = useState(''); const [rightId, setRightId] = useState(''); const [leftThreshold, setLeftThreshold] = useState(0.75); const [rightThreshold, setRightThreshold] = useState(0.3);
  const leftResult = useResource(() => leftId ? api.run(leftId) : Promise.resolve(null), [leftId]); const rightResult = useResource(() => rightId ? api.run(rightId) : Promise.resolve(null), [rightId]);
  if (catalog.loading) return <Loading label="Loading runs" />; if (catalog.error) return <Notice tone="danger" title="Could not load runs"><p>{catalog.error.message}</p></Notice>;
  const runs = catalog.data.runs || []; const left = runs.find((run) => run.id === leftId); const right = runs.find((run) => run.id === rightId); const ready = left && right && leftResult.data && rightResult.data;
  const leftAnalysis = ready ? videoAnalysis(leftResult.data, leftThreshold) : null; const rightAnalysis = ready ? videoAnalysis(rightResult.data, rightThreshold) : null; const warnings = ready ? comparisonWarnings(leftResult.data, rightResult.data) : [];
  const comparisonSections = ready ? sections.map((section) => ({ ...section, rows: section.rows.map((row) => { const first = row.value(leftAnalysis, leftThreshold, leftResult.data); const second = row.value(rightAnalysis, rightThreshold, rightResult.data); const delta = difference(first, second, row.direction, row.digits); return { label: row.label, left: display(first, row.digits), right: display(second, row.digits), difference: delta.text, tone: delta.tone }; }) })) : [];
  const conclusion = ready ? `Run A used threshold ${leftThreshold.toFixed(2)} and produced ${leftAnalysis.events.length} events; Run B used threshold ${rightThreshold.toFixed(2)} and produced ${rightAnalysis.events.length}. Green differences favour Run A, red differences favour Run B, and neutral measures have no automatic preferred direction. Review sensitivity, precision, false positives and runtime together before choosing a threshold.` : '';
  return <div className="page"><div className="page-title"><div><p>Threshold experiment</p><h1>Compare runs</h1><span>Apply independent temporary thresholds to two stored probability timelines. Difference means Run A minus Run B.</span></div></div>
    <Card title="Choose runs and thresholds"><div className="compare-run-grid">{[[leftId, setLeftId, leftThreshold, setLeftThreshold, 'Run A'], [rightId, setRightId, rightThreshold, setRightThreshold, 'Run B']].map(([value, setter, threshold, setThreshold, label]) => <fieldset key={label}><legend>{label}</legend><label><span>Stored run</span><select value={value} onChange={(event) => setter(event.target.value)}><option value="">Select a run</option>{runs.map((run) => <option key={run.id} value={run.id}>{run.recording_id} · {run.compute?.mode?.toUpperCase() || 'device unknown'} · {run.id.slice(0, 8)}</option>)}</select></label><label><span>Analysis threshold</span><div className="compare-threshold"><input type="range" min="0" max="1" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /><input type="number" min="0" max="1" step="0.01" value={threshold} onChange={(event) => { const next = Number(event.target.value); if (next >= 0 && next <= 1) setThreshold(next); }} /></div></label></fieldset>)}</div><p className="comparison-note">Thresholds are temporary and are not saved to either run.</p></Card>
    {!left || !right ? <Empty title="Select two runs">Choose Run A and Run B above.</Empty> : (leftResult.loading || rightResult.loading) ? <Loading label="Recalculating both results" /> : <>{warnings.map((warning) => <Notice tone="danger" title="Fair-comparison warning" key={warning}><p>{warning}</p></Notice>)}<Card title="Complete threshold-aware comparison" action={<button className="button small" type="button" onClick={() => downloadComparisonPdf({ left: leftResult.data, right: rightResult.data, leftThreshold, rightThreshold, sections: comparisonSections, warnings, conclusion })}><FileDown size={15} /> Export comparison PDF</button>}>{comparisonSections.map((section) => <div key={section.title}><h3 className="comparison-section-title">{section.title}</h3><div className="comparison-table difference-table"><div><strong>Measure</strong><strong>Run A</strong><strong>Run B</strong><strong>A - B</strong></div>{section.rows.map((row) => <div key={row.label}><span>{row.label}</span><b>{row.left}</b><b>{row.right}</b><em className={row.tone}>{row.difference}</em></div>)}</div></div>)}<div className="difference-legend"><span className="improved">Green: Run A improved</span><span className="worse">Red: Run A worsened</span><span>Grey: neutral or no preferred direction</span></div><div className="comparison-links"><Link className="button" to={`/runs/${left.id}`}>Open Run A</Link><Link className="button" to={`/runs/${right.id}`}>Open Run B</Link></div></Card></>}
  </div>;
}
