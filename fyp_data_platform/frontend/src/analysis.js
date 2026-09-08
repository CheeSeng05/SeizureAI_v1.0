export function eventsFromWindows(windows) {
  const events = []; let current = null;
  windows.forEach((window) => {
    if (!window.prediction) { if (current) events.push(current); current = null; return; }
    if (!current) current = { start_sec: window.start_sec, end_sec: window.end_sec, score: window.probability || 0 };
    else { current.end_sec = window.end_sec; current.score = Math.max(current.score, window.probability || 0); }
  });
  if (current) events.push(current);
  return events;
}

export function applyThreshold(windows = [], threshold = 0.5) {
  return windows.map((window) => ({ ...window, prediction: window.probability != null && Number(window.probability) >= threshold ? 1 : 0 }));
}

function divide(top, bottom) { return bottom ? top / bottom : 'N/A'; }

export function metricsAtThreshold(original, windows, events, groundTruth = {}) {
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
    true_positives: tp, true_negatives: tn, false_positives: fp, false_negatives: fn,
    labelled_seizure_windows: tp + fn, labelled_non_seizure_windows: tn + fp,
    predicted_seizure_windows: evaluated.filter((window) => window.prediction).length,
    predicted_non_seizure_windows: evaluated.filter((window) => !window.prediction).length,
    evaluated_windows: evaluated.length,
    excluded_windows: windows.length - evaluated.length,
    false_positives_per_hour: eegOnset > 0 ? events.filter((event) => event.end_sec <= eegOnset).length / (eegOnset / 3600) : 'N/A',
    onset_mean_absolute_error_sec: candidates.length && clinicalOnset != null ? Math.abs(candidates[0].start_sec - clinicalOnset) : 'N/A',
  };
}

export function videoAnalysis(data, threshold) {
  const windows = applyThreshold(data?.windows || [], threshold); const events = eventsFromWindows(windows);
  const metrics = data?.workflow_metrics?.video?.metrics || (data?.modality === 'video' ? data?.metrics : null);
  const probabilities = windows.map((window) => Number(window.probability)).filter(Number.isFinite);
  return { windows, events, maxProbability: probabilities.length ? Math.max(...probabilities) : null, metrics: metricsAtThreshold(metrics, windows, events, data?.ground_truth || {}) };
}
