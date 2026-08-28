import { useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../api/client';
import { useApi, useScoring } from '../hooks/useApi';
import Timeline from '../components/Timeline';
import { ApiErrorNotice, Card, Loading, Notice, fmtTime, num } from '../components/ui';

export default function Results() {
  const { id: routeId } = useParams();
  const [threshold, setThreshold] = useState(0.5);
  const [minConsecutive, setMinConsecutive] = useState(1);

  const list = useApi(() => api.recordings(), []);
  const id = routeId ?? list.data?.[0]?.recording_id ?? null;

  const rec = useApi(() => (id ? api.recording(id) : Promise.resolve(null)), [id]);
  const tl = useApi(() => (id ? api.timeline(id) : Promise.resolve(null)), [id]);
  const scored = useScoring(id, threshold, minConsecutive);

  const atShippedDefault = threshold === 0.5 && minConsecutive === 1;
  const m = scored.data?.metrics ?? null;
  const onsetError = m?.onset_error_sec ?? null;

  const confusion = useMemo(() => {
    const c = m ?? { tn: 0, fp: 0, fn: 0, tp: 0 };
    return [
      { label: 'True negative', value: c.tn, box: 'border-slate-200 bg-slate-50', text: 'text-slate-700' },
      { label: 'False positive', value: c.fp, box: 'border-red-200 bg-red-50', text: 'text-red-700' },
      { label: 'False negative', value: c.fn, box: 'border-red-200 bg-red-50', text: 'text-red-700' },
      { label: 'True positive', value: c.tp, box: 'border-green-200 bg-green-50', text: 'text-green-700' },
    ];
  }, [m]);

  if (list.error) return <ApiErrorNotice error={list.error} />;
  if (list.loading || rec.loading || tl.loading) return <Loading label="Loading analysis" />;

  if (!id || !rec.data) {
    return (
      <Notice tone="slate" title="Nothing analysed yet">
        Run inference in the model repository, then reload this page.
        <p className="mt-2 font-mono text-xs">python -m seizure_multimodal test --mode video</p>
      </Notice>
    );
  }

  const recording = rec.data;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">
          {recording.patient_id} · {recording.seizure_id}{' '}
          {recording.seizure_type && <span className="text-slate-400">({recording.seizure_type})</span>}
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          {recording.source_file} · {fmtTime(recording.duration_sec)} ·{' '}
          {recording.window_count.toLocaleString()} windows of {num(recording.window_sec, 3)}s
        </p>
      </div>

      {recording.seizure_offset_sec === null && (
        <Notice tone="blue">
          No seizure offset is annotated for this recording, so the timeline has no scored end and
          every window after clinical onset counts as ictal. No offset is inferred.
        </Notice>
      )}

      <Card label="Seizure probability — one bar per window">
        {tl.data ? (
          <Timeline
            recording={recording}
            probability={tl.data.probability}
            prediction={scored.data?.prediction ?? tl.data.prediction}
            threshold={threshold}
          />
        ) : (
          <Loading label="Loading timeline" />
        )}
      </Card>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <Card label="Decision threshold" className="lg:col-span-2">
          <div className="space-y-5">
            <div>
              <div className="mb-2 flex justify-between text-sm">
                <label htmlFor="threshold" className="text-slate-600">Probability threshold</label>
                <span className="font-mono font-semibold text-slate-900">{threshold.toFixed(2)}</span>
              </div>
              <input
                id="threshold" type="range" min="0.05" max="0.99" step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full accent-blue-600"
              />
            </div>

            <div>
              <div className="mb-2 flex justify-between text-sm">
                <label htmlFor="persist" className="text-slate-600">Consecutive windows required</label>
                <span className="font-mono font-semibold text-slate-900">{minConsecutive}</span>
              </div>
              <input
                id="persist" type="range" min="1" max="10" step="1"
                value={minConsecutive}
                onChange={(e) => setMinConsecutive(Number(e.target.value))}
                className="w-full accent-blue-600"
              />
              <p className="mt-2 text-xs text-slate-500">
                Requiring a run of windows suppresses isolated spikes. False alarms are typically a
                few seconds; a seizure is sustained.
              </p>
            </div>

            {atShippedDefault && (
              <Notice tone="amber">
                These are the model&apos;s shipped settings. Drag both sliders up to see how far the
                false-alarm rate falls before sensitivity is affected.
              </Notice>
            )}

            {scored.error && <p className="text-sm text-red-600">{scored.error.message}</p>}
          </div>
        </Card>

        <Card label="At these settings">
          <dl className={`space-y-3 text-sm transition-opacity ${scored.loading ? 'opacity-50' : ''}`}>
            {[
              ['Sensitivity', num(m?.sensitivity)],
              ['Specificity', num(m?.specificity)],
              ['Precision', num(m?.precision)],
              ['False alarms/hour', num(m?.false_positives_per_hour, 1)],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between">
                <dt className="text-slate-500">{label}</dt>
                <dd className="font-mono font-semibold text-slate-900">{value}</dd>
              </div>
            ))}

            <div className="flex justify-between border-t border-slate-100 pt-3">
              <dt className="text-slate-500">Detection vs clinical onset</dt>
              <dd className={`font-mono font-semibold ${onsetError !== null && onsetError < 0 ? 'text-green-600' : 'text-slate-900'}`}>
                {onsetError === null ? 'missed' : `${onsetError > 0 ? '+' : ''}${onsetError.toFixed(1)}s`}
              </dd>
            </div>

            {onsetError !== null && onsetError < 0 && (
              <p className="text-xs leading-relaxed text-green-700">
                Alarms {Math.abs(onsetError).toFixed(1)}s before the clinician-marked clinical onset.
              </p>
            )}
          </dl>
        </Card>
      </div>

      <Card label="Confusion — at these settings">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {confusion.map((c) => (
            <div key={c.label} className={`rounded-lg border p-4 text-center ${c.box}`}>
              <div className={`text-2xl font-semibold ${c.text}`}>{c.value}</div>
              <div className="mt-1 text-xs text-slate-500">{c.label}</div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs leading-relaxed text-slate-500">
          Windows between EEG onset and clinical onset are excluded from scoring, following the
          source model&apos;s label policy.
        </p>
      </Card>
    </div>
  );
}