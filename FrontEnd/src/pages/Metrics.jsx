import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';
import { ApiErrorNotice, Card, Loading, Notice, Stat, num } from '../components/ui';

const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_PERSIST = 1;

export default function Metrics() {
  const recs = useApi(() => api.recordings(), []);
  const [scored, setScored] = useState({});

  useEffect(() => {
    const list = recs.data;
    if (!list?.length) return;
    let alive = true;
    Promise.all(
      list.map((r) =>
        api.score(r.recording_id, DEFAULT_THRESHOLD, DEFAULT_PERSIST)
          .then((d) => [r.recording_id, d])
          .catch(() => [r.recording_id, null])
      )
    ).then((pairs) => { if (alive) setScored(Object.fromEntries(pairs)); });
    return () => { alive = false; };
  }, [recs.data]);

  if (recs.error) return <ApiErrorNotice error={recs.error} />;
  if (recs.loading) return <Loading />;

  const recordings = recs.data ?? [];
  const aurocs = Object.values(scored)
    .map((s) => s?.metrics?.auroc)
    .filter((v) => typeof v === 'number');
  const meanAuroc = aurocs.length ? aurocs.reduce((a, b) => a + b, 0) / aurocs.length : null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">Evaluation metrics</h2>
        <p className="mt-1 text-sm text-slate-500">
          At the model&apos;s shipped threshold of {DEFAULT_THRESHOLD}
        </p>
      </div>

      {recordings.length === 0 ? (
        <Notice tone="slate" title="Nothing to report yet">
          Metrics appear once at least one recording has been analysed.
        </Notice>
      ) : (
        <>
          <Notice tone="amber" title="Not a validation study">
            These figures describe {recordings.length} recording
            {recordings.length === 1 ? '' : 's'} under the source label policy.
            Patient-independent validation needs several patients.
          </Notice>

          {recordings.map((rec) => {
            const m = scored[rec.recording_id]?.metrics;
            return (
              <Card
                key={rec.recording_id}
                label={`${rec.patient_id} · ${rec.seizure_id} ${rec.seizure_type}`}
              >
                {!m ? (
                  <Loading label="Scoring" />
                ) : (
                  <>
                    <div className="grid grid-cols-2 gap-5 lg:grid-cols-4">
                      <Stat label="AUROC" value={num(m.auroc)} unavailable={m.auroc === null}
                            sub="threshold-free" />
                      <Stat label="Sensitivity" value={num(m.sensitivity)}
                            unavailable={m.sensitivity === null} sub="recall on ictal windows" />
                      <Stat label="Specificity" value={num(m.specificity)}
                            unavailable={m.specificity === null}
                            tone={m.specificity !== null && m.specificity < 0.7 ? 'red' : 'slate'}
                            sub="on pre-onset windows" />
                      <Stat label="Cohen's κ" value={num(m.cohen_kappa)}
                            unavailable={m.cohen_kappa === null}
                            tone={m.cohen_kappa !== null && m.cohen_kappa < 0.4 ? 'red' : 'slate'}
                            sub="vs clinician annotation" />
                    </div>

                    <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
                      <div className="rounded-lg border border-slate-200 p-4">
                        <div className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                          Full metric set
                        </div>
                        <dl className="space-y-2.5 text-sm">
                          {[
                            ['F1', num(m.f1, 4)],
                            ['Precision', num(m.precision, 4)],
                            ['False alarms/hour', num(m.false_positives_per_hour, 1)],
                            ['Merged alarm events', m.alarm_events ?? '—'],
                            ['True positives', m.tp],
                            ['False positives', m.fp],
                            ['False negatives', m.fn],
                            ['True negatives', m.tn],
                          ].map(([k, v]) => (
                            <div key={k} className="flex justify-between">
                              <dt className="text-slate-500">{k}</dt>
                              <dd className="font-mono font-medium text-slate-800">{v}</dd>
                            </div>
                          ))}
                          <div className="flex justify-between border-t border-slate-100 pt-2.5">
                            <dt className="text-slate-500">Offset error</dt>
                            <dd className="italic text-slate-400">
                              {m.offset_error_sec !== null
                                ? `${num(m.offset_error_sec, 2)} s`
                                : 'no reference offset'}
                            </dd>
                          </div>
                        </dl>
                      </div>

                      <div className="rounded-lg border border-slate-200 p-4">
                        <div className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                          Onset detection
                        </div>
                        {m.onset_error_sec === null ? (
                          <p className="py-6 text-center text-sm text-slate-400">
                            No alarm was raised after EEG onset at this threshold.
                          </p>
                        ) : (
                          <div className="py-2 text-center">
                            <div className={`text-4xl font-semibold ${m.onset_error_sec < 0 ? 'text-green-600' : 'text-slate-800'}`}>
                              {m.onset_error_sec > 0 ? '+' : ''}{num(m.onset_error_sec, 2)}s
                            </div>
                            <p className="mt-3 text-sm leading-relaxed text-slate-600">
                              First alarm at <span className="font-mono">{num(m.first_alarm_sec, 1)}s</span>,
                              against a clinical onset of{' '}
                              <span className="font-mono">{num(rec.clinical_onset_sec, 1)}s</span>.
                            </p>
                            {m.onset_error_sec < 0 && (
                              <p className="mt-3 text-xs text-slate-500">
                                The model flagged the seizure before the point a clinician recorded
                                visible symptoms. Indicative, not established.
                              </p>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </Card>
            );
          })}

          <Card label="Per-modality">
            <div className="space-y-4">
              {[
                ['Body pose (VSViG)', meanAuroc, true],
                ['EEG', null, false],
                ['Fusion', null, false],
              ].map(([name, value, ran]) => (
                <div key={name} className="flex items-center gap-4">
                  <span className="w-40 shrink-0 text-sm text-slate-600">{name}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                    {ran && value !== null && (
                      <div className="h-full rounded-full bg-blue-500" style={{ width: `${value * 100}%` }} />
                    )}
                  </div>
                  <span className={`w-24 text-right font-mono text-sm ${ran ? 'text-slate-800' : 'italic text-slate-400'}`}>
                    {ran ? num(value) : 'not run'}
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-5 text-xs leading-relaxed text-slate-500">
              VSViG is skeleton-only. There is no facial branch: the dataset masks patient faces, so
              action units cannot be extracted from this source.
            </p>
          </Card>
        </>
      )}
    </div>
  );
}