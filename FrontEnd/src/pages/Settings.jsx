import { CheckCircle2 } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';
import { ApiErrorNotice, Card, Loading, Notice } from '../components/ui';

export default function Settings() {
  const recs = useApi(() => api.recordings(), []);
  const health = useApi(() => api.health(), []);
  const firstId = recs.data?.[0]?.recording_id ?? null;
  const prov = useApi(
    () => (firstId ? api.provenance(firstId) : Promise.resolve(null)),
    [firstId]
  );

  if (recs.error) return <ApiErrorNotice error={recs.error} />;
  if (recs.loading) return <Loading />;

  const p = prov.data;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">Settings</h2>
        <p className="mt-1 text-sm text-slate-500">Model configuration and provenance</p>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card label="Model">
          {!p ? (
            <p className="text-sm text-slate-400">No run recorded yet.</p>
          ) : (
            <>
              <dl className="space-y-3 text-sm">
                {[
                  ['Test mode', p.test_mode],
                  ['Video model used', p.video_model_used ? 'yes' : 'no'],
                  ['EEG model used', p.eeg_model_used ? 'yes' : 'no'],
                  ['Fusion used', p.fusion_used ? 'yes' : 'no'],
                  ['Pairing type', p.pairing_type],
                  ['Training performed', p.training_performed ? 'yes' : 'no'],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4">
                    <dt className="shrink-0 text-slate-500">{k}</dt>
                    <dd className="text-right font-mono text-xs text-slate-800">{v}</dd>
                  </div>
                ))}
              </dl>
              <div className="mt-4 space-y-3 border-t border-slate-100 pt-4">
                {[['Video checkpoint', p.video_checkpoint], ['EEG checkpoint', p.eeg_checkpoint]].map(
                  ([label, value]) => (
                    <div key={label}>
                      <div className="mb-1 text-xs text-slate-500">{label}</div>
                      <div className="break-all rounded bg-slate-50 p-2 font-mono text-[11px] text-slate-700">
                        {value ?? '—'}
                      </div>
                    </div>
                  )
                )}
              </div>
            </>
          )}
        </Card>

        <Card label="Data provenance">
          <ul className="space-y-2.5 text-sm text-slate-600">
            {[
              'Source: WU-SAHZU-EMU-Video',
              'Patient faces masked by the data provider',
              'Annotations from Label.xlsx — 33 seizure rows',
              'No seizure offsets supplied; none inferred',
              'EEG CSV is an independent benchmark, never paired with video',
            ].map((line) => (
              <li key={line} className="flex gap-2">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-slate-300" />
                {line}
              </li>
            ))}
          </ul>
          {p?.label_policy && (
            <div className="mt-4 border-t border-slate-100 pt-4">
              <div className="mb-1 text-xs text-slate-500">Label policy</div>
              <p className="text-xs leading-relaxed text-slate-600">{p.label_policy}</p>
            </div>
          )}
        </Card>
      </div>

      <Card label="Dashboard configuration">
        <dl className="space-y-3 text-sm">
          {[
            ['Results directory', health.data?.results_dir ?? '—'],
            ['Recordings found', health.data?.recordings ?? '—'],
            ['API status', health.data?.status ?? 'unreachable'],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between gap-4">
              <dt className="shrink-0 text-slate-500">{k}</dt>
              <dd className="break-all text-right font-mono text-xs text-slate-800">{String(v)}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-xs text-slate-500">
          Set the results directory with <span className="font-mono">SEIZUREAI_RESULTS_DIR</span> when
          starting uvicorn.
        </p>
      </Card>

      <Notice tone="amber" title="Interpretation limits">
        Video-only evaluation is a real unimodal experiment, not a clinical validation. Real
        multimodal evaluation requires the same patient, recording, and time base. This video and the
        EEG benchmark are never real vEEG together.
      </Notice>
    </div>
  );
}