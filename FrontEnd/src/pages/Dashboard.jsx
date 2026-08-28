import { Link } from 'react-router-dom';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';
import { ApiErrorNotice, Card, Loading, Notice, Stat, fmtTime, num } from '../components/ui';

const BADGE = {
  completed: 'bg-green-50 text-green-700',
  blocked: 'bg-red-50 text-red-700',
  skipped: 'bg-slate-100 text-slate-600',
};

export default function Dashboard() {
  const recs = useApi(() => api.recordings(), []);
  const matrix = useApi(() => api.testMatrix(), []);

  if (recs.error) return <ApiErrorNotice error={recs.error} />;
  if (recs.loading) return <Loading />;

  const recordings = recs.data ?? [];
  const tests = matrix.data ?? [];
  const fusionRan = tests.some((t) => t.fusion_used);
  const completed = tests.filter((t) => t.status === 'completed').length;
  const windows = recordings.reduce((a, r) => a + r.window_count, 0);
  const patients = [...new Set(recordings.map((r) => r.patient_id))];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">Overview</h2>
        <p className="mt-1 text-sm text-slate-500">Evidence recorded to date</p>
      </div>

      {recordings.length === 0 ? (
        <Notice tone="slate" title="No recordings analysed yet">
          Run inference in the model repository, then reload.
          <p className="mt-2 font-mono text-xs">python -m seizure_multimodal test --mode video</p>
        </Notice>
      ) : (
        <>
          <Notice tone="amber" title="Video-only evaluation">
            The pretrained VSViG checkpoint was run without training or fine-tuning. EEG evaluation
            and multimodal fusion have not produced results — see the test matrix below.
          </Notice>

          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Recordings analysed" value={recordings.length} sub={patients.join(', ')} />
            <Stat label="Windows scored" value={windows.toLocaleString()} sub="non-overlapping" />
            <Stat label="Tests completed" value={`${completed} of ${tests.length}`} sub="see matrix" />
            <Stat label="Multimodal fusion" unavailable={!fusionRan}
                  value={fusionRan ? 'ran' : null} sub="Not a verified paired source" />
          </div>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
            <Card label="Test matrix" className="lg:col-span-2">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-slate-400">
                    <th className="pb-3 font-medium">Test</th>
                    <th className="pb-3 font-medium">Status</th>
                    <th className="pb-3 font-medium">Note</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {tests.length === 0 ? (
                    <tr><td colSpan={3} className="py-4 text-sm text-slate-400">No summary available.</td></tr>
                  ) : tests.map((t) => (
                    <tr key={t.test}>
                      <td className="py-3 font-medium text-slate-800">{t.test}</td>
                      <td className="py-3">
                        <span className={`rounded-md px-2 py-1 text-xs font-medium ${BADGE[t.status] ?? BADGE.skipped}`}>
                          {t.status}
                        </span>
                      </td>
                      <td className="py-3 text-xs text-slate-500">{t.reason ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <Card label="Latest recording">
              <dl className="space-y-2.5 text-sm">
                {[
                  ['Patient', recordings[0].patient_id],
                  ['Seizure', [recordings[0].seizure_id, recordings[0].seizure_type].filter(Boolean).join(' · ')],
                  ['Duration', fmtTime(recordings[0].duration_sec)],
                  ['Resolution', `${recordings[0].width}×${recordings[0].height}`],
                  ['Frame rate', `${num(recordings[0].fps, 2)} fps`],
                  ['EEG onset', recordings[0].eeg_onset_sec !== null ? `${recordings[0].eeg_onset_sec}s` : '—'],
                  ['Clinical onset', recordings[0].clinical_onset_sec !== null ? `${recordings[0].clinical_onset_sec}s` : '—'],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <dt className="text-slate-500">{k}</dt>
                    <dd className="font-medium text-slate-800">{v}</dd>
                  </div>
                ))}
                <div className="flex justify-between border-t border-slate-100 pt-2">
                  <dt className="text-slate-500">Seizure offset</dt>
                  <dd className="italic text-slate-400">
                    {recordings[0].seizure_offset_sec !== null
                      ? `${recordings[0].seizure_offset_sec}s`
                      : 'not annotated'}
                  </dd>
                </div>
              </dl>
              <Link
                to={`/results/${recordings[0].recording_id}`}
                className="mt-4 block rounded-lg bg-blue-600 py-2 text-center text-sm font-medium text-white hover:bg-blue-700"
              >
                Open analysis
              </Link>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}