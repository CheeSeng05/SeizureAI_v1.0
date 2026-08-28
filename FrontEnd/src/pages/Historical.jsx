import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Search } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';
import { ApiErrorNotice, Card, Loading, Notice, fmtTime } from '../components/ui';

export default function Historical() {
  const recs = useApi(() => api.recordings(), []);
  const [patient, setPatient] = useState('');
  const [type, setType] = useState('');

  if (recs.error) return <ApiErrorNotice error={recs.error} />;
  if (recs.loading) return <Loading />;

  const all = recs.data ?? [];
  const rows = all.filter(
    (r) =>
      (!patient || r.patient_id.toLowerCase().includes(patient.toLowerCase())) &&
      (!type || r.seizure_type === type)
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">Past analyses</h2>
        <p className="mt-1 text-sm text-slate-500">Every completed run</p>
      </div>

      <Card>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <input
            value={patient} onChange={(e) => setPatient(e.target.value)}
            placeholder="Patient ID"
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
          />
          <select
            value={type} onChange={(e) => setType(e.target.value)}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600"
          >
            <option value="">All seizure types</option>
            <option value="P">P — focal</option>
            <option value="PG">PG — partial-to-generalised</option>
          </select>
          <div />
          <button
            type="button"
            onClick={() => { setPatient(''); setType(''); }}
            className="flex items-center justify-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50"
          >
            <Search className="h-4 w-4" /> Clear filters
          </button>
        </div>
      </Card>

      <Card>
        <div className="mb-4 text-xs font-semibold uppercase tracking-widest text-slate-400">
          {rows.length} record{rows.length === 1 ? '' : 's'}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-slate-400">
                {['Patient', 'Seizure', 'Duration', 'Windows', 'EEG onset', 'Clinical onset', ''].map((h, i) => (
                  <th key={i} className="pb-3 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-sm text-slate-400">
                    {all.length ? 'No records match those filters.' : 'No recordings analysed yet.'}
                  </td>
                </tr>
              ) : rows.map((r) => (
                <tr key={r.recording_id}>
                  <td className="py-3.5 font-medium text-slate-800">{r.patient_id}</td>
                  <td className="py-3.5 text-slate-600">
                    {r.seizure_id}
                    {r.seizure_type && (
                      <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-xs">{r.seizure_type}</span>
                    )}
                  </td>
                  <td className="py-3.5 text-slate-600">{fmtTime(r.duration_sec)}</td>
                  <td className="py-3.5 font-mono text-slate-600">{r.window_count.toLocaleString()}</td>
                  <td className="py-3.5 font-mono text-slate-600">{r.eeg_onset_sec ?? '—'}</td>
                  <td className="py-3.5 font-mono text-slate-600">{r.clinical_onset_sec ?? '—'}</td>
                  <td className="py-3.5 text-right">
                    <Link to={`/results/${r.recording_id}`}
                          className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm hover:bg-slate-50">
                      Open
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Notice tone="slate">
        More recordings appear here as they are processed. The WU-SAHZU dataset contains 33 annotated
        seizures across multiple patients; patient-independent evaluation needs several of them.
      </Notice>
    </div>
  );
}