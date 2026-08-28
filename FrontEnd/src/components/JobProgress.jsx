import { AlertTriangle, Check, Loader2, X } from 'lucide-react';

const STAGES = [
  'Recording staged',
  'Starting inference',
  'Loading model checkpoints',
  'Inspecting data and labels',
  'Extracting pose keypoints',
  'Running VSViG inference',
  'Writing results',
];

const fmtDuration = (s) => {
  if (s === null || s === undefined) return '—';
  if (s < 60) return `${Math.round(s)}s`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
};

/**
 * Where the run has got to.
 *
 * The model emits no progress stream, so most of this is inferred from log
 * lines, elapsed time, and previous runs. Estimated values say so — a bar that
 * implies precision it does not have is worse than none, because a user cannot
 * then tell a slow run from a stuck one.
 */
export default function JobProgress({ job, onCancel }) {
  if (!job) return null;
  const p = job.progress ?? {};
  const running = ['queued', 'running'].includes(job.status);
  const failed = job.status === 'failed';
  const done = job.status === 'succeeded';
  const current = p.stage_index ?? 0;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className={`rounded-md px-2 py-1 text-xs font-medium ${
            { succeeded: 'bg-green-50 text-green-700',
              failed: 'bg-red-50 text-red-700',
              cancelled: 'bg-slate-100 text-slate-600' }[job.status]
            ?? 'bg-blue-50 text-blue-700'}`}>
            {job.status}
          </span>
          {running && <Loader2 className="h-4 w-4 animate-spin text-blue-500" />}
          <span className="text-sm text-slate-600">{p.stage_label}</span>
        </div>

        <div className="flex items-center gap-4 text-sm">
          <span className="text-slate-500">
            elapsed <span className="font-mono text-slate-800">{fmtDuration(p.elapsed_sec)}</span>
          </span>
          {running && p.eta_sec !== null && p.eta_sec !== undefined && (
            <span className="text-slate-500">
              ~<span className="font-mono text-slate-800">{fmtDuration(p.eta_sec)}</span> left
            </span>
          )}
          {running && onCancel && (
            <button
              type="button" onClick={onCancel}
              className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-sm hover:bg-slate-50"
            >
              <X className="h-3.5 w-3.5" /> Cancel
            </button>
          )}
        </div>
      </div>

      {/* bar */}
      <div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-100">
          {p.fraction !== null && p.fraction !== undefined ? (
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                failed ? 'bg-red-400' : done ? 'bg-green-500' : 'bg-blue-500'
              } ${p.fraction_is_estimate && running ? 'opacity-60' : ''}`}
              style={{ width: `${Math.max(2, p.fraction * 100)}%` }}
            />
          ) : (
            <div className="h-full w-1/3 animate-pulse rounded-full bg-blue-300" />
          )}
        </div>
        <div className="mt-1.5 flex justify-between text-xs text-slate-400">
          <span>
            {p.fraction === null || p.fraction === undefined
              ? 'Progress unknown — no previous run to compare against'
              : p.fraction_is_estimate
                ? `≈${Math.round(p.fraction * 100)}% — estimated from previous runs, not reported by the model`
                : `${Math.round(p.fraction * 100)}% — reported by the model`}
          </span>
        </div>
      </div>

      {/* stage list */}
      <ol className="space-y-2">
        {STAGES.map((label, i) => {
          const state = done || i < current ? 'done' : i === current && running ? 'active' : 'pending';
          return (
            <li key={label} className="flex items-center gap-3 text-sm">
              <span className={`grid h-5 w-5 shrink-0 place-items-center rounded-full text-[10px] ${
                state === 'done' ? 'bg-green-100 text-green-700'
                  : state === 'active' ? 'bg-blue-100 text-blue-700'
                  : 'bg-slate-100 text-slate-400'}`}>
                {state === 'done' ? <Check className="h-3 w-3" />
                  : state === 'active' ? <Loader2 className="h-3 w-3 animate-spin" />
                  : i + 1}
              </span>
              <span className={state === 'pending' ? 'text-slate-400' : 'text-slate-700'}>
                {label}
              </span>
            </li>
          );
        })}
      </ol>

      {p.stalled && running && (
        <div className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            No output for {fmtDuration(p.seconds_since_output)}. This model is silent between
            loading and completion, so that may be normal — check the process is still using CPU
            before cancelling.
          </div>
        </div>
      )}

      {job.error && <p className="text-sm text-red-600">{job.error}</p>}

      <div>
        <div className="mb-1.5 text-xs font-medium text-slate-500">Output</div>
        <pre className="max-h-56 overflow-auto rounded-lg bg-slate-900 p-4 font-mono text-[11px] leading-relaxed text-slate-200">
          {job.log?.length ? job.log.join('\n') : 'Waiting for output…'}
        </pre>
      </div>
    </div>
  );
}