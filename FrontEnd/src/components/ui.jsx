import { AlertTriangle, Info } from 'lucide-react';

export function Card({ label, className = '', children }) {
  return (
    <div className={`rounded-xl border border-slate-200 bg-white p-5 ${className}`}>
      {label && (
        <div className="mb-4 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
          {label}
        </div>
      )}
      {children}
    </div>
  );
}

const TONES = {
  amber: 'bg-amber-50 border-amber-200 text-amber-900',
  red: 'bg-red-50 border-red-200 text-red-900',
  blue: 'bg-blue-50 border-blue-200 text-blue-900',
  slate: 'bg-slate-50 border-slate-200 text-slate-600',
};

export function Notice({ tone = 'amber', title, className = '', children }) {
  const Icon = tone === 'blue' || tone === 'slate' ? Info : AlertTriangle;
  return (
    <div className={`flex gap-3 rounded-lg border p-4 ${TONES[tone]} ${className}`}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="text-sm leading-relaxed">
        {title && <div className="mb-1 font-semibold">{title}</div>}
        {children}
      </div>
    </div>
  );
}

/**
 * A statistic with no value renders as an explicit gap, never as zero.
 * On a clinical dashboard "0.000" and "we did not measure this" must not look
 * the same.
 */
export function Stat({ label, value, sub, tone = 'slate', unavailable = false }) {
  const toneClass = {
    red: 'text-red-600',
    amber: 'text-amber-600',
    green: 'text-green-600',
    slate: 'text-slate-900',
  }[tone];

  return (
    <Card>
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
        {label}
      </div>
      {unavailable ? (
        <>
          <div className="text-2xl font-semibold text-slate-300">&mdash;</div>
          <div className="mt-1 text-xs text-slate-400">{sub}</div>
        </>
      ) : (
        <>
          <div className={`text-3xl font-semibold ${toneClass}`}>{value}</div>
          {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
        </>
      )}
    </Card>
  );
}

export function Loading({ label = 'Loading' }) {
  return (
    <div className="flex items-center gap-3 py-12 text-sm text-slate-400">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-200 border-t-blue-500" />
      {label}…
    </div>
  );
}

export function ApiErrorNotice({ error }) {
  if (!error) return null;
  return (
    <Notice tone="red" title="Analysis API unavailable" className="mb-6">
      {error.message}
      <p className="mt-2 font-mono text-xs">cd backend &amp;&amp; uvicorn app.main:app --reload</p>
    </Notice>
  );
}

export const fmtTime = (s) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export const num = (v, d = 3) =>
  v === null || v === undefined ? '—' : Number(v).toFixed(d);