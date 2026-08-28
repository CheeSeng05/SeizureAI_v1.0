import { fmtTime } from './ui';

/**
 * One bar per analysis window, height = seizure probability.
 *
 * The shaded band is the EEG-to-clinical transition, excluded from scoring by
 * the source model's label policy — showing it makes clear why the confusion
 * matrix has fewer entries than the recording has windows.
 */
export default function Timeline({recording, probability, prediction, threshold}) {
    const { duration_sec: duration, eeg_onset_sec: eeg, clinical_onset_sec: clin } = recording;
    const pct = (t) => ( t === null || t === undefined ? null : (t/duration) * 100 );
    const eegPct = pct(eeg);
    const clinPct = pct(clin);

    const colour = (i, p) => {
        if(!prediction?.[i]) return '#cbd5e1';
        return p >= 0.99 ? '#dc2626' : '#f97316';
    };

    return (
    <div>
      <div className="relative h-24 overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
        {eegPct !== null && clinPct !== null && (
          <div
            className="absolute inset-y-0 z-0 border-x border-amber-300 bg-amber-100/70"
            style={{ left: `${eegPct}%`, width: `${clinPct - eegPct}%` }}
          />
        )}
 
        <div className="absolute inset-0 z-10 flex items-end">
          {probability.map((p, i) => (
            <div
              key={i}
              className="min-w-0 flex-1"
              style={{ height: `${Math.max(2, p * 100)}%`, background: colour(i, p) }}
            />
          ))}
        </div>
 
        {eegPct !== null && (
          <div className="absolute inset-y-0 z-20 border-l-2 border-dashed border-amber-600"
               style={{ left: `${eegPct}%` }} />
        )}
        {clinPct !== null && (
          <div className="absolute inset-y-0 z-20 border-l-2 border-red-600"
               style={{ left: `${clinPct}%` }} />
        )}
        <div className="absolute inset-x-0 z-20 border-t border-dashed border-slate-400"
             style={{ bottom: `${threshold * 100}%` }} />
      </div>
 
      <div className="relative mt-1 h-10 text-[11px] text-slate-500">
        <span className="absolute left-0">0:00</span>
        {eegPct !== null && (
          <span className="absolute -translate-x-1/2 font-medium text-amber-700"
                style={{ left: `${eegPct}%` }}>
            EEG onset<br />{fmtTime(eeg)}
          </span>
        )}
        {clinPct !== null && (
          <span className="absolute -translate-x-1/2 font-medium text-red-700"
                style={{ left: `${clinPct}%` }}>
            Clinical onset<br />{fmtTime(clin)}
          </span>
        )}
        <span className="absolute right-0">{fmtTime(duration)}</span>
      </div>
 
      <div className="mt-6 flex flex-wrap gap-4 text-xs text-slate-500">
        {[
          ['bg-slate-300', 'Below threshold'],
          ['bg-orange-500', 'Alarm'],
          ['bg-red-600', 'Alarm ≥ 0.99'],
          ['bg-amber-100 border border-amber-300', 'Transition — excluded from scoring'],
        ].map(([cls, label]) => (
          <span key={label} className="flex items-center gap-1.5">
            <i className={`h-2 w-3 rounded-sm ${cls}`} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
