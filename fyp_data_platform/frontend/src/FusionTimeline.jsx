import { useState } from 'react';
import { Card, Empty, formatTime, percent } from './components';
import FusionEvidence from './FusionEvidence';
import { pickerSeconds, timelinePage, selectedWindowAtTime } from './fusionTimeline';

const probability = value => value == null ? '—' : Number(value).toFixed(4);
export default function FusionTimeline({ windows, runId }) {
  const [selected, setSelected] = useState(null);
  const [playhead, setPlayhead] = useState(null);
  const [page, setPage] = useState(0);
  const [start, setStart] = useState(['', '']);
  const [end, setEnd] = useState(['', '']);
  const [decision, setDecision] = useState('all');
  const [range, setRange] = useState({});
  const [error, setError] = useState('');
  const result = timelinePage(windows, page, { ...range, decision });
  function search(event) {
    event.preventDefault();
    const from = pickerSeconds(...start); const to = pickerSeconds(...end);
    if (Number.isNaN(from) || Number.isNaN(to)) { setError('Use whole minutes and seconds from 0 to 59.'); return; }
    if (from != null && to != null && to <= from) { setError('End time must be after start time.'); return; }
    setError(''); setRange({ start: from, end: to }); setPage(0); setSelected(null); setPlayhead(null);
  }
  function reset() { setStart(['', '']); setEnd(['', '']); setDecision('all'); setRange({}); setError(''); setPage(0); setSelected(null); setPlayhead(null); }
  const active = selected || result.rows[0];
  return <Card title="Multimodal prediction timeline">
    <p>Select a window to inspect its synchronized video and EEG evidence. All times use the EEG recording clock.</p>
    {active && <FusionEvidence runId={runId} selected={active} playhead={playhead} onTime={setPlayhead} />}
    <h3>Combined prediction windows</h3>
    <form className="timeline-search" onSubmit={search}>
      {[['From time', start, setStart], ['To time', end, setEnd]].map(([label, values, update]) => <fieldset className="timeline-time-picker" key={label}><legend>{label}</legend><label>Minutes<input type="number" min="0" step="1" value={values[0]} placeholder="Any" aria-label={`${label} minutes`} onChange={event => update([event.target.value, values[1]])} /></label><span>:</span><label>Seconds<select value={values[1]} aria-label={`${label} seconds`} onChange={event => update([values[0], event.target.value])}><option value="">—</option>{Array.from({ length: 60 }, (_, second) => <option key={second} value={String(second)}>{String(second).padStart(2, '0')}</option>)}</select></label></fieldset>)}
      <label>Decision<select value={decision} onChange={event => { setDecision(event.target.value); setPage(0); setSelected(null); setPlayhead(null); }}><option value="all">All decisions</option><option value="positive">Possible seizure</option><option value="negative">No detection</option><option value="eeg-only">EEG only</option></select></label>
      <button className="button" type="submit">Search time</button><button className="button" type="button" onClick={reset}>Clear</button>
    </form>
    <p id="timeline-search-help">Choose elapsed minutes and seconds, then Search time. Leave both parts of a boundary blank for no limit. Decision filtering applies immediately.</p>
    {error && <p role="alert">{error}</p>}
    <div>
      {!result.total ? <Empty title={windows.length ? 'No windows match these filters' : 'Timeline unavailable'}>{windows.length ? 'Change the decision or time range, or clear the filters.' : 'The pipeline did not save fusion windows.'}</Empty> : <div className="timeline-table-scroll"><table className="timeline-table">
        <thead><tr><th>Window</th><th>Duration</th><th>EEG probability</th><th>Video probability</th><th>Fusion probability</th><th>Video coverage</th><th>Decision</th></tr></thead>
        <tbody>{result.rows.map((window, index) => {
          const playing = selectedWindowAtTime(window, active, playhead);
          return <tr key={window.window_id || `${window.start_sec}-${index}`} className={`${active === window ? 'selected-window' : ''} ${playing ? 'playing-window' : ''}`}>
            <td><button type="button" className="button" aria-pressed={active === window} onClick={() => { setSelected(window); setPlayhead(null); }}>{formatTime(window.start_sec)}–{formatTime(window.end_sec)}</button>{playing && <small> Selected playback window</small>}</td>
            <td>{(window.end_sec-window.start_sec).toFixed(2)} s</td>
            <td>{probability(window.eeg_probability)}</td><td>{probability(window.video_probability)}</td><td>{probability(window.fused_probability)}</td><td>{percent(window.video_coverage)}</td>
            <td>{window.fusion_available ? Number(window.fused_prediction) === 1 ? 'Possible seizure' : 'No detection' : 'EEG only'}</td>
          </tr>;
        })}</tbody>
      </table></div>}
    </div>
    <div className="timeline-pagination"><span role="status">{result.total ? `Showing ${result.current * 10 + 1}–${Math.min((result.current + 1) * 10, result.total)} of ${result.total} windows · Page ${result.current + 1} of ${result.pages}` : '0 windows'}</span><button type="button" className="button" disabled={!result.total || result.current === 0} onClick={() => { setPage(result.current - 1); setSelected(null); setPlayhead(null); }}>Previous</button><button type="button" className="button" disabled={!result.total || result.current + 1 >= result.pages} onClick={() => { setPage(result.current + 1); setSelected(null); setPlayhead(null); }}>Next</button></div>
    <p>Download fusion/predictions.csv under Processing details and downloads for the complete traceable timeline.</p>
  </Card>;
}
