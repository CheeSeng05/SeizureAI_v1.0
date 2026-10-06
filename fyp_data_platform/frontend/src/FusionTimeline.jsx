import { useState } from 'react';
import { Card, Empty, formatTime, percent } from './components';
import FusionEvidence from './FusionEvidence';
import { parseTimelineTime, timelinePage } from './fusionTimeline';

const probability = value => value == null ? '—' : Number(value).toFixed(4);
export default function FusionTimeline({ windows, runId }) {
  const [selected, setSelected] = useState(null);
  const [playhead, setPlayhead] = useState(null);
  const [page, setPage] = useState(0);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [range, setRange] = useState({});
  const [error, setError] = useState('');
  const result = timelinePage(windows, page, range);
  function search(event) {
    event.preventDefault();
    const from = parseTimelineTime(start); const to = parseTimelineTime(end);
    if (Number.isNaN(from) || Number.isNaN(to)) { setError('Enter seconds, m:ss, or h:mm:ss (for example 90 or 1:30).'); return; }
    if (from != null && to != null && to <= from) { setError('End time must be after start time.'); return; }
    setError(''); setRange({ start: from, end: to }); setPage(0); setSelected(null); setPlayhead(null);
  }
  function reset() { setStart(''); setEnd(''); setRange({}); setError(''); setPage(0); setSelected(null); setPlayhead(null); }
  const active = selected || result.rows[0];
  return <Card title="Multimodal prediction timeline">
    <p>Select a window to inspect its synchronized video and EEG evidence. All times use the EEG recording clock.</p>
    {active && <FusionEvidence runId={runId} selected={active} playhead={playhead} onTime={setPlayhead} />}
    <h3>Combined prediction windows</h3>
    <form className="timeline-search" onSubmit={search}>
      <label>From time<input value={start} onChange={event => setStart(event.target.value)} placeholder="0:00" aria-describedby="timeline-search-help" /></label>
      <label>To time<input value={end} onChange={event => setEnd(event.target.value)} placeholder="Optional" aria-describedby="timeline-search-help" /></label>
      <button className="button" type="submit">Search time</button><button className="button" type="button" onClick={reset}>Clear</button>
    </form>
    <p id="timeline-search-help">Use seconds, m:ss, or h:mm:ss. Search includes windows overlapping the time range; leave either field blank for an open range.</p>
    {error && <p role="alert">{error}</p>}
    <div>
      {!result.total ? <Empty title={windows.length ? 'No windows in this time range' : 'Timeline unavailable'}>{windows.length ? 'Change the time range or clear the search.' : 'The pipeline did not save fusion windows.'}</Empty> : <div className="timeline-table-scroll"><table className="timeline-table">
        <thead><tr><th>Window</th><th>Duration</th><th>EEG probability</th><th>Video probability</th><th>Fusion probability</th><th>Video coverage</th><th>Decision</th></tr></thead>
        <tbody>{result.rows.map((window, index) => {
          const playing = playhead != null && playhead >= window.start_sec && playhead < window.end_sec;
          return <tr key={window.window_id || `${window.start_sec}-${index}`} className={`${active === window ? 'selected-window' : ''} ${playing ? 'playing-window' : ''}`}>
            <td><button type="button" className="button" aria-pressed={active === window} onClick={() => { setSelected(window); setPlayhead(null); }}>{formatTime(window.start_sec)}–{formatTime(window.end_sec)}</button>{playing && <small> Playing</small>}</td>
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
