import { useEffect, useRef, useState } from 'react';
import { Empty, formatTime } from './components';

async function read(url, signal) {
  const response = await fetch(url, { signal }); const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Evidence unavailable.');
  return data;
}
function EEGWaveform({ runId, selected, playhead }) {
  const [channel, setChannel] = useState('0'); const [data, setData] = useState(null); const [error, setError] = useState('');
  const [names, setNames] = useState([]);
  const start = Math.max(0, selected.start_sec - 4); const end = Math.min(start + 30, selected.end_sec + 4);
  useEffect(() => {
    const controller = new AbortController(); setData(null); setError('');
    read(`/api/runs/${runId}/evidence/eeg?start=${start}&end=${end}&channel=${channel}`, controller.signal).then(value => { setData(value); setNames(value.channel_names); }).catch(error => { if (error.name !== 'AbortError') setError(error.message); });
    return () => controller.abort();
  }, [runId, start, end, channel]);
  return <div>
    <label>EEG channel <select value={channel} onChange={event => setChannel(event.target.value)}><option value="all">All channels</option>{names.length ? names.map((name, index) => <option value={String(index)} key={index}>{name}</option>) : <option value="0">Channel 1</option>}</select></label>
    <p>Original recorded channels, not the model's processed montage. Each trace has its own labelled amplitude scale. Min/max display bins preserve peaks.</p>
    {error ? <p role="alert">{error}</p> : !data ? <p role="status">Loading {channel === 'all' ? 'all EEG channels' : 'EEG samples'}…</p> : <div className={channel === 'all' ? 'eeg-channel-stack' : ''} role="region" aria-label={channel === 'all' ? 'All EEG channels, scroll to view more' : 'EEG trace'} tabIndex={channel === 'all' ? 0 : undefined}>
      {(data.channels || [data]).map(trace => <div className="eeg-channel-row" key={trace.channel}><EEGTrace data={trace} selected={selected} playhead={playhead} start={start} end={end} /></div>)}
    </div>}
    <p>Blue shading marks the same selected prediction window on every channel. Blank intervals have no recorded samples.{channel === 'all' && ' Scroll within the viewer to see all channels.'}</p>
  </div>;
}
function EEGTrace({ data, selected, playhead, start, end }) {
  const points = data.traces.flat();
  const low = points.reduce((value, point) => Math.min(value, point[1]), Infinity);
  const high = points.reduce((value, point) => Math.max(value, point[2]), -Infinity);
  const span = Math.max(high - low, .001);
  const x = time => 60 + 900 * (time - start) / (end - start);
  const y = value => 210 - 170 * (value - low) / span;
  return <div>
    <strong>{data.name || data.channel_names?.[data.channel]} · {data.unit} · {data.sampling_rate} Hz</strong>

    {!points.length ? <Empty title="No EEG samples in this interval">The recording may contain an acquisition gap.</Empty> : <svg viewBox="0 0 1000 270" role="img" aria-label={`EEG channel ${data.name || data.channel_names?.[data.channel]}, selected window ${selected.start_sec} to ${selected.end_sec} seconds`} className="eeg-evidence-chart">
      <rect x={x(selected.start_sec)} y="25" width={Math.max(0, x(Math.min(end, selected.end_sec)) - x(selected.start_sec))} height="195" fill="#2563eb" opacity="0.22" />
      {data.traces.map((trace, index) => <g key={index}><polyline points={trace.map(([time, min, max]) => [x(time), y((min + max) / 2)].join(',')).join(' ')} fill="none" stroke="#38bdf8" strokeWidth="1" /><path d={trace.map(([time, min, max]) => `M${x(time)},${y(min)}L${x(time)},${y(max)}`).join(' ')} fill="none" stroke="#38bdf8" strokeWidth="1.2" /></g>)}
      {playhead != null && playhead >= start && playhead <= end && <line x1={x(playhead)} x2={x(playhead)} y1="25" y2="220" stroke="#fbbf24" strokeWidth="2" />}
      <text x="5" y="45" fill="currentColor" fontSize="13">{high.toFixed(1)}</text><text x="5" y="212" fill="currentColor" fontSize="13">{low.toFixed(1)}</text>
      {[0,1,2,3,4].map(index => <text key={index} x={x(start + (end-start)*index/4)} y="250" textAnchor="middle" fill="currentColor" fontSize="14">{formatTime(start + (end-start)*index/4)}</text>)}
    </svg>}

  </div>;
}
function VideoEvidence({ runId, selected, videos, onTime }) {
  const overlapping = videos.filter(video => video.global_end_sec > selected.start_sec && video.global_start_sec < selected.end_sec);
  const [choice, setChoice] = useState(null);
  const segment = overlapping.find(video => video.input_index === choice) || overlapping[0];
  const [preview, setPreview] = useState(null); const [error, setError] = useState(''); const ref = useRef(null); const playingWindow = useRef(false);
  useEffect(() => {
    setPreview(null); setError('');
    if (!segment) return;
    const controller = new AbortController(); let timer;
    async function poll() {
      try {
        const state = await read(`/api/runs/${runId}/evidence/video/${segment.input_index}/preview`, controller.signal);
        setPreview(state);
        if (state.status === 'preparing' && !controller.signal.aborted) timer = setTimeout(poll, 2000);
      } catch (error) { if (error.name !== 'AbortError') setError(error.message); }
    }
    poll(); return () => { controller.abort(); clearTimeout(timer); };
  }, [runId, segment?.input_index]);
  const seek = () => {
    if (!ref.current || !segment) return;
    ref.current.pause(); playingWindow.current = false;
    ref.current.currentTime = Math.max(0, selected.start_sec - segment.global_start_sec);
    onTime(Math.max(selected.start_sec, segment.global_start_sec));
  };
  useEffect(seek, [selected, segment?.input_index]);
  if (!segment) return <Empty title="No synchronized video for this window">No retained video segment overlaps this interval. EEG evidence may still be available.</Empty>;
  const left = Math.max(selected.start_sec, segment.global_start_sec); const right = Math.min(selected.end_sec, segment.global_end_sec);
  return <div>
    {overlapping.length > 1 && <label>Video segment <select value={segment.input_index} onChange={event => setChoice(Number(event.target.value))}>{overlapping.map(video => <option key={video.input_index} value={video.input_index}>{video.filename}</option>)}</select></label>}
    <p>{segment.filename} · Recording time {formatTime(left)}–{formatTime(right)} ({(right-left).toFixed(2)} seconds available in this segment)</p>
    {error || preview?.status === 'failed' ? <p role="alert">{error || preview.error}</p> : preview?.status !== 'ready' ? <p role="status">Preparing browser video…</p> : <>
      <video ref={ref} src={preview.url} controls preload="metadata" className="evidence-video" onLoadedMetadata={seek} onError={() => setError('The browser could not play this video preview.')} onTimeUpdate={() => {
        const time = ref.current.currentTime + segment.global_start_sec; onTime(time);
        if (playingWindow.current && time >= right) { ref.current.pause(); playingWindow.current = false; }
      }} />
      <button type="button" className="button" onClick={() => { seek(); playingWindow.current = true; ref.current.play().catch(error => setError(error.message)); }}>Play selected window</button>
      <p>Playback is mapped to the EEG recording clock. Highlighted table rows overlap the current playback time.</p>
    </>}
  </div>;
}
export default function FusionEvidence({ runId, selected, playhead, onTime }) {
  const [tab, setTab] = useState('video'); const [sources, setSources] = useState(null); const [error, setError] = useState('');
  useEffect(() => { const controller = new AbortController(); setSources(null); setError(''); read(`/api/runs/${runId}/evidence`, controller.signal).then(setSources).catch(error => { if (error.name !== 'AbortError') setError(error.message); }); return () => controller.abort(); }, [runId]);
  return <section className="fusion-evidence">
    <h3>Evidence for the selected window</h3>
    <div className="timeline-tabs" role="tablist" aria-label="Evidence type">{['video', 'eeg'].map((value, index) => <button className="button" type="button" role="tab" id={`evidence-tab-${value}`} aria-selected={value === tab} aria-controls="evidence-panel" tabIndex={value === tab ? 0 : -1} key={value} onClick={() => setTab(value)} onKeyDown={event => { if (['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) { event.preventDefault(); const next = event.key === 'Home' ? 'video' : event.key === 'End' ? 'eeg' : index ? 'video' : 'eeg'; setTab(next); document.getElementById(`evidence-tab-${next}`)?.focus(); } }}>{value === 'video' ? 'Video playback' : 'EEG waveform'}</button>)}</div>
    <p>Selected: <strong>{formatTime(selected.start_sec)}–{formatTime(selected.end_sec)}</strong> · {(selected.end_sec-selected.start_sec).toFixed(2)} seconds{playhead != null && ` · Playback: ${formatTime(playhead)}`}</p>
    <div role="tabpanel" id="evidence-panel" aria-labelledby={`evidence-tab-${tab}`} tabIndex="0">
      {error ? <p role="alert">{error}</p> : !sources ? <p role="status">Loading evidence sources…</p> : tab === 'video' ? <VideoEvidence runId={runId} selected={selected} videos={sources.videos} onTime={onTime} /> : sources.eeg_available ? <EEGWaveform runId={runId} selected={selected} playhead={playhead} /> : <Empty title="EEG waveform unavailable">The raw EEG input for this run is not retained. Predictions remain available below.</Empty>}
    </div>
  </section>;
}
