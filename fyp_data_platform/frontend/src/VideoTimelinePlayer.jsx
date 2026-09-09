import { useRef, useState } from 'react';
import { PlayCircle } from 'lucide-react';
import { Empty, Notice, formatTime, percent } from './components';
import SkeletonOverlay from './SkeletonOverlay';

function markerPosition(value, start, end) {
  if (value === null || value === undefined || end <= start || value < start || value > end) return null;
  return Math.max(0, Math.min(100, (value - start) / (end - start) * 100));
}

function overlaps(event, start, end) {
  if (start === null || start === undefined) return false;
  if (end !== null && end !== undefined) return event.end_sec > start && event.start_sec < end;
  return event.start_sec <= start && event.end_sec >= start;
}

export default function VideoTimelinePlayer({ source, windows = [], events = [], groundTruth = {}, duration: suppliedDuration, threshold = 0.5 }) {
  const videoRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(suppliedDuration || 0);
  const [playbackError, setPlaybackError] = useState(false);
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState(null);
  const [viewRange, setViewRange] = useState(null);
  const duration = videoDuration || suppliedDuration || windows.at(-1)?.end_sec || 0;
  const viewStart = viewRange?.start ?? 0; const viewEnd = viewRange?.end ?? duration;
  const seek = (time) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, Math.min(duration || time, time));
    videoRef.current.play().catch(() => {});
  };
  const truthStart = groundTruth.clinical_onset_sec ?? groundTruth.eeg_onset_sec;
  const truthEnd = groundTruth.seizure_end_sec;
  const classifiedEvents = events.map((event, index) => ({ ...event, id: `model-${index}`, category: overlaps(event, truthStart, truthEnd) ? 'matched' : 'model-only' }));
  const workbookEntries = [['EEG onset', groundTruth.eeg_onset_sec], ['Clinical onset', groundTruth.clinical_onset_sec], ['Seizure end', groundTruth.seizure_end_sec]]
    .filter(([, time]) => time !== null && time !== undefined)
    .map(([label, time]) => ({ id: `workbook-${label}`, category: 'workbook', label, time }));
  const entries = [...classifiedEvents.map((event) => ({ ...event, time: event.start_sec })), ...workbookEntries]
    .filter((entry) => filter === 'all' || filter === 'high' || entry.category === filter);
  const visibleEntries = entries.filter((entry) => filter !== 'high' || (entry.category !== 'workbook' && entry.score >= 0.8));
  const matchedCount = classifiedEvents.filter((event) => event.category === 'matched').length;
  const currentPosition = markerPosition(currentTime, viewStart, viewEnd);
  const eegPosition = markerPosition(groundTruth.eeg_onset_sec, viewStart, viewEnd);
  const clinicalPosition = markerPosition(groundTruth.clinical_onset_sec, viewStart, viewEnd);
  const endPosition = markerPosition(groundTruth.seizure_end_sec, viewStart, viewEnd);
  const truthStartPosition = clinicalPosition ?? eegPosition;
  const windowIsMatched = (window) => classifiedEvents.some((event) => event.category === 'matched' && window.start_sec < event.end_sec && window.end_sec > event.start_sec);
  const visibleWindows = windows.filter((window) => window.end_sec > viewStart && window.start_sec < viewEnd);
  const selectEntry = (entry) => { setSelected(entry); seek(Math.max(0, entry.time - 3)); };
  const zoomToSelected = () => { if (!selected) return; const start = Math.max(0, (selected.start_sec ?? selected.time) - 10); const end = Math.min(duration, (selected.end_sec ?? selected.time) + 10); setViewRange({ start, end: Math.max(end, start + 1) }); };

  if (!source) return <Empty title="Source video unavailable">The original video is no longer registered for this run.</Empty>;
  return <div className="video-analysis">
    <div className="video-layout">
      <div className="video-frame">
        <div className="skeleton-stage">
        {source.browser_playable && !playbackError ? <video ref={videoRef} controls preload="metadata" src={source.url} onLoadedMetadata={(event) => setVideoDuration(event.currentTarget.duration)} onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)} onError={() => setPlaybackError(true)} /> : <Notice tone="danger" title="Browser preview unavailable"><p>This file's format or codec cannot be played reliably in the browser. The original file remains available to download.</p></Notice>}
        <SkeletonOverlay videoRef={videoRef} url={source.pose_url} />
        </div>
        <div className="video-caption"><strong>{source.filename}</strong><span>{formatTime(currentTime)} / {formatTime(duration)}</span></div>
      </div>
      <div className="detection-sidebar">
        <div className="detection-heading"><div><strong>Events and labels</strong><span>{classifiedEvents.length} model events · {matchedCount} matched</span></div></div>
        <div className="event-filters" aria-label="Filter timeline events">{[['all', 'All'], ['matched', 'Matched'], ['model-only', 'Model only'], ['workbook', 'Workbook'], ['high', 'High confidence']].map(([value, label]) => <button type="button" key={value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>{label}</button>)}</div>
        {!visibleEntries.length ? <Empty title="No events in this view">No model detection or workbook timestamp matches this filter.</Empty> : <div className="detection-jumps">{visibleEntries.map((entry) => <button type="button" className={`${entry.category} ${selected?.id === entry.id ? 'selected' : ''}`} key={entry.id} onClick={() => selectEntry(entry)}><PlayCircle /><span>{entry.category === 'workbook' ? <><strong>{entry.label} · {formatTime(entry.time)}</strong><small>Workbook ground truth · starts 3 seconds before</small></> : <><strong>{formatTime(entry.start_sec)} - {formatTime(entry.end_sec)}</strong><small>{entry.category === 'matched' ? 'Model + workbook match' : 'Model detection only'} · peak {percent(entry.score)}</small></>}</span></button>)}</div>}
      </div>
    </div>
    {selected && <div className={`event-detail ${selected.category}`}><div><strong>{selected.category === 'workbook' ? selected.label : selected.category === 'matched' ? 'Matched model event' : 'Model-only event'}</strong><span>{selected.category === 'workbook' ? `Timestamp ${formatTime(selected.time)}` : `${formatTime(selected.start_sec)}–${formatTime(selected.end_sec)} · duration ${(selected.end_sec - selected.start_sec).toFixed(1)}s · peak ${percent(selected.score)}`}</span></div><div><button className="button small" type="button" onClick={() => selectEntry(selected)}>Play event</button><button className="button small" type="button" onClick={zoomToSelected}>Zoom to event</button></div></div>}
    <div className="timeline-actions"><span>Showing {formatTime(viewStart)}–{formatTime(viewEnd)}</span>{viewRange && <button className="button small" type="button" onClick={() => setViewRange(null)}>Return to full timeline</button>}</div>
    <div className="timeline-legend"><span className="model-only-key">Model only</span>{workbookEntries.length > 0 && <span className="matched-key">Model matched to workbook</span>}{workbookEntries.length > 0 && <span className="truth-key">Workbook ground truth</span>}<span className="threshold-key">Selected threshold</span><span className="playhead-key">Current playback</span></div>
    {!windows.length ? <Empty title="No probability timeline">This run did not produce timestamped video windows.</Empty> : <div className="interactive-timeline">
      <div className="timeline-bars"><i className="threshold-line" style={{ bottom: `${threshold * 100}%` }} title={`Detection threshold ${threshold.toFixed(2)}`} />{truthStartPosition !== null && endPosition !== null && endPosition > truthStartPosition && <i className="truth-region" style={{ left: `${truthStartPosition}%`, width: `${endPosition - truthStartPosition}%` }} />}{eegPosition !== null && <i className="truth-marker eeg" style={{ left: `${eegPosition}%` }} title={`EEG onset ${formatTime(groundTruth.eeg_onset_sec)}`} />}{clinicalPosition !== null && <i className="truth-marker clinical" style={{ left: `${clinicalPosition}%` }} title={`Clinical onset ${formatTime(groundTruth.clinical_onset_sec)}`} />}{currentPosition !== null && <i className="playback-marker" style={{ left: `${currentPosition}%` }} />}{visibleWindows.map((window, index) => { const category = window.prediction ? (windowIsMatched(window) ? 'positive matched' : 'positive model-only') : ''; return <button type="button" key={index} aria-label={`Seek to ${formatTime(window.start_sec)}, probability ${percent(window.probability)}`} title={`${formatTime(window.start_sec)}-${formatTime(window.end_sec)} · probability ${percent(window.probability)}`} className={category} style={{ height: `${Math.max(3, (window.probability || 0) * 100)}%` }} onClick={() => seek(window.start_sec)} />; })}</div>
      <div className="timeline-axis"><span>{formatTime(viewStart)}</span><span>{formatTime((viewStart + viewEnd) / 2)}</span><span>{formatTime(viewEnd)}</span></div>
    </div>}
    {(!source.browser_playable || playbackError) && <a className="button" href={source.url} download={source.filename}>Download original video</a>}
  </div>;
}
