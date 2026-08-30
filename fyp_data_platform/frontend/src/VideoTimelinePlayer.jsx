import { useRef, useState } from 'react';
import { PlayCircle } from 'lucide-react';
import { Empty, Notice, formatTime, percent } from './components';

function markerPosition(value, duration) {
  if (value === null || value === undefined || !duration) return null;
  return Math.max(0, Math.min(100, value / duration * 100));
}

function overlaps(event, start, end) {
  if (start === null || start === undefined) return false;
  if (end !== null && end !== undefined) return event.end_sec > start && event.start_sec < end;
  return event.start_sec <= start && event.end_sec >= start;
}

export default function VideoTimelinePlayer({ source, windows = [], events = [], groundTruth = {}, duration: suppliedDuration }) {
  const videoRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(suppliedDuration || 0);
  const [playbackError, setPlaybackError] = useState(false);
  const [filter, setFilter] = useState('all');
  const duration = videoDuration || suppliedDuration || windows.at(-1)?.end_sec || 0;
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
    .filter((entry) => filter === 'all' || entry.category === filter);
  const matchedCount = classifiedEvents.filter((event) => event.category === 'matched').length;
  const currentPosition = markerPosition(currentTime, duration);
  const eegPosition = markerPosition(groundTruth.eeg_onset_sec, duration);
  const clinicalPosition = markerPosition(groundTruth.clinical_onset_sec, duration);
  const endPosition = markerPosition(groundTruth.seizure_end_sec, duration);
  const truthStartPosition = clinicalPosition ?? eegPosition;
  const windowIsMatched = (window) => classifiedEvents.some((event) => event.category === 'matched' && window.start_sec < event.end_sec && window.end_sec > event.start_sec);

  if (!source) return <Empty title="Source video unavailable">The original video is no longer registered for this run.</Empty>;
  return <div className="video-analysis">
    <div className="video-layout">
      <div className="video-frame">
        {source.browser_playable && !playbackError ? <video ref={videoRef} controls preload="metadata" src={source.url} onLoadedMetadata={(event) => setVideoDuration(event.currentTarget.duration)} onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)} onError={() => setPlaybackError(true)} /> : <Notice tone="danger" title="Browser preview unavailable"><p>This file's format or codec cannot be played reliably in the browser. The original file remains available to download.</p></Notice>}
        <div className="video-caption"><strong>{source.filename}</strong><span>{formatTime(currentTime)} / {formatTime(duration)}</span></div>
      </div>
      <div className="detection-sidebar">
        <div className="detection-heading"><div><strong>Events and labels</strong><span>{classifiedEvents.length} model events · {matchedCount} matched</span></div></div>
        <div className="event-filters" aria-label="Filter timeline events">{[['all', 'All'], ['matched', 'Matched'], ['model-only', 'Model only'], ['workbook', 'Workbook']].map(([value, label]) => <button type="button" key={value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>{label}</button>)}</div>
        {!entries.length ? <Empty title="No events in this view">No model detection or workbook timestamp matches this filter.</Empty> : <div className="detection-jumps">{entries.map((entry) => <button type="button" className={entry.category} key={entry.id} onClick={() => seek(Math.max(0, entry.time - 3))}><PlayCircle /><span>{entry.category === 'workbook' ? <><strong>{entry.label} · {formatTime(entry.time)}</strong><small>Workbook ground truth · starts 3 seconds before</small></> : <><strong>{formatTime(entry.start_sec)} - {formatTime(entry.end_sec)}</strong><small>{entry.category === 'matched' ? 'Model + workbook match' : 'Model detection only'} · peak {percent(entry.score)}</small></>}</span></button>)}</div>}
      </div>
    </div>
    <div className="timeline-legend"><span className="model-only-key">Model only</span>{workbookEntries.length > 0 && <span className="matched-key">Model matched to workbook</span>}{workbookEntries.length > 0 && <span className="truth-key">Workbook ground truth</span>}<span className="playhead-key">Current playback</span></div>
    {!windows.length ? <Empty title="No probability timeline">This run did not produce timestamped video windows.</Empty> : <div className="interactive-timeline">
      <div className="timeline-bars">{truthStartPosition !== null && endPosition !== null && endPosition > truthStartPosition && <i className="truth-region" style={{ left: `${truthStartPosition}%`, width: `${endPosition - truthStartPosition}%` }} />}{eegPosition !== null && <i className="truth-marker eeg" style={{ left: `${eegPosition}%` }} title={`EEG onset ${formatTime(groundTruth.eeg_onset_sec)}`} />}{clinicalPosition !== null && <i className="truth-marker clinical" style={{ left: `${clinicalPosition}%` }} title={`Clinical onset ${formatTime(groundTruth.clinical_onset_sec)}`} />}{currentPosition !== null && <i className="playback-marker" style={{ left: `${currentPosition}%` }} />}{windows.map((window, index) => { const category = window.prediction ? (windowIsMatched(window) ? 'positive matched' : 'positive model-only') : ''; return <button type="button" key={index} aria-label={`Seek to ${formatTime(window.start_sec)}, probability ${percent(window.probability)}`} title={`${formatTime(window.start_sec)}-${formatTime(window.end_sec)} - ${percent(window.probability)}`} className={category} style={{ height: `${Math.max(3, (window.probability || 0) * 100)}%` }} onClick={() => seek(window.start_sec)} />; })}</div>
      <div className="timeline-axis"><span>0:00</span><span>{formatTime(duration / 2)}</span><span>{formatTime(duration)}</span></div>
    </div>}
    {(!source.browser_playable || playbackError) && <a className="button" href={source.url} download={source.filename}>Download original video</a>}
  </div>;
}
