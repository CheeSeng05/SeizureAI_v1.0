import { useRef, useState } from 'react';
import { Activity, CheckCircle2, FileSpreadsheet, FileVideo, UploadCloud, X } from 'lucide-react';
import { api } from '../api';
import { Card, Notice, formatBytes } from '../components';

const modes = [
  { id: 'video', title: 'Video', detail: 'Run the video model only.', icon: FileVideo },
  { id: 'eeg', title: 'EEG', detail: 'Run the EEG model on CSV data or an EDF/Nicolet recording.', icon: Activity },
  { id: 'fusion', title: 'Video + EEG', detail: 'Register synchronized video segments and an EDF or Nicolet EEG recording.', icon: CheckCircle2 },
];

function FileChoice({ title, detail, accept, file, inputRef, onChoose, onRemove, icon: Icon }) {
  return <Card title={title}>
    <div className="compact-dropzone" onClick={() => inputRef.current.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); onChoose(event.dataTransfer.files[0] || null); }}>
      <input ref={inputRef} type="file" hidden accept={accept} onChange={(event) => onChoose(event.target.files[0] || null)} />
      <UploadCloud /><strong>{file ? 'Replace file' : 'Drop file here'}</strong><span>{detail}</span>
    </div>
    {file && <div className="selected-label"><Icon /><div><strong>{file.name}</strong><span>{formatBytes(file.size)}</span></div><button type="button" onClick={(event) => { event.stopPropagation(); onRemove(); }}><X /></button></div>}
  </Card>;
}

function MultiFileChoice({ files, inputRef, onChoose, onRemove }) {
  const addFiles = (incoming) => {
    const merged = [...files];
    Array.from(incoming || []).forEach((file) => {
      if (!merged.some((item) => item.name.toLowerCase() === file.name.toLowerCase())) merged.push(file);
    });
    onChoose(merged);
  };
  return <Card title="Video segments">
    <div className="compact-dropzone" onClick={() => inputRef.current.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); addFiles(event.dataTransfer.files); }}>
      <input ref={inputRef} type="file" hidden multiple accept=".mp4,.avi,.mkv,.mov" onChange={(event) => addFiles(event.target.files)} />
      <UploadCloud /><strong>{files.length ? 'Add more videos' : 'Choose synchronized videos'}</strong><span>One or more AVI, MP4, MKV or MOV segments</span>
    </div>
    {files.map((file) => <div className="selected-label" key={`${file.name}-${file.size}`}><FileVideo /><div><strong>{file.name}</strong><span>{formatBytes(file.size)}</span></div><button type="button" onClick={() => onRemove(file)}><X /></button></div>)}
  </Card>;
}

export default function UploadPage() {
  const videoInput = useRef(null); const eegInput = useRef(null); const labelInput = useRef(null);
  const [mode, setMode] = useState('video');
  const [video, setVideo] = useState(null); const [videos, setVideos] = useState([]); const [eeg, setEeg] = useState(null); const [labelFile, setLabelFile] = useState(null);
  const [compareLabels, setCompareLabels] = useState(false);
  const [videoOffsets, setVideoOffsets] = useState({});
  const [labelSource, setLabelSource] = useState('file');
  const [intervals, setIntervals] = useState([{ start: '', end: '', label: '1' }]);
  const [fields, setFields] = useState({ dataset_id: 'ui-recordings', dataset_name: 'UI uploaded recordings', patient_id: 'local-patient-001', recording_id: '', seizure_id: '', seizure_type: '' });
  const [state, setState] = useState({ busy: false, error: null, uploaded: null });
  const needsVideo = mode !== 'eeg'; const needsEeg = mode !== 'video';
  const manualLabels = mode === 'fusion' && labelSource === 'manual';
  const labelIsCsv = manualLabels || /\.csv$/i.test(labelFile?.name || '');
  const intervalsReady = intervals.length > 0 && intervals.every((row) => row.start !== '' && row.end !== '' && Number.isFinite(Number(row.start)) && Number.isFinite(Number(row.end)) && Number(row.start) >= 0 && Number(row.end) > Number(row.start)) && [...intervals].sort((a, b) => Number(a.start) - Number(b.start)).every((row, index, rows) => !index || Number(row.start) >= Number(rows[index - 1].end));
  const labelReady = !needsVideo || !compareLabels || ((manualLabels ? intervalsReady : labelFile) && fields.patient_id && (labelIsCsv || (fields.seizure_id && fields.seizure_type)));
  const selectedVideos = mode === 'fusion' ? videos : (video ? [video] : []);
  const edfFusion = mode === 'fusion' && /\.edf$/i.test(eeg?.name || '');
  const offsetsReady = !edfFusion || videos.every((file) => videoOffsets[file.name] !== undefined && videoOffsets[file.name] !== '' && Number.isFinite(Number(videoOffsets[file.name])) && Number(videoOffsets[file.name]) >= 0);
  const ready = (!needsVideo || selectedVideos.length > 0) && (!needsEeg || eeg) && labelReady && offsetsReady && (mode !== 'fusion' || fields.recording_id);
  const submit = async () => {
    setState({ busy: true, error: null, uploaded: null });
    const labels = manualLabels ? new File(['start_sec,end_sec,label\n' + [...intervals].sort((a, b) => Number(a.start) - Number(b.start)).map((row) => `${Number(row.start)},${Number(row.end)},${row.label}`).join('\n') + '\n'], 'human_annotations.csv', { type: 'text/csv' }) : labelFile;
    try { setState({ busy: false, error: null, uploaded: await api.upload(mode, selectedVideos, eeg, { ...fields, ...(edfFusion ? { video_start_offsets: JSON.stringify(Object.fromEntries(videos.map((file) => [file.name, Number(videoOffsets[file.name])]))) } : {}) }, needsVideo && compareLabels ? labels : null) }); }
    catch (error) { setState({ busy: false, error, uploaded: null }); }
  };
  return <div className="page narrow">
    <div className="page-title"><div><p>Data registration</p><h1>Upload recordings</h1><span>Register video, EEG, or a verified synchronized multimodal recording.</span></div></div>
    {state.error && <Notice tone="danger" title="Upload failed"><p>{state.error.message}</p></Notice>}
    {state.uploaded && <Notice title="Upload complete"><p>{state.uploaded.uploaded.length} input file{state.uploaded.uploaded.length === 1 ? '' : 's'} stored and registered successfully.</p></Notice>}
    <div className="mode-picker">{modes.map(({ id, title, detail, icon: Icon }) => <button type="button" className={mode === id ? 'selected' : ''} key={id} onClick={() => { setMode(id); if (id === 'eeg') { setCompareLabels(false); setLabelFile(null); } }}><Icon /><span><strong>{title}</strong><small>{detail}</small></span></button>)}</div>
    {mode === 'fusion' && <Notice title="Synchronization required"><p>Use recordings from the same patient and session. Nicolet .e uses embedded video timing; EDF requires you to enter each video start time relative to EEG start. Enter 0 only when both recordings start together.</p></Notice>}
    <div className="upload-inputs">
      {needsVideo && (mode === 'fusion' ? <MultiFileChoice files={videos} inputRef={videoInput} onChoose={setVideos} onRemove={(file) => setVideos(videos.filter((item) => item !== file))} /> : <FileChoice title="Video file" detail="MP4, AVI, MKV or MOV" accept=".mp4,.avi,.mkv,.mov" file={video} inputRef={videoInput} onChoose={setVideo} onRemove={() => setVideo(null)} icon={FileVideo} />)}
      {needsEeg && <FileChoice title="EEG recording or dataset" detail={mode === 'fusion' ? 'EDF or Nicolet .e recording' : 'CSV (X1-X178 + y), EDF, or Nicolet .e'} accept={mode === 'fusion' ? '.edf,.e' : '.csv,.edf,.e'} file={eeg} inputRef={eegInput} onChoose={setEeg} onRemove={() => setEeg(null)} icon={FileSpreadsheet} />}
    </div>
    {edfFusion && <Card title="EDF video synchronization"><p>Enter where each video begins on the EEG timeline, in seconds from EEG start. These timings are human-declared.</p><div className="form-grid">{videos.map((file) => <label key={file.name}><span>{file.name} · start in EEG seconds</span><input type="number" min="0" step="0.001" placeholder="e.g. 0 or 12.5" value={videoOffsets[file.name] ?? ''} onChange={(event) => setVideoOffsets({ ...videoOffsets, [file.name]: event.target.value })} /></label>)}</div></Card>}
    <Card title="Catalog identity">
      <div className="form-grid">{[['dataset_id', 'Dataset ID'], ['dataset_name', 'Dataset name'], ['patient_id', 'Patient ID'], ['recording_id', `Recording ID${mode === 'fusion' ? '' : ' (optional)'}`]].map(([key, label]) => <label key={key}><span>{label}</span><input value={fields[key]} onChange={(event) => setFields({ ...fields, [key]: event.target.value })} /></label>)}</div>
      {mode === 'fusion' && <Notice><p>Recording ID is required. Nicolet timing is checked against embedded metadata. EDF timing is declared by you and checked against the EEG and video durations.</p></Notice>}
      {needsVideo && <div className={`comparison-option ${compareLabels ? 'enabled' : ''}`}><div><strong>Compare {mode === 'fusion' ? 'video + EEG' : 'video'} with ground truth</strong><span>Use timestamped CSV intervals or a matching Label.xlsx workbook.</span></div><label className="switch"><input type="checkbox" checked={compareLabels} onChange={(event) => { setCompareLabels(event.target.checked); if (!event.target.checked) setLabelFile(null); }} /><i /></label></div>}
      {needsVideo && compareLabels && mode === 'fusion' && <div className="title-actions"><button type="button" className="button" onClick={() => setLabelSource('file')}>Upload labels</button><button type="button" className="button" onClick={() => setLabelSource('manual')}>Type time intervals</button></div>}{needsVideo && compareLabels && manualLabels && <div className="comparison-panel"><p>Enter human-annotated intervals in seconds from EEG start. Add seizure and non-seizure intervals; gaps remain unlabelled. Intervals must not overlap.</p>{intervals.map((row, index) => <div className="form-grid" key={index}>{[['start', 'Start (seconds)'], ['end', 'End (seconds)']].map(([key, title]) => <label key={key}><span>{title}</span><input type="number" min="0" step="0.001" value={row[key]} onChange={(event) => setIntervals(intervals.map((item, i) => i === index ? { ...item, [key]: event.target.value } : item))} /></label>)}<label><span>Annotation</span><select value={row.label} onChange={(event) => setIntervals(intervals.map((item, i) => i === index ? { ...item, label: event.target.value } : item))}><option value="1">Seizure</option><option value="0">Non-seizure</option></select></label><button className="button" type="button" onClick={() => setIntervals(intervals.filter((_, i) => i !== index))}>Remove interval</button></div>)}<button className="button" type="button" onClick={() => setIntervals([...intervals, { start: '', end: '', label: '1' }])}>Add interval</button>{!intervalsReady && <p>Enter valid start/end times without overlapping intervals.</p>}</div>}{needsVideo && compareLabels && !manualLabels && <div className="comparison-panel"><Notice title={labelIsCsv ? 'Interval label matching' : 'Workbook matching'}><p>{labelIsCsv ? 'CSV labels require start_sec, end_sec and label columns. Use seconds from recording start (EEG start for multimodal), with 1 or S for seizure and 0 or N for non-seizure. Unlabelled gaps are excluded. Seizure ID and type are optional for CSV labels.' : 'Patient ID, Seizure ID and Seizure type must match exactly one row in Label.xlsx.'}</p></Notice><div className="label-upload"><div><strong>Ground-truth labels</strong><span>Required - .csv or .xlsx format</span></div><input ref={labelInput} type="file" hidden accept=".csv,.xlsx" onChange={(event) => setLabelFile(event.target.files[0] || null)} /><button className="button" type="button" onClick={() => labelInput.current.click()}>{labelFile ? 'Replace labels' : 'Choose labels'}</button></div>{labelFile && <div className="selected-label"><CheckCircle2 /><div><strong>{labelFile.name}</strong><span>{formatBytes(labelFile.size)}</span></div><button onClick={() => setLabelFile(null)}><X /></button></div>}{!labelIsCsv && <div className="form-grid label-identity">{[['seizure_id', 'Seizure ID, e.g. Sz2'], ['seizure_type', 'Seizure type, e.g. PG']].map(([key, label]) => <label key={key}><span>{label}</span><input value={fields[key]} onChange={(event) => setFields({ ...fields, [key]: event.target.value })} /></label>)}</div>}</div>}
      <button className="button primary full" disabled={!ready || state.busy} onClick={submit}>{state.busy ? 'Uploading...' : <><CheckCircle2 size={17} /> Store and register</>}</button>
    </Card>
  </div>;
}
