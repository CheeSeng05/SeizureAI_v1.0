import { useRef, useState } from 'react';
import { Activity, CheckCircle2, FileSpreadsheet, FileVideo, UploadCloud, X } from 'lucide-react';
import { api } from '../api';
import { Card, Notice, formatBytes } from '../components';

const modes = [
  { id: 'video', title: 'Video', detail: 'Run the video model only.', icon: FileVideo },
  { id: 'eeg', title: 'EEG', detail: 'Run the EEG model on a labelled CSV.', icon: Activity },
  { id: 'unimodal', title: 'Both independently', detail: 'Run video and EEG separately. Results are not fused.', icon: CheckCircle2 },
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

export default function UploadPage() {
  const videoInput = useRef(null); const eegInput = useRef(null); const labelInput = useRef(null);
  const [mode, setMode] = useState('video');
  const [video, setVideo] = useState(null); const [eeg, setEeg] = useState(null); const [labelFile, setLabelFile] = useState(null);
  const [compareLabels, setCompareLabels] = useState(false);
  const [fields, setFields] = useState({ dataset_id: 'ui-recordings', dataset_name: 'UI uploaded recordings', patient_id: 'local-patient-001', recording_id: '', seizure_id: '', seizure_type: '' });
  const [state, setState] = useState({ busy: false, error: null, uploaded: null });
  const needsVideo = mode !== 'eeg'; const needsEeg = mode !== 'video';
  const labelReady = !needsVideo || !compareLabels || (labelFile && fields.patient_id && fields.seizure_id && fields.seizure_type);
  const ready = (!needsVideo || video) && (!needsEeg || eeg) && labelReady && (mode !== 'unimodal' || fields.recording_id);
  const submit = async () => {
    setState({ busy: true, error: null, uploaded: null });
    try { setState({ busy: false, error: null, uploaded: await api.upload(mode, video ? [video] : [], eeg, fields, compareLabels ? labelFile : null) }); }
    catch (error) { setState({ busy: false, error, uploaded: null }); }
  };
  return <div className="page narrow">
    <div className="page-title"><div><p>Data registration</p><h1>Upload recordings</h1><span>Choose which independent model workflows to prepare.</span></div></div>
    {state.error && <Notice tone="danger" title="Upload failed"><p>{state.error.message}</p></Notice>}
    {state.uploaded && <Notice title="Upload complete"><p>{state.uploaded.uploaded.length} input file{state.uploaded.uploaded.length === 1 ? '' : 's'} stored and registered successfully.</p></Notice>}
    <div className="mode-picker">{modes.map(({ id, title, detail, icon: Icon }) => <button type="button" className={mode === id ? 'selected' : ''} key={id} onClick={() => { setMode(id); if (id === 'eeg') { setCompareLabels(false); setLabelFile(null); } }}><Icon /><span><strong>{title}</strong><small>{detail}</small></span></button>)}</div>
    {mode === 'unimodal' && <Notice title="Independent results"><p>Both models will run on the same catalog entry, but their predictions will not be combined or presented as fusion.</p></Notice>}
    <div className="upload-inputs">
      {needsVideo && <FileChoice title="Video file" detail="MP4, AVI, MKV or MOV" accept=".mp4,.avi,.mkv,.mov" file={video} inputRef={videoInput} onChoose={setVideo} onRemove={() => setVideo(null)} icon={FileVideo} />}
      {needsEeg && <FileChoice title="EEG dataset" detail="CSV with X1-X178 and ground-truth column y" accept=".csv" file={eeg} inputRef={eegInput} onChoose={setEeg} onRemove={() => setEeg(null)} icon={FileSpreadsheet} />}
    </div>
    <Card title="Catalog identity">
      <div className="form-grid">{[['dataset_id', 'Dataset ID'], ['dataset_name', 'Dataset name'], ['patient_id', 'Patient ID'], ['recording_id', `Recording ID${mode === 'unimodal' ? '' : ' (optional)'}`]].map(([key, label]) => <label key={key}><span>{label}</span><input value={fields[key]} onChange={(event) => setFields({ ...fields, [key]: event.target.value })} /></label>)}</div>
      {mode === 'unimodal' && <Notice><p>Recording ID is required so the video and EEG files are registered under one entry.</p></Notice>}
      {needsVideo && <div className={`comparison-option ${compareLabels ? 'enabled' : ''}`}><div><strong>Compare video with ground truth</strong><span>Enable this only when you have a matching Label.xlsx workbook.</span></div><label className="switch"><input type="checkbox" checked={compareLabels} onChange={(event) => { setCompareLabels(event.target.checked); if (!event.target.checked) setLabelFile(null); }} /><i /></label></div>}
      {needsVideo && compareLabels && <div className="comparison-panel"><Notice title="Workbook matching"><p>Patient ID, Seizure ID and Seizure type must match exactly one row in Label.xlsx.</p></Notice><div className="label-upload"><div><strong>Ground-truth workbook</strong><span>Required - .xlsx format</span></div><input ref={labelInput} type="file" hidden accept=".xlsx" onChange={(event) => setLabelFile(event.target.files[0] || null)} /><button className="button" type="button" onClick={() => labelInput.current.click()}>{labelFile ? 'Replace workbook' : 'Choose workbook'}</button></div>{labelFile && <div className="selected-label"><CheckCircle2 /><div><strong>{labelFile.name}</strong><span>{formatBytes(labelFile.size)}</span></div><button onClick={() => setLabelFile(null)}><X /></button></div>}<div className="form-grid label-identity">{[['seizure_id', 'Seizure ID, e.g. Sz2'], ['seizure_type', 'Seizure type, e.g. PG']].map(([key, label]) => <label key={key}><span>{label}</span><input value={fields[key]} onChange={(event) => setFields({ ...fields, [key]: event.target.value })} /></label>)}</div></div>}
      <button className="button primary full" disabled={!ready || state.busy} onClick={submit}>{state.busy ? 'Uploading...' : <><CheckCircle2 size={17} /> Store and register</>}</button>
    </Card>
  </div>;
}
