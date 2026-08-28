import { useRef, useState } from 'react';
import { CheckCircle2, FileVideo, UploadCloud, X } from 'lucide-react';
import { api } from '../api';
import { Card, Notice, formatBytes } from '../components';

export default function UploadPage() {
  const input = useRef(null);
  const [files, setFiles] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [fields, setFields] = useState({ dataset_id: 'ui-videos', dataset_name: 'UI uploaded videos', patient_id: 'local-patient-001', recording_id: '' });
  const [state, setState] = useState({ busy: false, error: null, uploaded: null });
  const choose = (items) => { setFiles(Array.from(items)); setState({ busy: false, error: null, uploaded: null }); };
  const submit = async () => {
    setState({ busy: true, error: null, uploaded: null });
    try { setState({ busy: false, error: null, uploaded: (await api.upload(files, fields)).uploaded }); }
    catch (error) { setState({ busy: false, error, uploaded: null }); }
  };
  return <div className="page narrow">
    <div className="page-title"><div><p>Data registration</p><h1>Upload recordings</h1><span>Videos are checksummed, stored in MinIO, and registered in PostgreSQL.</span></div></div>
    {state.error && <Notice tone="danger" title="Upload failed"><p>{state.error.message}</p></Notice>}
    {state.uploaded && <Notice title="Upload complete"><p>{state.uploaded.length} recording{state.uploaded.length === 1 ? '' : 's'} registered successfully.</p></Notice>}
    <div className="upload-grid">
      <Card title="Video files">
        <div className={`dropzone ${dragging ? 'dragging' : ''}`} onClick={() => input.current.click()}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
          onDrop={(event) => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files); }}>
          <input ref={input} type="file" hidden multiple accept=".mp4,.avi,.mkv,.mov" onChange={(event) => choose(event.target.files)} />
          <UploadCloud /><strong>Drop videos here</strong><span>or click to browse · MP4, AVI, MKV, MOV</span>
        </div>
        {files.length > 0 && <div className="file-list">{files.map((file) => <div key={`${file.name}-${file.size}`}><FileVideo /><div><strong>{file.name}</strong><span>{formatBytes(file.size)}</span></div><button onClick={() => setFiles(files.filter((item) => item !== file))}><X /></button></div>)}</div>}
      </Card>
      <Card title="Catalog identity">
        <div className="form-grid">
          {[['dataset_id', 'Dataset ID'], ['dataset_name', 'Dataset name'], ['patient_id', 'Patient ID'], ['recording_id', 'Recording ID (optional)']].map(([key, label]) => <label key={key}><span>{label}</span><input value={fields[key]} onChange={(event) => setFields({ ...fields, [key]: event.target.value })} /></label>)}
        </div>
        <Notice><p>If recording ID is empty, the sanitized filename becomes the recording ID.</p></Notice>
        <button className="button primary full" disabled={!files.length || state.busy} onClick={submit}>
          {state.busy ? 'Uploading…' : <><CheckCircle2 size={17} /> Store and register</>}
        </button>
      </Card>
    </div>
  </div>;
}
