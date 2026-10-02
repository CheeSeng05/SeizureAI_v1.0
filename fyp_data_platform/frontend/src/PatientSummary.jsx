import { useState } from 'react';
import { api } from './api';
import { Card, Notice, formatDate } from './components';

export default function PatientSummary({ runId, patient }) {
  const [saved, setSaved] = useState(patient);
  const [editing, setEditing] = useState(false);
  const [age, setAge] = useState(patient?.age ?? '');
  const [notes, setNotes] = useState(patient?.notes || '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  async function save(event) {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const updated = await api.updatePatient(runId, { age: age === '' ? null : Number(age), notes });
      setSaved(updated); setEditing(false); setMessage('Patient information saved.');
    } catch (error) { setError(error.message); }
    finally { setBusy(false); }
  }
  return <Card title="Patient information" action={saved && !editing && <button className="button" onClick={() => { setAge(saved.age ?? ''); setNotes(saved.notes || ''); setEditing(true); setMessage(''); }}>Edit patient information</button>}>
    {!saved ? <p>No patient is linked to this recording.</p> : <>
      <dl className="metric-list"><div><dt>Patient ID</dt><dd>{saved.id}</dd></div><div><dt>Age (years, as entered)</dt><dd>{saved.age ?? 'Not provided'}</dd></div></dl>
      {editing ? <form className="patient-form" onSubmit={save}>
        <p>Optional information shared across this patient's recordings in this dataset.</p>
        <label>Age (years)<input type="number" min="0" max="130" step="1" value={age} onChange={event => setAge(event.target.value)} disabled={busy} /></label>
        <label>Relevant patient information<textarea rows="4" maxLength="4000" value={notes} onChange={event => setNotes(event.target.value)} placeholder="Relevant history or context for reviewing this recording" disabled={busy} /></label>
        {error && <Notice tone="danger"><p role="alert">{error}</p></Notice>}
        <div className="title-actions"><button className="button" type="submit" disabled={busy}>{busy ? 'Saving…' : 'Save patient information'}</button><button className="button" type="button" disabled={busy} onClick={() => { setEditing(false); setError(''); }}>Cancel</button></div>
      </form> : <><p className="patient-notes">{saved.notes || 'No additional patient information provided.'}</p>{saved.updated_at && <small>Patient information updated {formatDate(saved.updated_at)}</small>}</>}
      {message && <p role="status">{message}</p>}
    </>}
  </Card>;
}
