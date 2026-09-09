import { useState } from 'react';
import { api } from './api';
import { useResource } from './hooks';
import { Card, Notice, formatBytes } from './components';

export default function ArchivedItems() {
  const catalog = useResource(api.archived, []);
  const [selection, setSelection] = useState(null), [confirmation, setConfirmation] = useState(''), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const perform = async (action) => { setBusy(true); setError(''); try { await action(); catalog.refresh(); } catch (err) { setError(err.message); } finally { setBusy(false); } };
  return <Card title="Archived items">
    <p>Restore items here or permanently remove their managed copies. Original files on your computer are kept. Deleting a recording also removes its runs, labels and EEG inputs.</p>
    <button className="button small" disabled={busy} onClick={catalog.refresh}>Refresh</button>
    {(error || catalog.error) && <Notice tone="danger"><p role="alert">{error || catalog.error.message}</p></Notice>}
    {catalog.loading ? <p>Loading archived items…</p> : ['recordings', 'runs'].map((kind) => <section key={kind}><h3>Archived {kind}</h3>
      {!catalog.data?.[kind]?.length ? <p>No archived {kind}.</p> : catalog.data[kind].map((item) => <div className="archived-item" key={item.id}>
        <div><strong>{item.recording_source_id || item.recording_id}</strong><small>{item.id}</small>{item.parent_archived && <small>Hidden with archived recording. Restore the recording first.</small>}{item.deleting && <small>Deletion started. Retry deletion to finish cleanup.</small>}</div>
        <button className="button small" disabled={busy || item.deleting || item.parent_archived} onClick={() => perform(() => api.archive(kind, item.id, false))}>Restore</button>
        <button className="button small" disabled={busy || (item.parent_archived && !item.archived)} onClick={() => perform(async () => { const preview = await api.deletionPreview(kind, item.id); setSelection({ ...preview, kind }); setConfirmation(''); })}>{item.deleting ? 'Retry deletion' : 'Delete permanently…'}</button>
      </div>)}
    </section>)}
    {selection && <div className="delete-confirmation" role="region" aria-label="Confirm permanent deletion"><h3>Confirm permanent deletion</h3>
      <p>{selection.runs} runs and {selection.files} files ({formatBytes(selection.bytes)}), plus working copies, will be removed. This cannot be undone.</p>
      <label>Type <strong>{selection.id}</strong> to confirm<input value={confirmation} disabled={busy} onChange={(event) => setConfirmation(event.target.value)} /></label>
      <div><button className="button" disabled={busy} onClick={() => setSelection(null)}>Cancel</button><button className="button" disabled={busy || confirmation !== selection.id} onClick={() => perform(async () => { await api.deleteItem(selection.kind, selection.id, confirmation); setSelection(null); })}>{busy ? 'Removing…' : 'Permanently delete'}</button></div>
    </div>}
  </Card>;
}
