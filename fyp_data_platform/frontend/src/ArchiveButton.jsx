import { useState } from 'react';
import { api } from './api';

export default function ArchiveButton({ kind, id, refresh, disabled }) {
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  return <span><button type="button" className="button small" disabled={disabled || busy} onClick={async (event) => {
    event.preventDefault(); event.stopPropagation();
    if (!window.confirm(kind === 'recordings' ? 'Archive this recording and hide its runs? Restore it later in Settings.' : 'Archive this run? Restore it later in Settings.')) return;
    setBusy(true); setError('');
    try { await api.archive(kind, id); refresh(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }}>{busy ? 'Archiving…' : 'Archive'}</button>{error && <small role="alert">{error}</small>}</span>;
}
