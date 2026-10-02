import { useState } from 'react';
import { Square } from 'lucide-react';
import { api } from './api';

export default function StopRunButton({ id, requested }) {
  const [waiting, setWaiting] = useState(false);
  const [error, setError] = useState(null);
  const stop = async () => {
    setWaiting(true); setError(null);
    try { await api.stopRun('run', id); }
    catch (error) { setError(error.message); setWaiting(false); }
  };
  return <div><button className="button" type="button" disabled={waiting || requested} onClick={stop}><Square size={16} />{waiting || requested ? 'Stopping…' : 'Stop model run'}</button>{error && <p role="alert">{error}</p>}</div>;
}
