import { AlertTriangle, Info, LoaderCircle } from 'lucide-react';

export function Card({ title, action, className = '', children }) {
  return <section className={`card ${className}`}>
    {(title || action) && <header className="card-head"><h3>{title}</h3>{action}</header>}
    <div className="card-body">{children}</div>
  </section>;
}

export function Notice({ tone = 'info', title, children }) {
  const Icon = tone === 'danger' ? AlertTriangle : Info;
  return <div className={`notice ${tone}`}><Icon size={18} /><div>{title && <strong>{title}</strong>}{children}</div></div>;
}

export function Loading({ label = 'Loading' }) {
  return <div className="loading"><LoaderCircle size={20} /> {label}…</div>;
}

export function Empty({ title, children }) {
  return <div className="empty"><strong>{title}</strong><p>{children}</p></div>;
}

export function Badge({ status }) {
  return <span className={`badge ${status || 'unknown'}`}><i />{status || 'unknown'}</span>;
}

export function Stat({ label, value, detail }) {
  return <div className="stat"><span>{label}</span><strong>{value ?? '—'}</strong>{detail && <small>{detail}</small>}</div>;
}

export const formatBytes = (bytes) => {
  if (bytes == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = Number(bytes); let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(index ? 1 : 0)} ${units[index]}`;
};

export const formatDate = (value) => value ? new Date(value).toLocaleString() : '—';
export const formatTime = (seconds) => seconds == null ? '—' : `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
export const formatDuration = (seconds) => {
  if (seconds == null) return '—';
  if (seconds < 60) return `${Number(seconds).toFixed(2)}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remaining = Math.floor(seconds % 60);
  return hours ? `${hours}h ${minutes}m ${remaining}s` : `${minutes}m ${remaining}s`;
};
export const percent = (value) => value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
