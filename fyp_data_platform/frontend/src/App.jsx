import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { useEffect } from 'react';
import { Activity, Brain, Database, FileClock, GitCompare, LayoutDashboard, Server, Settings as SettingsIcon, Upload } from 'lucide-react';
import { useResource } from './hooks';
import { api } from './api';
import Dashboard from './pages/Dashboard';
import UploadPage from './pages/UploadPage';
import Recordings from './pages/Recordings';
import Runs from './pages/Runs';
import RunResult from './pages/RunResult';
import System from './pages/System';
import Settings from './pages/Settings';
import RunCompare from './pages/RunCompare';

const links = [
  ['/', 'Dashboard', LayoutDashboard, true],
  ['/upload', 'Upload', Upload],
  ['/recordings', 'Recordings', Database],
  ['/runs', 'Model runs', FileClock],
  ['/compare', 'Compare runs', GitCompare],
  ['/system', 'System', Server],
  ['/settings', 'Settings', SettingsIcon],
];

function applyPreferences() {
  const theme = localStorage.getItem('seizureai-theme') || 'system';
  const dark = theme === 'dark' || (theme === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.documentElement.dataset.contrast = localStorage.getItem('seizureai-contrast') || 'normal';
  document.documentElement.dataset.motion = localStorage.getItem('seizureai-motion') || 'normal';
}

export default function App() {
  const health = useResource(api.health, []);
  useEffect(() => { applyPreferences(); window.addEventListener('seizureai-preferences', applyPreferences); return () => window.removeEventListener('seizureai-preferences', applyPreferences); }, []);
  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span><Brain /></span><div><strong>SeizureAI</strong><small>VIDEO ANALYSIS</small></div></div>
      <nav>
        <p>Workspace</p>
        {links.map(([to, label, Icon, end]) => <NavLink key={to} to={to} end={end}
          className={({ isActive }) => isActive ? 'active' : ''}>
          <Icon size={18} />{label}
        </NavLink>)}
      </nav>
      <div className="research-note"><Activity size={17} /><div><strong>Research prototype</strong><small>Not a medical device</small></div></div>
    </aside>
    <div className="workspace">
      <header className="topbar">
        <div><strong>Multimodal research platform</strong><small>PostgreSQL catalog · MinIO artifacts · authoritative inference</small></div>
        <span className={`service ${health.error ? 'offline' : ''}`}><i />{health.error ? 'Service offline' : health.loading ? 'Connecting' : 'System ready'}</span>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/recordings" element={<Recordings />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:id" element={<RunResult />} />
          <Route path="/compare" element={<RunCompare />} />
          <Route path="/system" element={<System />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  </div>;
}
