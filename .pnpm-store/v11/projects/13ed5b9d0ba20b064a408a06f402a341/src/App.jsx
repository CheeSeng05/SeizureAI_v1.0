import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { Activity, Brain, Database, FileClock, LayoutDashboard, Server, Upload } from 'lucide-react';
import { useResource } from './hooks';
import { api } from './api';
import Dashboard from './pages/Dashboard';
import UploadPage from './pages/UploadPage';
import Recordings from './pages/Recordings';
import Runs from './pages/Runs';
import RunResult from './pages/RunResult';
import System from './pages/System';

const links = [
  ['/', 'Dashboard', LayoutDashboard, true],
  ['/upload', 'Upload', Upload],
  ['/recordings', 'Recordings', Database],
  ['/runs', 'Model runs', FileClock],
  ['/system', 'System', Server],
];

export default function App() {
  const health = useResource(api.health, []);
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
          <Route path="/system" element={<System />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  </div>;
}
