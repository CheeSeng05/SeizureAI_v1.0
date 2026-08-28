import { useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Historical from './pages/Historical';
import Login from './pages/Login';
import Metrics from './pages/Metrics';
import Results from './pages/Results';
import Settings from './pages/Settings';
import Upload from './pages/Upload';

export default function App() {
  // Placeholder gate so the demo has a front door. Real deployment needs
  // server-side auth — this only hides the UI, not the API.
  const [authed, setAuthed] = useState(false);

  if (!authed) {
    return (
      <Routes>
        <Route path="*" element={<Login onSignIn={() => setAuthed(true)} />} />
      </Routes>
    );
  }

  return (
    <Routes>
      <Route element={<Layout onSignOut={() => setAuthed(false)} />}>
        <Route index element={<Dashboard />} />
        <Route path="upload" element={<Upload />} />
        <Route path="historical" element={<Historical />} />
        <Route path="results" element={<Results />} />
        <Route path="results/:id" element={<Results />} />
        <Route path="metrics" element={<Metrics />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}