import { useState } from 'react';
import { Card, Notice } from '../components';
import ArchivedItems from '../ArchivedItems';

const read = (key, fallback) => localStorage.getItem(key) || fallback;

export default function Settings() {
  const [theme, setTheme] = useState(() => read('seizureai-theme', 'system'));
  const [contrast, setContrast] = useState(() => read('seizureai-contrast', 'normal'));
  const [motion, setMotion] = useState(() => read('seizureai-motion', 'normal'));
  const save = (key, value, setter) => { localStorage.setItem(key, value); setter(value); window.dispatchEvent(new Event('seizureai-preferences')); };
  return <div className="page narrow">
    <div className="page-title"><div><p>Preferences</p><h1>Settings</h1><span>Appearance and accessibility preferences are saved in this browser.</span></div></div>
    <Card title="Appearance"><div className="setting-group"><label><span>Colour theme</span><select value={theme} onChange={(event) => save('seizureai-theme', event.target.value, setTheme)}><option value="system">Use system setting</option><option value="light">Light</option><option value="dark">Dark</option></select></label><label><span>Contrast</span><select value={contrast} onChange={(event) => save('seizureai-contrast', event.target.value, setContrast)}><option value="normal">Standard</option><option value="high">High contrast</option></select></label></div></Card>
    <Card title="Accessibility"><div className="setting-group"><label><span>Animation</span><select value={motion} onChange={(event) => save('seizureai-motion', event.target.value, setMotion)}><option value="normal">Standard</option><option value="reduced">Reduce motion</option></select></label></div></Card>
    <Notice title="Local preference only"><p>These options do not change model processing, saved runs, other users, or server configuration.</p></Notice>
    <ArchivedItems />
  </div>;
}
