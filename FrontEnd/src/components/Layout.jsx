import {NavLink, Outlet, useNavigate} from 'react-router-dom';
import {
    BarChart3, Brain, ClipboardList, Database, LayoutGrid, LogOut,
    Settings as Cog, Upload,
} from 'lucide-react';

import { useApi } from '../hooks/Useapi';
import { api } from '../api/client';

const NAV = [
    { group: 'Core', items: [
        { to: '/', label: 'Dashboard', icon: LayoutGrid, end: true },
        { to: '/upload', label: 'Upload recording', icon: Upload },
        { to: '/historical', label: 'Past analyses', icon: Database },
    ]},
    { group: 'Analysis', items: [
        { to: '/results', label: 'Results', icon: ClipboardList },
        { to: '/metrics', label: 'Eval metircs', icon: BarChart3}, 
    ]},
    { group: 'System', items: [
        { to: '/settings', label: 'Settings', icon: Cog},
    ]},
];

export default function Layout({ onSignOut }) {
    const navigate = useNavigate();
    const {data: health } = useApi(() => api.health(), []);
    const count = health?.recordings ?? 0;

    return (
    <div className="flex h-full bg-slate-50 text-slate-900">
      <aside className="flex w-60 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-5">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-sky-400 to-blue-600">
            <Brain className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="font-semibold leading-tight text-slate-900">SeizureAI</div>
            <div className="text-[10px] font-medium tracking-widest text-slate-400">VEEG ANALYSIS</div>
          </div>
        </div>
 
        <nav className="flex-1 overflow-y-auto px-3 py-4">
          {NAV.map(({ group, items }) => (
            <div key={group} className="mb-5">
              <div className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                {group}
              </div>
              {items.map(({ to, label, icon: Icon, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  className={({ isActive }) =>
                    `mb-0.5 flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${
                      isActive
                        ? 'bg-blue-50 font-medium text-blue-700'
                        : 'text-slate-600 hover:bg-slate-50'
                    }`
                  }
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
 
        <div className="flex items-center gap-3 border-t border-slate-100 px-4 py-4">
          <div className="grid h-8 w-8 place-items-center rounded-full bg-indigo-500 text-xs font-semibold text-white">
            C
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-medium text-slate-800">Cal</div>
            <div className="text-xs text-slate-400">Student researcher</div>
          </div>
          <button
            type="button"
            onClick={() => { onSignOut(); navigate('/login'); }}
            className="text-slate-400 hover:text-slate-600"
            aria-label="Sign out"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </aside>
 
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-7">
          <div className="text-sm font-medium text-slate-700">SeizureAI</div>
          <span className="flex items-center gap-2 rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
            <span className="text-xs font-medium text-amber-700">
              Video-only · {count} recording{count === 1 ? '' : 's'}
            </span>
          </span>
        </header>
 
        <main className="flex-1 overflow-y-auto p-7">
          <Outlet />
        </main>
      </div>
    </div>
  );
}