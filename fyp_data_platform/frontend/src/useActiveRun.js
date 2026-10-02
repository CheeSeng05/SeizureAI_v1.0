import { useSyncExternalStore } from 'react';
import { api } from './api';
import { createActiveRunMonitor } from './activeRun';

const storage = {
  getItem: (key) => window.localStorage.getItem(key),
  setItem: (key, value) => window.localStorage.setItem(key, value),
  removeItem: (key) => window.localStorage.removeItem(key),
};
const monitor = createActiveRunMonitor(api, { storage });

export function useActiveRun() {
  const state = useSyncExternalStore(monitor.subscribe, monitor.getSnapshot);
  return { ...state, startRun: monitor.start, stopRun: monitor.stop };
}
