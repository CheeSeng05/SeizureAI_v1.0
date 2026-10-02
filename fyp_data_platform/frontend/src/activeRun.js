import { elapsedSeconds } from './runTiming.js';

const storageKey = 'seizureai-active-job';

// This store lives outside the routed page. The server remains authoritative.
export function createActiveRunMonitor(api, { storage, schedule = setTimeout, cancel = clearTimeout } = {}) {
  let reference = null;
  try {
    const saved = JSON.parse(storage?.getItem(storageKey) || 'null');
    if (saved && ['job', 'run'].includes(saved.type) && typeof saved.id === 'string') reference = saved;
  } catch { /* Browser storage is optional. Server discovery still works. */ }
  let state = { job: null, error: null, checking: true, completedVersion: 0, stopping: false, notice: null };
  let requestingStop = false;
  let timer; let generation = 0; let starting = false; let failure = null;
  const listeners = new Set();
  const update = (changes) => { state = { ...state, ...changes }; listeners.forEach((listener) => listener()); };
  const remember = (value) => {
    reference = value;
    try { if (value) storage?.setItem(storageKey, JSON.stringify(value)); else storage?.removeItem(storageKey); } catch { /* optional */ }
  };
  const poll = async () => {
    if (starting || requestingStop) return;
    const currentGeneration = generation;
    try {
      let selected = reference;
      let current;
      if (selected) {
        try { current = await api[selected.type === 'job' ? 'job' : 'run'](selected.id); }
        catch (error) { if (error.status !== 404) throw error; selected = null; }
      }
      if (!selected) {
        const catalog = await api.catalog();
        const running = catalog.runs.find((run) => run.status === 'running');
        if (running) {
          selected = { type: 'run', id: running.id };
          current = await api.run(running.id);
        }
      }
      if (currentGeneration !== generation) return;
      if (current?.status === 'running') {
        failure = null;
        remember(selected);
        const progress = { ...current.progress };
        if (selected.type === 'run') progress.elapsed_sec = elapsedSeconds(current);
        update({ job: { ...current, progress }, error: null, checking: false, stopping: Boolean(current.stop_requested) });
      } else {
        const hadRun = Boolean(reference || state.job);
        remember(null);
        if (current?.status === 'failed') failure = new Error(current.error || 'The model run failed. Open Model runs for details.');
        update({ job: null, checking: false, stopping: false, notice: current?.status === 'stopped' ? 'Run stopped. You can start a new run; uploaded files are unchanged.' : state.notice, error: failure, completedVersion: state.completedVersion + Number(hadRun) });
      }
    } catch (error) {
      if (currentGeneration === generation) update({ error, checking: false });
    } finally {
      if (currentGeneration === generation && listeners.size) timer = schedule(poll, 3000);
    }
  };
  return {
    getSnapshot: () => state,
    subscribe(listener) {
      listeners.add(listener);
      if (listeners.size === 1) poll();
      return () => { listeners.delete(listener); if (!listeners.size) { cancel(timer); generation++; } };
    },
    async stop() {
      if (!reference || !state.job || state.stopping || requestingStop) return;
      generation++; cancel(timer); requestingStop = true;
      update({ stopping: true, error: null });
      try { await api.stopRun(reference.type, reference.id); }
      catch (error) { update({ stopping: false, error }); }
      finally { requestingStop = false; if (listeners.size) timer = schedule(poll, 0); }
    },
    async start(row, mode, requireLabels = false) {
      if (state.job || starting || state.checking) return;
      generation++; cancel(timer); starting = true; failure = null;
      update({ job: { status: 'starting', recording_id: row.recording_source_id, progress: { stage_label: 'Starting model run', percent: 0 } }, error: null, notice: null, stopping: false });
      try {
        const started = await api.startRun(row.dataset_id, row.recording_source_id, mode, requireLabels);
        remember({ type: 'job', id: started.id });
        update({ job: { ...started, recording_id: row.recording_source_id }, error: null });
      } catch (error) {
        failure = error;
        update({ job: null, error });
      } finally {
        starting = false;
        if (listeners.size) timer = schedule(poll, 0);
      }
    },
  };
}
