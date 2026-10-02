import test from 'node:test';
import assert from 'node:assert/strict';
import { createActiveRunMonitor } from './activeRun.js';

const flush = () => new Promise((resolve) => setImmediate(resolve));
test('stop keeps controls locked until the server confirms process shutdown', async () => {
  let status = 'running'; let requested = false; let stops = 0;
  const fixture = setup({ job: async () => ({ id: 'j', status, stop_requested: requested }), stopRun: async (type, id) => { assert.equal(type, 'job'); assert.equal(id, 'j'); requested = true; stops++; } }, { type: 'job', id: 'j' });
  const unsubscribe = fixture.monitor.subscribe(() => {}); await flush();
  await fixture.monitor.stop(); await fixture.monitor.stop();
  assert.equal(stops, 1);
  await fixture.tick();
  assert.equal(fixture.monitor.getSnapshot().stopping, true);
  assert.ok(fixture.monitor.getSnapshot().job);
  status = 'stopped'; await fixture.tick();
  assert.equal(fixture.monitor.getSnapshot().job, null);
  assert.equal(fixture.monitor.getSnapshot().error, null);
  assert.match(fixture.monitor.getSnapshot().notice, /Run stopped/);
  unsubscribe();
});
function setup(overrides = {}, saved = null) {
  const values = new Map(saved ? [['seizureai-active-job', JSON.stringify(saved)]] : []);
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value), removeItem: (key) => values.delete(key) };
  const timers = new Map(); let next = 0;
  const api = { catalog: async () => ({ runs: [] }), ...overrides };
  const options = { storage, schedule: (fn) => { timers.set(++next, fn); return next; }, cancel: (id) => timers.delete(id) };
  return { monitor: createActiveRunMonitor(api, options), api, options, timers, values, tick: async () => { const tasks = [...timers.values()]; timers.clear(); for (const fn of tasks) await fn(); } };
}

test('recovers an already-running server run after navigation and browser reload', async () => {
  let completed = 29;
  const fixture = setup({
    catalog: async () => ({ runs: [{ id: 'existing', status: 'running' }] }),
    run: async () => ({ id: 'existing', status: 'running', started_at: new Date(Date.now() - 60000).toISOString(), progress: { completed_units: completed, total_units: 749, percent: completed / 749 * 100 } }),
  });
  let unsubscribe = fixture.monitor.subscribe(() => {}); await flush();
  assert.equal(fixture.monitor.getSnapshot().job.progress.completed_units, 29);
  assert.ok(fixture.monitor.getSnapshot().job.progress.elapsed_sec >= 59);
  unsubscribe(); assert.equal(fixture.timers.size, 0);
  completed = 30;
  unsubscribe = fixture.monitor.subscribe(() => {}); await flush();
  assert.equal(fixture.monitor.getSnapshot().job.progress.completed_units, 30);
  unsubscribe();
  const reloaded = createActiveRunMonitor(fixture.api, fixture.options);
  unsubscribe = reloaded.subscribe(() => {}); await flush();
  assert.equal(reloaded.getSnapshot().job.id, 'existing');
  unsubscribe();
});

test('retains last progress on a network error and clears it when the run finishes', async () => {
  let status = 'running'; let offline = false;
  const fixture = setup({ job: async () => { if (offline) throw new Error('offline'); return { id: 'job', status, progress: { completed_units: 8 } }; } }, { type: 'job', id: 'job' });
  const unsubscribe = fixture.monitor.subscribe(() => {}); await flush();
  offline = true; await fixture.tick();
  assert.equal(fixture.monitor.getSnapshot().job.progress.completed_units, 8);
  assert.equal(fixture.monitor.getSnapshot().error.message, 'offline');
  offline = false; status = 'completed'; await fixture.tick();
  assert.equal(fixture.monitor.getSnapshot().job, null);
  assert.equal(fixture.monitor.getSnapshot().completedVersion, 1);
  assert.equal(fixture.values.size, 0);
  unsubscribe();
});

test('expired in-memory job recovers through the durable run catalog', async () => {
  const fixture = setup({ job: async () => { throw Object.assign(new Error('missing'), { status: 404 }); }, catalog: async () => ({ runs: [{ id: 'run', status: 'running' }] }), run: async () => ({ id: 'run', status: 'running', progress: { completed_units: 11 } }) }, { type: 'job', id: 'old' });
  const unsubscribe = fixture.monitor.subscribe(() => {}); await flush();
  assert.equal(fixture.monitor.getSnapshot().job.id, 'run');
  unsubscribe();
});

test('a start response arriving after navigation is retained and not started twice', async () => {
  let resolveStart; let starts = 0;
  const fixture = setup({ startRun: () => { starts++; return new Promise((resolve) => { resolveStart = resolve; }); }, job: async () => ({ id: 'new', status: 'running', progress: { percent: 2 } }) });
  let unsubscribe = fixture.monitor.subscribe(() => {}); await flush();
  const row = { dataset_id: 'd', recording_source_id: 'r' };
  const starting = fixture.monitor.start(row, 'video');
  await fixture.monitor.start(row, 'video');
  unsubscribe();
  resolveStart({ id: 'new', status: 'running' }); await starting;
  unsubscribe = fixture.monitor.subscribe(() => {}); await flush();
  assert.equal(starts, 1);
  assert.equal(fixture.monitor.getSnapshot().job.progress.percent, 2);
  unsubscribe();
});

test('an old pending request cannot replace the remounted page progress', async () => {
  let resolveOld; let calls = 0;
  const fixture = setup({ job: () => ++calls === 1 ? new Promise((resolve) => { resolveOld = resolve; }) : Promise.resolve({ id: 'j', status: 'running', progress: { completed_units: 40 } }) }, { type: 'job', id: 'j' });
  const first = fixture.monitor.subscribe(() => {}); first();
  const second = fixture.monitor.subscribe(() => {}); await flush();
  resolveOld({ id: 'j', status: 'completed' }); await flush();
  assert.equal(fixture.monitor.getSnapshot().job.progress.completed_units, 40);
  assert.equal(fixture.timers.size, 1);
  second();
});
