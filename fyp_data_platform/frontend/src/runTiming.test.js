import test from 'node:test';
import assert from 'node:assert/strict';
import { elapsedSeconds } from './runTiming.js';
test('running duration advances from UTC start, including naive database dates', () => {
  for (const started_at of ['2026-10-01T06:00:00', '2026-10-01T06:00:00+00:00']) {
    assert.equal(elapsedSeconds({ status: 'running', started_at }, Date.parse('2026-10-01T06:01:05Z')), 65);
  }
});
test('completed and failed runs never keep ticking', () => {
  assert.equal(elapsedSeconds({ status: 'completed', runtime: { timings: { total_sec: 17 } } }), 17);
  assert.equal(elapsedSeconds({ status: 'failed', elapsed_sec: 12 }), 12);
  assert.equal(elapsedSeconds({ status: 'completed' }), null);
});
