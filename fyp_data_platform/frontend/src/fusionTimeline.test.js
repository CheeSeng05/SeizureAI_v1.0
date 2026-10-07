import test from 'node:test';
import assert from 'node:assert/strict';
import { parseTimelineTime, timelinePage } from './fusionTimeline.js';
const windows = Array.from({ length: 23 }, (_, index) => ({ start_sec: index * 2, end_sec: index * 2 + 4 }));
test('pagination reaches all windows, clamps pages and handles no matches', () => {
  assert.equal(timelinePage(windows, 0).rows.length, 10);
  assert.equal(timelinePage(windows, 1).rows[0].start_sec, 20);
  assert.equal(timelinePage(windows, 2).rows.length, 3);
  assert.equal(timelinePage(windows, 9).current, 2);
  assert.deepEqual([0,1,2].flatMap(page => timelinePage(windows, page).rows), windows);
  assert.equal(timelinePage(windows, 2, { start: 100 }).total, 0);
});
test('time parsing validates minutes and supports seconds and hours', () => {
  for (const time of ['90', '1:30', '0:01:30']) assert.equal(parseTimelineTime(time), 90);
  assert.equal(parseTimelineTime(''), null);
  assert.equal(parseTimelineTime('1:00:00'), 3600);
  for (const time of ['-1', 'abc', '1:60', '1:99:00']) assert.ok(Number.isNaN(parseTimelineTime(time)));
});
test('search includes overlapping windows and excludes touching boundaries', () => {
  assert.deepEqual(timelinePage(windows, 0, { start: 4, end: 6 }).rows, windows.slice(1,3));
  assert.equal(timelinePage(windows, 0, { end: 2 }).rows.length, 1);
});

import { pickerSeconds, windowDecision } from './fusionTimeline.js';
test('decision and time filters compose before pagination', () => {
  const rows = Array.from({ length: 30 }, (_, index) => ({ start_sec: index*2, end_sec: index*2+4, fusion_available: index % 3 !== 0, fused_prediction: index % 3 === 1 ? 1 : 0 }));
  assert.equal(timelinePage(rows, 0, { decision: 'positive' }).total, 10);
  assert.ok(timelinePage(rows, 0, { decision: 'eeg-only' }).rows.every(row => !row.fusion_available));
  assert.equal(timelinePage(rows, 5, { decision: 'negative', start: 20, end: 30 }).current, 0);
  assert.ok(timelinePage(rows, 0, { decision: 'negative', start: 20, end: 30 }).rows.every(row => windowDecision(row) === 'negative' && row.end_sec > 20 && row.start_sec < 30));
});
test('minute and second picker handles empty boundaries and long recordings', () => {
  assert.equal(pickerSeconds('', ''), null);
  assert.equal(pickerSeconds('120', '30'), 7230);
  assert.equal(pickerSeconds('', '0'), 0);
  assert.equal(pickerSeconds('1', ''), 60);
  assert.ok(Number.isNaN(pickerSeconds('-1', '0')));
  assert.ok(Number.isNaN(pickerSeconds('0', '60')));
});

import { videoWindowInterval, selectedWindowAtTime, preciseTime } from './fusionTimeline.js';
test('EEG window maps to video clock using fractional segment offset, not row count', () => {
  const selected = { start_sec: 12, end_sec: 16 };
  assert.deepEqual(videoWindowInterval(selected, { global_start_sec: 5, global_end_sec: 40 }), { start: 12, end: 16, localStart: 7, localEnd: 11 });
  const interval = videoWindowInterval(selected, { global_start_sec: 4.416, global_end_sec: 40 });
  assert.ok(Math.abs(interval.localStart - 7.584) < 1e-9);
  assert.deepEqual(videoWindowInterval({ start_sec: 2, end_sec: 6 }, { global_start_sec: 5, global_end_sec: 40 }), { start: 5, end: 6, localStart: 0, localEnd: 1 });
  assert.equal(videoWindowInterval({ start_sec: 0, end_sec: 4 }, { global_start_sec: 5, global_end_sec: 40 }), null);
});
test('overlapping earlier window is not marked as selected playback', () => {
  const selected = { start_sec: 12, end_sec: 16 };
  assert.equal(selectedWindowAtTime({ start_sec: 10, end_sec: 14 }, selected, 12), false);
  assert.equal(selectedWindowAtTime(selected, selected, 12), true);
  assert.equal(selectedWindowAtTime(selected, selected, 16), false);
  assert.equal(preciseTime(7.584), '0:07.58');
  assert.equal(preciseTime(59.999), '1:00.00');
});
