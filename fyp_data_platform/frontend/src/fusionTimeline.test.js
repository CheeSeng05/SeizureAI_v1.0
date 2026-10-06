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
