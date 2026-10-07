export function parseTimelineTime(value) {
  const text = value.trim();
  if (!text) return null;
  if (!/^\d+(?:\.\d+)?$|^\d+:\d{1,2}(?:\.\d+)?$|^\d+:\d{1,2}:\d{1,2}(?:\.\d+)?$/.test(text)) return NaN;
  const parts = text.split(':').map(Number);
  if (parts.length > 1 && parts.slice(1).some(part => part >= 60)) return NaN;
  return parts.reduce((seconds, part) => seconds * 60 + part, 0);
}

export function windowDecision(window) {
  return !window.fusion_available ? 'eeg-only' : Number(window.fused_prediction) === 1 ? 'positive' : 'negative';
}

export function pickerSeconds(minutes, seconds) {
  if (minutes === '' && seconds === '') return null;
  const m = Number(minutes || 0); const s = Number(seconds || 0);
  return Number.isSafeInteger(m) && m >= 0 && Number.isInteger(s) && s >= 0 && s < 60 ? m * 60 + s : NaN;
}

export function timelinePage(windows, page, range = {}, size = 10) {
  const filtered = windows.filter(window =>
    (range.start == null || Number(window.end_sec) > range.start) &&
    (range.end == null || Number(window.start_sec) < range.end) &&
    (!range.decision || range.decision === 'all' || windowDecision(window) === range.decision));
  const pages = Math.max(1, Math.ceil(filtered.length / size));
  const current = Math.min(Math.max(0, page), pages - 1);
  return { rows: filtered.slice(current * size, (current + 1) * size), total: filtered.length, pages, current };
}

export function videoWindowInterval(window, segment) {
  const start = Math.max(Number(window.start_sec), Number(segment.global_start_sec));
  const end = Math.min(Number(window.end_sec), Number(segment.global_end_sec));
  if (![start, end, Number(segment.global_start_sec)].every(Number.isFinite) || end <= start) return null;
  return { start, end, localStart: start - Number(segment.global_start_sec), localEnd: end - Number(segment.global_start_sec) };
}

export function selectedWindowAtTime(window, selected, time) {
  return window === selected && time != null && time >= window.start_sec && time < window.end_sec;
}

export function preciseTime(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const ticks = Math.round(Math.max(0, seconds) * 100);
  return `${Math.floor(ticks / 6000)}:${String(Math.floor(ticks / 100) % 60).padStart(2, '0')}.${String(ticks % 100).padStart(2, '0')}`;
}
