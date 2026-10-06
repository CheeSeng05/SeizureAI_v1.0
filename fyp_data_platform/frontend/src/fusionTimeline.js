export function parseTimelineTime(value) {
  const text = value.trim();
  if (!text) return null;
  if (!/^\d+(?:\.\d+)?$|^\d+:\d{1,2}(?:\.\d+)?$|^\d+:\d{1,2}:\d{1,2}(?:\.\d+)?$/.test(text)) return NaN;
  const parts = text.split(':').map(Number);
  if (parts.length > 1 && parts.slice(1).some(part => part >= 60)) return NaN;
  return parts.reduce((seconds, part) => seconds * 60 + part, 0);
}

export function timelinePage(windows, page, range = {}, size = 10) {
  const filtered = windows.filter(window =>
    (range.start == null || Number(window.end_sec) > range.start) &&
    (range.end == null || Number(window.start_sec) < range.end));
  const pages = Math.max(1, Math.ceil(filtered.length / size));
  const current = Math.min(Math.max(0, page), pages - 1);
  return { rows: filtered.slice(current * size, (current + 1) * size), total: filtered.length, pages, current };
}
