export function elapsedSeconds(run, now = Date.now()) {
  if (!run) return null;
  if (run.status !== 'running') return run.runtime?.timings?.total_sec ?? run.elapsed_sec ?? null;
  const timestamp = run.started_at;
  const started = timestamp ? Date.parse(/[zZ]|[+-]\d\d:\d\d$/.test(timestamp) ? timestamp : `${timestamp}Z`) : NaN;
  return Number.isFinite(started) ? Math.max(0, (now - started) / 1000) : run.elapsed_sec ?? null;
}
