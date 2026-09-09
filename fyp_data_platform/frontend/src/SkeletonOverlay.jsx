import { useEffect, useRef, useState } from 'react';

export default function SkeletonOverlay({ videoRef, url }) {
  const canvasRef = useRef(null);
  const [enabled, setEnabled] = useState(false);
  const [pose, setPose] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { setPose(null); setError(''); setEnabled(false); }, [url]);
  useEffect(() => {
    if (!enabled || !url || pose) return;
    const controller = new AbortController();
    fetch(url, { signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error('Could not load skeleton data.');
      const data = await response.json();
      if (data.version !== 1 || !data.fps || !data.width || !data.height || !Array.isArray(data.frames)) throw new Error('Unsupported skeleton data.');
      setPose(data); setError('');
    }).catch((err) => { if (err.name !== 'AbortError') setError(err.message); });
    return () => controller.abort();
  }, [enabled, url, pose]);
  useEffect(() => {
    if (!enabled || !pose) return;
    const canvas = canvasRef.current; const video = videoRef.current;
    if (!canvas || !video) return;
    let frameCallback; let animation; let stopped = false;
    const draw = (time = video.currentTime) => {
      const rect = video.getBoundingClientRect();
      canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
      const ratio = window.devicePixelRatio || 1;
      const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, width, height);
      // Only draw the exact processed frame. Missing poses and trailing frames stay empty.
      const frameIndex = Math.floor(time * pose.fps + 0.001);
      let low = 0, high = pose.frames.length - 1;
      while (low <= high) { const mid = (low + high) >> 1; if (pose.frames[mid][0] < frameIndex) low = mid + 1; else high = mid - 1; }
      const frame = pose.frames[low]; if (!frame || frame[0] !== frameIndex) return;
      const scale = Math.min(width / pose.width, height / pose.height);
      const x0 = (width - pose.width * scale) / 2, y0 = (height - pose.height * scale) / 2;
      const valid = (p) => p && p.length === 3 && p.every(Number.isFinite) && p[0] >= 0 && p[1] >= 0 && p[0] <= pose.width && p[1] <= pose.height && p[2] >= 0.1;
      const color = (score) => score >= 0.6 ? '#34d399' : score >= 0.3 ? '#fbbf24' : '#fb7185';
      ctx.lineWidth = 3 * ratio; ctx.lineCap = 'round';
      for (const [a,b] of pose.edges) { const p = frame[1][a], q = frame[1][b]; if (!valid(p) || !valid(q)) continue; ctx.strokeStyle = color(Math.min(p[2], q[2])); ctx.beginPath(); ctx.moveTo(x0+p[0]*scale,y0+p[1]*scale); ctx.lineTo(x0+q[0]*scale,y0+q[1]*scale); ctx.stroke(); }
      for (const p of frame[1]) { if (!valid(p)) continue; ctx.fillStyle=color(p[2]); ctx.beginPath(); ctx.arc(x0+p[0]*scale,y0+p[1]*scale,4*ratio,0,2*Math.PI); ctx.fill(); }
    };
    const tick = (_, metadata) => { if (stopped) return; draw(metadata?.mediaTime ?? video.currentTime); if (video.requestVideoFrameCallback) frameCallback=video.requestVideoFrameCallback(tick); else animation=requestAnimationFrame(tick); };
    const redraw = () => draw(); const resize = new ResizeObserver(redraw); resize.observe(video);
    video.addEventListener('seeked', redraw); video.addEventListener('loadedmetadata', redraw); tick();
    return () => { stopped=true; if (frameCallback !== undefined) video.cancelVideoFrameCallback(frameCallback); cancelAnimationFrame(animation); resize.disconnect(); video.removeEventListener('seeked', redraw); video.removeEventListener('loadedmetadata', redraw); };
  }, [enabled, pose, videoRef]);
  return <>
    {enabled && pose && <canvas className="skeleton-canvas" ref={canvasRef} aria-hidden="true" />}
    <div className="skeleton-controls"><label><input type="checkbox" checked={enabled} disabled={!url} onChange={(event) => setEnabled(event.target.checked)} /> Show skeleton</label>
      {!url ? <small>No saved pose data. Rerun this video to enable the overlay.</small> : <small>Pose confidence score: green ≥0.60, yellow ≥0.30, pink ≥0.10. Missing points are hidden. These colours do not explain seizure predictions.</small>}
      <button className="button small" type="button" onClick={async (event) => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await event.currentTarget.closest('.skeleton-stage').requestFullscreen(); } catch { setError('Fullscreen is unavailable in this browser.'); } }}>Fullscreen with overlay</button>
      {enabled && !pose && !error && <small>Loading skeleton…</small>}{error && <small role="alert">{error}</small>}
    </div>
  </>;
}
