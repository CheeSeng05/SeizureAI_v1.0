"""Local, disposable H.264 viewing copies; never used as model inputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

from .storage.keys import safe_entity


class VideoPreviews:
    def __init__(self, work_root, storage):
        self.work_root = Path(work_root)
        self.storage = storage
        self.lock = threading.RLock()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-preview")

    @property
    def busy(self):
        return any(state['status'] == 'preparing' for state in self.jobs.values())

    def path(self, dataset, recording, key, digest):
        identity = hashlib.sha256(f'v1:{key}:{digest}'.encode()).hexdigest()
        return self.work_root / safe_entity(dataset) / safe_entity(recording) / 'browser-preview' / f'{identity}.mp4'

    def ensure(self, destination, key, source_path=None):
        destination = Path(destination)
        with self.lock:
            if destination.is_file() and destination.stat().st_size:
                return {'status': 'ready'}
            state = self.jobs.get(destination)
            if state:
                return dict(state)
            if not shutil.which('ffmpeg'):
                return {'status': 'failed', 'error': 'Video preview requires FFmpeg. Rebuild the updated app image.'}
            self.jobs[destination] = {'status': 'preparing'}
            self.executor.submit(self._convert, destination, key, source_path)
            return {'status': 'preparing'}

    def _convert(self, destination, key, source_path=None):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
                original = Path(directory) / ('source' + Path(key).suffix)
                output = Path(directory) / 'preview.mp4'
                if source_path is not None:
                    shutil.copy2(source_path, original)
                else:
                    self.storage.download_file(key, original)
                subprocess.run([
                    'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-threads', '2', '-i', str(original), '-map', '0:v:0', '-an',
                    '-map_metadata', '-1', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-filter_threads', '1', '-c:v', 'libx264', '-threads', '2',
                    '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                    '-fps_mode', 'passthrough', '-movflags', '+faststart', str(output),
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600)
                if not output.is_file() or not output.stat().st_size:
                    raise ValueError('No preview frames produced')
                output.replace(destination)
            state = {'status': 'ready'}
        except Exception:
            state = {'status': 'failed', 'error': 'Could not prepare this video for browser playback. The original is still available to download.'}
        with self.lock:
            self.jobs[destination] = state
