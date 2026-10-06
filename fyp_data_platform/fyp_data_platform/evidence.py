"""Read-only evidence from retained, run-specific inputs on the recording clock."""
import math
from pathlib import Path
import numpy as np
from .storage.keys import safe_entity


def input_path(settings, row, value):
    config = row.config_json or {}
    root = (settings.work_root / safe_entity(config.get('dataset_id', '')) / safe_entity(config.get('recording_id', '')) / safe_entity(row.id) / 'inputs').resolve()
    path = Path(str(value or '')).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('The original input for this run is no longer retained locally. Evidence is unavailable.')
    return path


def evidence_sources(settings, row):
    data = (row.config_json or {}).get('config', {}).get('data', {})
    documents = (row.metrics_json or {}).get('metrics', {})
    manifest = next((doc for key, doc in documents.items() if str(key).endswith('video_segment_manifest.json') and isinstance(doc, dict)), {})
    paths = data.get('video_segments') or []
    videos = []
    if manifest.get('sync_verified'):
        for segment in manifest.get('segments', []):
            matches = [value for value in paths if Path(value).name == segment.get('filename')]
            if len(matches) != 1 or not segment.get('sync_verified'):
                continue
            try:
                input_path(settings, row, matches[0])
            except ValueError:
                continue
            videos.append({**{key: segment.get(key) for key in ('segment_id', 'filename', 'global_start_sec', 'global_end_sec')}, 'input_index': paths.index(matches[0])})
    try:
        path = input_path(settings, row, data.get('eeg_file'))
        eeg = path.suffix.lower() in {'.e', '.edf'}
    except ValueError:
        eeg = False
    return {'videos': videos, 'eeg_available': eeg}


def waveform(path, start, end, channel):
    from seizure_multimodal.eeg.loaders import load_eeg
    if not all(math.isfinite(value) for value in (start, end)) or start < 0 or end <= start or end - start > 30:
        raise ValueError('Choose an EEG interval of at most 30 seconds.')
    with load_eeg(path) as recording:
        meta = recording.metadata
        if not 0 <= channel < len(meta.channel_names):
            raise ValueError('Unknown EEG channel.')
        rate = float(meta.sampling_rates[channel])
        if not math.isfinite(rate) or rate <= 0 or rate > 100000:
            raise ValueError('Unsupported EEG sampling rate.')
        segments = meta.extra.get('timeline_segments') or [{'global_start_sec': 0, 'global_end_sec': meta.duration_sec, 'source_sample_start': 0, 'source_sample_stop': int(meta.duration_sec * rate)}]
        traces = []
        for segment in segments:
            origin = float(segment['global_start_sec'])
            left, right = max(start, origin), min(end, float(segment['global_end_sec']))
            if right <= left:
                continue
            offset = int(segment['source_sample_start'])
            first = max(offset, offset + math.ceil((left - origin) * rate - 1e-7))
            last = min(int(segment['source_sample_stop']), offset + math.ceil((right - origin) * rate - 1e-7))
            if last <= first:
                continue
            values = recording.read_samples(channel_indices=[channel], start_sample=first, stop_sample=last)[0]
            if not np.isfinite(values).all():
                raise ValueError('EEG samples contain non-finite values.')
            # Min/max bins preserve peaks; gaps stay separate rather than being joined.
            stride = max(1, math.ceil(rate * (end - start) / 1000))
            points = []
            for index in range(0, len(values), stride):
                chunk = values[index:index + stride]
                points.append([origin + (first - offset + index) / rate, float(chunk.min()), float(chunk.max())])
            traces.append(points)
        return {'start_sec': start, 'end_sec': end, 'channel': channel, 'channel_names': list(meta.channel_names), 'unit': meta.signal_units[channel] or 'unknown units', 'sampling_rate': rate, 'traces': traces}
