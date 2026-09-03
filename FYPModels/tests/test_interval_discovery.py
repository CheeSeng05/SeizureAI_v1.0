from unittest.mock import patch

from seizure_multimodal.checkpoints.provenance import sha256_file
from seizure_multimodal.data_inspection import MP4VideoMetadata
from seizure_multimodal.discovery import discover_sources, inspection_bundle
from seizure_multimodal.video.raw_pipeline import load_video_annotation, reference_label_for_interval


def test_csv_inspection_and_hash_verified_recording_identity(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'metadata is mocked, not inference')
    labels = tmp_path / 'labels.csv'
    labels.write_text('recording_id,patient_id,start_sec,end_sec,label\nstitched-test-001,p1,0,5,0\nstitched-test-001,p1,5,10,1\n', encoding='utf-8')
    config = {'data': {'video': str(video), 'video_labels': str(labels), 'video_provenance': {'expected_sha256': sha256_file(video), 'patient_id': 'P1', 'recording_id': 'stitched-test-001'}}}
    metadata = MP4VideoMetadata(str(video), 10.0, 30.0, 300, 640, 480, 'avc1', False)
    with patch('seizure_multimodal.discovery.inspect_mp4', return_value=metadata):
        inspected = inspection_bundle(tmp_path, config)
        source = discover_sources(tmp_path, config)[0]
    assert inspected['video_labels']['data_rows'] == 2
    assert source.recording_id == 'stitched-test-001'
    annotation = load_video_annotation(labels, 10, patient_id=source.patient_id, recording_id=source.recording_id)
    assert annotation.alignment_status == 'aligned_to_video_time'
    assert len(annotation.intervals) == 2
    assert reference_label_for_interval(annotation, 0, 5) == 0
    assert reference_label_for_interval(annotation, 5, 10) == 1


def test_unverified_provenance_cannot_select_recording(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'video')
    config = {'data': {'video': str(video), 'video_labels': None, 'video_provenance': {'expected_sha256': 'wrong', 'recording_id': 'untrusted'}}}
    metadata = MP4VideoMetadata(str(video), 10.0, 30.0, 300, 640, 480, 'avc1', False)
    with patch('seizure_multimodal.discovery.inspect_mp4', return_value=metadata):
        assert discover_sources(tmp_path, config)[0].recording_id == 'video'
