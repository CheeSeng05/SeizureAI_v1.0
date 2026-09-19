import json
from types import SimpleNamespace
from seizure_multimodal.video.pose_export import PoseExport

def test_pose_export_preserves_pixels_confidence_and_missing_frames(tmp_path):
    path = tmp_path/'pose.json'
    export = PoseExport(path, SimpleNamespace(width=1920,height=1080,fps=25))
    export.add(0, [(10.1234,20.1234,0.9),(-1,-1,0),(float('nan'),20,0.7)])
    export.add(2, [(50,60,0.2)])
    export.close()
    data = json.loads(path.read_text())
    assert data['width'] == 1920
    assert data['frames'] == [[0, [[10.12,20.12,0.9],None,None]], [2, [[50.0,60.0,0.2]]]]
    assert len(data['edges']) == 17

def test_pose_export_optional():
    export = PoseExport(None, None)
    export.add(0, [])
    export.close()

def test_raw_pipeline_exports_processed_frames_only(tmp_path, monkeypatch):
    import sys
    import numpy as np
    from seizure_multimodal.video import raw_pipeline as raw
    metadata = SimpleNamespace(width=1920,height=1080,fps=25,frame_count=32,duration_sec=1.28,as_dict=lambda: {})
    monkeypatch.setattr(raw, 'inspect_mp4', lambda _: metadata)
    class Capture:
        index = 0
        def isOpened(self): return True
        def read(self):
            self.index += 1
            return (True, np.zeros((2,2,3))) if self.index <= 32 else (False, None)
        def release(self): pass
    monkeypatch.setitem(sys.modules,'cv2',SimpleNamespace(VideoCapture=lambda _:Capture()))
    points = np.array([[100,200,0.8]]*18)
    monkeypatch.setattr(raw,'LightweightOpenPose',lambda **_:SimpleNamespace(is_inference_only=True,predict_many=lambda frames:[(points,True) for _ in frames]))
    class Adapter:
        is_inference_only = True
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
    monkeypatch.setattr(raw,'VSViGAdapter',Adapter)
    monkeypatch.setattr(raw,'extract_vsvig_patches',lambda *_:np.zeros((1,)))
    monkeypatch.setattr(raw,'_flush_clips',lambda _, pending, **kwargs:([{'frame':p[0]} for p in pending],[np.zeros(384) for p in pending]))
    output = tmp_path/'pose.json'
    rows, _, details = raw.predict_video_file(video_path='sample.mp4',label_path=None,vsvig_source='model.py',vsvig_checkpoint='model.pth',dynamic_partition='order.pt',pose_source='pose',pose_checkpoint='pose.pth',device='cpu',pose_output_path=output)
    frames = json.loads(output.read_text())['frames']
    assert len(rows) == 1
    assert len(frames) == 30
    assert frames[0][0] == 0 and frames[-1][0] == 29
    assert details['discarded_trailing_frames'] == 2
