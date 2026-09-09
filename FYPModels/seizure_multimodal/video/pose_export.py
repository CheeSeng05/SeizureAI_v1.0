"""Stream original-image OpenPose coordinates without retaining a whole video in RAM."""
import json
import math
from pathlib import Path

EDGES = [(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
         (1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)]

class PoseExport:
    def __init__(self, path, metadata):
        self.stream = None
        self.count = 0
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = path.open('w', encoding='utf-8')
            header = {"version": 1, "width": metadata.width, "height": metadata.height,
                      "fps": metadata.fps, "edges": EDGES,
                      "policy": "Largest detected person per frame. Colours show pose confidence, not seizure importance."}
            self.stream.write(json.dumps(header)[:-1] + ',"frames":[')

    def add(self, frame_index, points):
        if self.stream is None:
            return
        clean = []
        for x, y, score in points:
            values = [float(x), float(y), float(score)]
            clean.append([round(values[0], 2), round(values[1], 2), round(max(0, min(1, values[2])), 3)]
                         if all(math.isfinite(v) for v in values) and x >= 0 and y >= 0 and score > 0
                         else None)
        if self.count:
            self.stream.write(',')
        self.stream.write(json.dumps([frame_index, clean], separators=(',', ':'), allow_nan=False))
        self.count += 1

    def close(self):
        if self.stream:
            self.stream.write(']}')
            self.stream.close()
            self.stream = None
