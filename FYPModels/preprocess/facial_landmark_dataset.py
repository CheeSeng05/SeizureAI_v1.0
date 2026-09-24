import json
import numpy as np
import pandas as pd


def aperture(points, vertical_pairs, corners):
    width = np.linalg.norm(points[corners[0]] - points[corners[1]])
    if width < 1e-6:
        return np.nan

    gaps = [
        np.linalg.norm(points[a] - points[b])
        for a, b in vertical_pairs
    ]
    return float(np.mean(gaps) / width)


# Zero-based indices for the 68-point landmark layout.
regions = {
    "eye_36_41": ([(37, 41), (38, 40)], (36, 39)),
    "eye_42_47": ([(43, 47), (44, 46)], (42, 45)),
    "mouth":     ([(61, 67), (62, 66), (63, 65)], (48, 54)),
}

rows = []
path = "FYPModels/results/facemap_video_S1/facemap.jsonl"

with open(path) as file:
    for frame_id, line in enumerate(file):
        frame = json.loads(line)
        features = frame["features"]

        points = (
            None if features is None
            else np.asarray(features["landmarks_crop"], dtype=float)
        )

        # Numerical validity only: this does not verify landmark placement.
        numeric_ok = (
            points is not None
            and points.shape == (68, 2)
            and np.isfinite(points).all()
        )

        for region, (pairs, corners) in regions.items():
            rows.append({
                "frame_id": frame_id,
                "time_sec": frame["time_sec"],
                "region": region,
                "aperture": (
                    aperture(points, pairs, corners)
                    if numeric_ok else np.nan
                ),
            })

measurements = pd.DataFrame(rows)

# Missing observations remain NaN; differences across them stay unknown.
groups = measurements.groupby("region", sort=False)
dt = groups["time_sec"].diff()
measurements["velocity"] = (
    groups["aperture"].diff() / dt.where(dt > 0)
)

print(measurements.dropna().head())