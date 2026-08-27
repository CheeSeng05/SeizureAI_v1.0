def extract_video(video_path: str, output: str, sample_id: str, fps: float = 5.0, window_seconds: float = 8.0) -> None:
    try:
        import cv2, mediapipe as mp
    except ImportError as e:
        raise SystemExit("Video extraction requires opencv-python and mediapipe. Install requirements.txt") from e
    cap = cv2.VideoCapture(video_path); source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    every = max(1, round(source_fps / fps)); pose_seq, face_seq = [], []
    pose_api = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    face_api = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    frame = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if frame % every: frame += 1; continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pr, fr = pose_api.process(rgb), face_api.process(rgb)
        pose = np.zeros((33, 4), np.float32)
        if pr.pose_landmarks:
            pose = np.asarray([[p.x, p.y, p.z, p.visibility] for p in pr.pose_landmarks.landmark], np.float32)
            hip = (pose[23, :3] + pose[24, :3]) / 2; scale = np.linalg.norm(pose[11, :3] - pose[12, :3]) + 1e-6
            pose[:, :3] = (pose[:, :3] - hip) / scale
        # Stable, privacy-reduced facial geometry: selected landmarks relative to eye distance.
        face = np.zeros(20, np.float32)
        if fr.multi_face_landmarks:
            pts = np.asarray([[p.x, p.y] for p in fr.multi_face_landmarks[0].landmark], np.float32)
            pairs = [(33,133),(362,263),(159,145),(386,374),(61,291),(13,14),(78,308),(1,152),(10,152),(234,454)]
            scale = np.linalg.norm(pts[33]-pts[263]) + 1e-6
            face = np.asarray([(pts[a]-pts[b])/scale for a,b in pairs], np.float32).reshape(-1)
        pose_seq.append(pose); face_seq.append(face); frame += 1
    cap.release(); pose_api.close(); face_api.close()
    length = max(1, round(fps * window_seconds)); out = Path(output); (out/"pose").mkdir(parents=True,exist_ok=True); (out/"face").mkdir(parents=True,exist_ok=True)
    records=[]
    for i in range(0, len(pose_seq)-length+1, max(1,length//2)):
        sid=f"{sample_id}_{i:06d}"; pp=out/"pose"/f"{sid}.npy"; fp=out/"face"/f"{sid}.npy"
        np.save(pp,np.asarray(pose_seq[i:i+length])); np.save(fp,np.asarray(face_seq[i:i+length]))
        records.append(dict(sample_id=sid,patient_id=sample_id,split=patient_split(sample_id),label=0,
            pose_path=str(pp.relative_to(out)),face_path=str(fp.relative_to(out)),eeg_path=""))
    write_manifest(out/"video_manifest.csv",records)
    print(f"Wrote {len(records)} unlabeled windows. Edit labels before training.")