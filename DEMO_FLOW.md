# SeizureAI: three-minute live demo

Prepared from the current repository on 12 October 2026. This is a demo guide, not a slide deck. The frontend, API orchestration, storage/catalog code, and main video, EEG, and synchronization paths were reviewed. The local application did not respond at `http://localhost:8080`, so stored run availability and browser behavior still need a rehearsal.

## The story to tell

**Register a recording → run a model → inspect possible events → review synchronized evidence → compare and export.**

The main message: “SeizureAI brings video and EEG research workflows into one app, so researchers can review predictions alongside the original evidence and retain traceable results.”

Give most of the time to result inspection and multimodal evidence. Those show more value than visiting every sidebar page.

## What the app actually does

- **React interface:** Dashboard, Upload, Recordings, Model runs, individual results, Compare runs, System, and Settings.
- **Flask backend:** validates inputs, registers their identities, starts a background job, exposes progress/cancellation, and serves results and evidence.
- **PostgreSQL catalog:** links datasets, patients, recordings, files, and model runs. Patient age/notes can be edited from a result page.
- **MinIO storage in the Docker setup:** retains inputs and output artifacts with SHA-256 checksums. The code also supports other configured storage backends.
- **Model execution:** the platform stages stored inputs into a separate run directory, generates configuration, invokes the `FYPModels` pipeline, and saves its outputs and runtime/device information. Uploading alone does not start inference. Only one inference job can run at a time.
- **Video:** lightweight OpenPose extracts keypoints; the frozen VSViG model processes 30-frame clips and returns probabilities. Completed single-video runs can show playback, saved skeletons, event jumps, labels, and a synchronized probability timeline.
- **EEG:** supports benchmark CSV (`X1`–`X178` and `y`) and raw EDF/Nicolet `.e` recordings. Raw signals are mapped to the required 18-channel bipolar montage, resampled to 256 Hz as needed, filtered/normalized, and processed in four-second windows with a two-second stride by the pretrained EEG model.
- **Real fusion:** requires Nicolet `.e` synchronization metadata and matching video segments. Video predictions are aligned by overlap onto the EEG clock. Default fusion is `0.80 × EEG probability + 0.20 × video probability`. Insufficient video coverage leaves a window EEG-only. These weights are an engineering baseline.
- **Interactive review:** the video analysis threshold changes events and available label-backed metrics from saved probabilities without rerunning the model. AUROC remains unchanged. The fusion result has a window table and tabs for corresponding video playback and original EEG waveforms.
- **Reporting:** run PDFs, stored artifact downloads, and a comparison PDF. Compare runs currently compares saved results; it does not have independent threshold sliders, despite the README describing them.

## Prepare before presenting

1. Start the app using your usual CPU or GPU Compose command. Confirm the dashboard loads and reports System ready.
2. Have a completed **single-video evaluation** with registered ground truth, a playable preview, saved pose data, and an event you can jump to. A prediction-only run works for playback, but cannot demonstrate label-backed metrics.
3. Have a completed **verified fusion run** whose original `.e` file and video inputs are retained. Choose a window with video coverage. Check that both Video playback and EEG waveform load.
4. Have two completed comparable runs for Compare runs, preferably the same recording, model, and label mode. Use CPU/CUDA runs only if you actually have that pair; configuration should also match for a meaningful runtime comparison.
5. Choose a registered recording for the brief live inference launch. Ensure no other job is active. Rehearse the launch; results shown afterward will come from a previously completed run.
6. Open separate browser tabs: Dashboard; Upload; Recordings; completed video result; completed fusion result; Compare runs. Keep the compare pair selected in its own tab because navigating away resets its selection.
7. In Upload, prepare a video selection and catalog fields. You can show the ground-truth switch and the Video + EEG option without transferring a large file live. If an actual upload is required, use a small, valid, non-sensitive demo file and only include it if registration consistently finishes within this slot.
8. Warm up video previews and both fusion evidence tabs before the talk. AVI/MKV/MOV conversion may take time. Pause playback, restore the video threshold to 0.50, and leave useful scroll positions ready.
9. Use de-identified material suitable for the audience. Enlarge the browser enough for timestamps and probabilities to be readable. Rehearse PDF export once.

Do not budget for inference to finish inside the demo. Make the switch to a completed run explicit.

## Exact 180-second flow

| Time | Screen | Action | Audience takeaway |
|---|---|---|---|
| 0:00–0:15 | Dashboard | Point to recording/run counts and the workspace. | One place to manage the research workflow. |
| 0:15–0:40 | Upload | Show Video, EEG, Video + EEG; briefly reveal ground-truth fields in Video mode; point to catalog identity and Store and register. | Supports both modalities, identities, and optional reference labels. |
| 0:40–1:00 | Recordings | Find the prepared recording; click Evaluate video for labelled evaluation, or Run Video Model for prediction. Point to Current model run and Stop model run. | Actual inference starts from stored inputs, with progress and cancellation. |
| 1:00–1:55 | Completed video result | Jump to an event, play briefly, enable Show skeleton, point to probability/playhead, change threshold, and show one or two metrics. | Predictions can be inspected against the original video and reference timing. |
| 1:55–2:35 | Completed fusion result | Show EEG/video/fusion probabilities and coverage; select a prepared window; Play selected window; switch to EEG waveform. | Both modalities can be inspected for the same interval on the EEG clock. |
| 2:35–2:50 | Compare runs | Show the prepared pair, A − B values, and runtime or one available metric. | Saved experiments can be compared with context and warnings. |
| 2:50–3:00 | Completed video result | Return to the top and click Export report. Finish speaking as the PDF downloads. | Findings can be shared in a research report. |

### 0:00–0:15 — introduce the purpose

Say:

> “SeizureAI is a research platform for analysing video and EEG recordings. This dashboard brings stored recordings and model runs into one workspace. I’ll show registration, inference, and how the results can be reviewed.”

Point once to the counts; do not explain the database architecture during this slot.

### 0:15–0:40 — show registration

Say:

> “Users can register video, EEG, or synchronized video and EEG. Dataset, patient, and recording IDs keep the inputs linked. For video evaluation, reference labels can be attached here. Fusion requires synchronization metadata from the Nicolet recording, so unrelated files are not treated as a valid pair.”

Show the optional ground-truth fields in Video mode, then the Video + EEG option and its synchronization notice. If you only show the form, describe how registration works; do not claim you just uploaded the prepared recording.

### 0:40–1:00 — launch inference

Say:

> “From Recordings, I can start the appropriate model on stored inputs. The app reports processing stages, progress, and elapsed time, and provides a stop control. Processing takes longer than this demo, so I’ll now open a previously completed run.”

Click **Evaluate video** if you want the new job to use the registered labels. **Run Video Model** starts prediction without loading those labels. Do not wait for a specific percentage or completion.

If starting a new job would disrupt your setup, point to the actions and use a run started before the talk to demonstrate progress; describe it accurately.

### 1:00–1:55 — show the strongest video interaction

Suggested action order:

1. Open the prepared video result and scroll to **Video and seizure timeline**.
2. Click the rehearsed event; playback starts three seconds before it. Let it play for about five seconds, then pause.
3. Enable **Show skeleton** and point to the probability display and timeline playback marker.
4. Point to a reference marker or a matched event if available.
5. Scroll to **Detection threshold**; type a rehearsed value such as `0.70`, then show the changing event summary or default-versus-selected table. The event count may increase or decrease as intervals split or merge; narrate what actually happens.
6. Briefly show **Video evaluation metrics**, focusing on sensitivity and precision if available. Reset to `0.50` if time permits.

Say:

> “Each possible event links directly to the source video. The probability timeline follows playback, and the skeleton shows the pose data used by the video pipeline. Reference timing helps distinguish matched and model-only detections. Changing the analysis threshold updates the displayed events and label-backed metrics immediately, using saved probabilities. It does not rerun the model.”

Do not read the whole metrics list. The skeleton colours indicate pose confidence, not why the model predicted a seizure.

### 1:55–2:35 — show synchronized multimodal evidence

Say:

> “This completed run uses a verified video–EEG pair. Each window shows EEG, aligned video, and fused probabilities, along with video coverage. The default weighting is eighty percent EEG and twenty percent video. Selecting a window lets me review its video and original EEG waveform for the same interval. Windows without sufficient video remain EEG-only.”

Point to one row and its probability columns. Select that window, show **Play selected window**, then select **EEG waveform**. Point to the blue shading for the selected interval. Use one channel for readability; show All channels only if it fits comfortably.

The evidence uses tabs, so show video and waveform sequentially. The waveform displays original recorded channels, not the model’s processed bipolar montage.

### 2:35–2:50 — compare experiments

Say:

> “Compare runs shows saved results side by side, including available metrics and processing time. Differences are Run A minus Run B, and warnings flag differences that may make a comparison unfair.”

Point to one meaningful row. Use the prepared selection; do not spend the slot searching dropdowns. Grey event-count differences do not automatically mean better or worse performance.

### 2:50–3:00 — export and close

Say:

> “Finally, I can export the reviewed result as a PDF. SeizureAI connects recording management, model predictions, evidence review, and reporting in a research prototype.”

Click **Export report** on the completed video result. Keep the app’s research-use context clear: these are possible events, not confirmed diagnoses.

## If a prepared feature is unavailable

- **No verified fusion result:** replace 1:55–2:35 with a completed EEG result and its detected windows/artifacts. Explain that verified synchronization is required for fusion, but do not present an unrelated pair as fused evidence. If no EEG run is available either, use the slot for video label filtering and event zoom.
- **No labels:** demonstrate prediction playback, threshold adjustment, and event changes. Say that evaluation metrics need reference labels. Replace the metrics action with Zoom to event.
- **No saved pose data:** skip Show skeleton and use Zoom to event. Older runs require a new inference run to produce pose artifacts.
- **No useful comparison pair:** use 2:35–2:50 for Processing details and downloads, showing actual device, runtime, and stored artifacts.
- **Preview stalls:** move to the probability/event timeline, then the cached fusion waveform. Do not wait through conversion during the talk.
- **Running job has not progressed:** move on to the completed run at 1:00. If the new job is unnecessary after the demo, stop it through the app when appropriate.
- **Running late:** cut the comparison visit first and keep the evidence review and export close. Never spend the remaining time waiting for inference.

## Features to keep for questions

System device/CPU/CUDA details; editable patient notes; archive/restore; artifact checksums and downloads; appearance, contrast, and reduced-motion preferences; detailed runtime stages; all-channel EEG inspection; comparison PDF export.

If asked about the models: the video and EEG encoders use pretrained checkpoints, and testing does not train them. The deployed fusion is weighted probability fusion, not the untrained feature-fusion network. Do not quote a training accuracy as expected performance on a new recording or claim fusion improves accuracy without a valid evaluation.

## Pocket cue card

**0:00 Dashboard → 0:15 Upload → 0:40 Run/progress → 1:00 Video event + skeleton + threshold → 1:55 Fusion window + EEG → 2:35 Compare → 2:50 Export → 3:00 Stop.**

## Code anchors used for this guide

- `fyp_data_platform/frontend/src/App.jsx`: routes and navigation.
- `fyp_data_platform/frontend/src/pages/UploadPage.jsx`: actual upload modes, fields, and label controls.
- `fyp_data_platform/frontend/src/pages/Recordings.jsx`: actual launch buttons and progress controls.
- `fyp_data_platform/frontend/src/pages/RunResult.jsx`, `VideoTimelinePlayer.jsx`, `SkeletonOverlay.jsx`, and `analysis.js`: result interactions and threshold analysis.
- `fyp_data_platform/frontend/src/FusionTimeline.jsx` and `FusionEvidence.jsx`: window selection and video/EEG evidence tabs.
- `fyp_data_platform/frontend/src/pages/RunCompare.jsx` and `pdfReport.js`: comparison and PDF export.
- `fyp_data_platform/fyp_data_platform/ui.py`, `integrations/seizure_model.py`, and `db/models.py`: API, execution, storage persistence, and catalog relationships.
- `FYPModels/seizure_multimodal/video/raw_pipeline.py`, `eeg/eeg_adapter.py`, `eeg/recording_inference.py`, `synchronization/video_segments.py`, `fusion/late_fusion.py`, and `configs/inference.yaml`: model input paths, verified synchronization, and fusion settings.
