# Video + EEG uploads

Choose **Video + EEG**, select an EDF or Nicolet `.e` recording and the corresponding video files, and enter a shared Recording ID.

For EDF, enter each video's start in seconds relative to EEG recording start. Use `0` only if that video and EEG start together. These offsets are human-declared synchronization, checked against recording durations; they are not automatically measured from EDF. Nicolet `.e` continues to use its embedded synchronization metadata.

Enable **Compare video + EEG with ground truth** and choose either:

- **Upload labels**: timestamped CSV or the existing `Label.xlsx` workbook format. Workbook labels require the matching Patient ID, Seizure ID, and Seizure type.
- **Type time intervals**: enter start/end seconds and choose Seizure or Non-seizure for each interval. These annotations are stored as `human_annotations.csv`.

All multimodal labels use seconds from **EEG start**, including videos that begin later. Intervals must be finite, non-negative, non-overlapping and within EEG duration. Label CSV example:

```csv
start_sec,end_sec,label
0,10,0
10,20,1
20,50,0
```

`1` or `S` means seizure; `0` or `N` means non-seizure. Patient and recording columns are optional; supplied identities must match the recording being uploaded. Gaps remain unlabelled, so provide reviewed non-seizure intervals when evaluating false positives or specificity.

From **Registered recordings**, choose **Evaluate fusion** for comparison with human labels. **Run Multimodal Fusion** remains a prediction-only run. Evaluation applies the same majority-overlap policy as video CSV labels, excluding equal-overlap ties, unlabelled windows, and windows without sufficient video coverage. Results include classification metrics and per-window ground truth; annotations are never model inputs.
