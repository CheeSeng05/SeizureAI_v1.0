@extends('layouts.app')

@section('title', 'Upload Data')
@section('page-title', 'Upload vEEG Data')

@section('content')

<div class="page-header">
    <div class="page-title">Upload vEEG Data</div>
    <div class="page-sub">Upload anonymised video-EEG recordings for AI analysis</div>
</div>

{{-- Success / error flash messages --}}
@if (session('success'))
    <div style="background:#062010; border:0.5px solid #0a3a1a; border-radius:8px; padding:12px 16px; margin-bottom:16px; font-size:13px; color:#34d399;">
        <i class="ti ti-circle-check"></i> {{ session('success') }}
    </div>
@endif

@if ($errors->any())
    <div style="background:#200a0a; border:0.5px solid #3a1010; border-radius:8px; padding:12px 16px; margin-bottom:16px; font-size:13px; color:#f87171;">
        @foreach ($errors->all() as $error)
            <div><i class="ti ti-alert-circle"></i> {{ $error }}</div>
        @endforeach
    </div>
@endif

<form method="POST" action="{{ route('upload.store') }}" enctype="multipart/form-data" id="upload-form">
    @csrf

    {{-- Drop zone --}}
    <div class="card mb-16" style="margin-bottom:16px;">
        <div class="upload-zone" onclick="document.getElementById('video-file').click()">
            <div class="upload-icon"><i class="ti ti-cloud-upload"></i></div>
            <div class="upload-text">Drop vEEG video files here, or click to browse</div>
            <div class="upload-sub">Supported: .mp4, .avi, .mkv · Max 4 GB per file · DICOM-linked metadata accepted</div>
        </div>

        {{-- Hidden file input --}}
        <input
            type="file"
            id="video-file"
            name="veeg_video"
            accept=".mp4,.avi,.mkv"
            style="display:none;"
            onchange="showFileProgress(this)"
        />

        {{-- Upload progress (shown after file selected) --}}
        <div id="upload-progress" style="display:none; margin-top:16px;">
            <div class="flex-between" style="font-size:12px; color:#7090b0; margin-bottom:4px;">
                <span id="upload-filename">No file selected</span>
                <span id="upload-pct">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="upload-fill" style="width:0%;"></div>
            </div>
        </div>
    </div>

    <div class="grid-2">

        {{-- Patient metadata --}}
        <div class="card">
            <div class="section-title">Patient metadata</div>

            <div class="form-group">
                <label class="form-label" for="patient_id">Patient ID (anonymised)</label>
                <input class="form-input" type="text" id="patient_id" name="patient_id"
                       value="{{ old('patient_id') }}" placeholder="e.g. P-0071" required />
            </div>

            <div class="form-group">
                <label class="form-label" for="recording_date">Recording date</label>
                <input class="form-input" type="date" id="recording_date" name="recording_date"
                       value="{{ old('recording_date') }}" required />
            </div>

            <div class="form-group">
                <label class="form-label" for="session_type">Session type</label>
                <select class="form-input" id="session_type" name="session_type">
                    <option value="routine" {{ old('session_type') === 'routine' ? 'selected' : '' }}>Routine monitoring</option>
                    <option value="presurgical" {{ old('session_type') === 'presurgical' ? 'selected' : '' }}>Pre-surgical evaluation</option>
                    <option value="postsurgery" {{ old('session_type') === 'postsurgery' ? 'selected' : '' }}>Post-surgery follow-up</option>
                </select>
            </div>

            <div class="form-group">
                <label class="form-label" for="ward">Recording ward / unit</label>
                <input class="form-input" type="text" id="ward" name="ward"
                       value="{{ old('ward') }}" placeholder="e.g. HUKM Ward 7B" />
            </div>

            <div class="form-group" style="margin-bottom:0;">
                <label class="form-label" for="notes">Clinician notes (optional)</label>
                <textarea class="form-input" id="notes" name="notes"
                          rows="3" placeholder="e.g. Suspected TLE, right hemisphere…">{{ old('notes') }}</textarea>
            </div>
        </div>

        {{-- Analysis configuration --}}
        <div class="card">
            <div class="section-title">Analysis configuration</div>

            <div class="form-group">
                <label class="form-label" for="epilepsy_type">Epilepsy type</label>
                <select class="form-input" id="epilepsy_type" name="epilepsy_type">
                    <option value="tle">Temporal Lobe Epilepsy (TLE)</option>
                    <option value="focal_other">Focal (other)</option>
                    <option value="generalised">Generalised</option>
                </select>
            </div>

            <div class="form-group">
                <label class="form-label">Feature modalities</label>
                <div class="tag-row">
                    <label style="cursor:pointer;">
                        <input type="checkbox" name="modalities[]" value="facial_aus" checked hidden />
                        <span class="badge badge-blue modality-badge" onclick="toggleModality(this)">✓ Facial AUs</span>
                    </label>
                    <label style="cursor:pointer;">
                        <input type="checkbox" name="modalities[]" value="body_pose" checked hidden />
                        <span class="badge badge-blue modality-badge" onclick="toggleModality(this)">✓ Body Pose</span>
                    </label>
                    <label style="cursor:pointer;">
                        <input type="checkbox" name="modalities[]" value="optical_flow" checked hidden />
                        <span class="badge badge-blue modality-badge" onclick="toggleModality(this)">✓ Optical Flow</span>
                    </label>
                    <label style="cursor:pointer;">
                        <input type="checkbox" name="modalities[]" value="audio" hidden />
                        <span class="badge badge-gray modality-badge" onclick="toggleModality(this)">+ Audio</span>
                    </label>
                </div>
            </div>

            <div class="form-group">
                <label class="form-label" for="fusion_strategy">Fusion strategy</label>
                <select class="form-input" id="fusion_strategy" name="fusion_strategy">
                    <option value="late">Late fusion (cross-modal attention)</option>
                    <option value="early">Early fusion (concatenated tokens)</option>
                </select>
            </div>

            <div class="form-group">
                <label class="form-label" for="explainability">Explainability overlay</label>
                <select class="form-input" id="explainability" name="explainability">
                    <option value="gradcam_pose">Grad-CAM + Pose attention (recommended)</option>
                    <option value="gradcam">Grad-CAM only</option>
                    <option value="none">None</option>
                </select>
            </div>

            <div class="form-group" style="margin-bottom:0;">
                <label class="form-label" for="confidence_threshold">Confidence threshold for flagging (%)</label>
                <input class="form-input" type="number" id="confidence_threshold"
                       name="confidence_threshold" value="{{ old('confidence_threshold', 70) }}"
                       min="50" max="99" />
            </div>

            <button type="submit" class="btn btn-primary w-full justify-center mt-14">
                <i class="ti ti-player-play"></i> Run Analysis
            </button>
        </div>

    </div>
</form>

@endsection

@push('scripts')
<script>
    function showFileProgress(input) {
        if (!input.files.length) return;
        const file = input.files[0];
        document.getElementById('upload-filename').textContent = file.name;
        document.getElementById('upload-progress').style.display = 'block';

        // Simulate progress bar (replace with real XHR upload progress in production)
        let p = 0;
        const fill = document.getElementById('upload-fill');
        const pct  = document.getElementById('upload-pct');
        const iv = setInterval(() => {
            p = Math.min(p + Math.random() * 7, 100);
            const rounded = Math.round(p);
            fill.style.width = rounded + '%';
            pct.textContent  = rounded + '%';
            if (p >= 100) {
                clearInterval(iv);
                pct.textContent = 'Ready ✓';
                pct.style.color = '#34d399';
            }
        }, 100);
    }

    function toggleModality(badge) {
        const checkbox = badge.closest('label').querySelector('input[type=checkbox]');
        const isOn = badge.classList.contains('badge-blue');
        if (isOn) {
            badge.classList.replace('badge-blue', 'badge-gray');
            badge.textContent = badge.textContent.replace('✓', '+');
            checkbox.checked = false;
        } else {
            badge.classList.replace('badge-gray', 'badge-blue');
            badge.textContent = badge.textContent.replace('+', '✓');
            checkbox.checked = true;
        }
    }
</script>
@endpush
