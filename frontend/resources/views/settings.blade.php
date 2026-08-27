@extends('layouts.app')

@section('title', 'Settings')
@section('page-title', 'Settings')

@section('content')

<div class="page-header">
    <div class="page-title">Settings</div>
    <div class="page-sub">System configuration, account management, and data governance</div>
</div>

{{-- Flash messages --}}
@if (session('success'))
    <div style="background:#062010; border:0.5px solid #0a3a1a; border-radius:8px; padding:12px 16px; margin-bottom:16px; font-size:13px; color:#34d399;">
        <i class="ti ti-circle-check"></i> {{ session('success') }}
    </div>
@endif

<div class="grid-2">

    {{-- =====================================================
         ACCOUNT
    ====================================================== --}}
    <div class="card">
        <div class="section-title">Account</div>
        <form method="POST" action="{{ route('settings.account') }}">
            @csrf
            @method('PATCH')

            <div class="form-group">
                <label class="form-label" for="name">Full name</label>
                <input class="form-input" type="text" id="name" name="name"
                       value="{{ old('name', auth()->user()->name ?? '') }}" required />
            </div>

            <div class="form-group">
                <label class="form-label" for="email">Email</label>
                <input class="form-input" type="email" id="email" name="email"
                       value="{{ old('email', auth()->user()->email ?? '') }}" required />
            </div>

            <div class="form-group">
                <label class="form-label" for="institution">Institution</label>
                <input class="form-input" type="text" id="institution" name="institution"
                       value="{{ old('institution', auth()->user()->institution ?? 'Monash University Malaysia') }}" />
            </div>

            <div class="form-group" style="margin-bottom:0;">
                <label class="form-label" for="role">Role</label>
                <select class="form-input" id="role" name="role">
                    <option value="researcher"  {{ (auth()->user()->role ?? '') === 'researcher'  ? 'selected' : '' }}>Student Researcher</option>
                    <option value="clinician"   {{ (auth()->user()->role ?? '') === 'clinician'   ? 'selected' : '' }}>Clinician / Neurologist</option>
                    <option value="admin"       {{ (auth()->user()->role ?? '') === 'admin'       ? 'selected' : '' }}>System Administrator</option>
                </select>
            </div>

            <button type="submit" class="btn btn-sm btn-primary mt-14">
                <i class="ti ti-device-floppy"></i> Save changes
            </button>
        </form>
    </div>

    {{-- =====================================================
         MODEL CONFIGURATION
    ====================================================== --}}
    <div class="card">
        <div class="section-title">Model configuration</div>
        <form method="POST" action="{{ route('settings.model') }}">
            @csrf
            @method('PATCH')

            <div class="form-group">
                <label class="form-label" for="fusion_strategy">Default fusion strategy</label>
                <select class="form-input" id="fusion_strategy" name="fusion_strategy">
                    <option value="late"  {{ ($modelConfig->fusion_strategy ?? 'late') === 'late'  ? 'selected' : '' }}>Late fusion (cross-modal attention)</option>
                    <option value="early" {{ ($modelConfig->fusion_strategy ?? 'late') === 'early' ? 'selected' : '' }}>Early fusion (concatenated tokens)</option>
                </select>
            </div>

            <div class="form-group">
                <label class="form-label" for="confidence_threshold">
                    Confidence threshold for flagging (%)
                </label>
                <input class="form-input" type="number" id="confidence_threshold"
                       name="confidence_threshold"
                       value="{{ old('confidence_threshold', $modelConfig->confidence_threshold ?? 70) }}"
                       min="50" max="99" />
                <div style="font-size:11px; color:#3a5070; margin-top:5px;">
                    Results below this % are auto-flagged for review
                </div>
            </div>

            <div class="form-group">
                <label class="form-label" for="explainability">Default explainability overlay</label>
                <select class="form-input" id="explainability" name="explainability">
                    <option value="gradcam_pose" {{ ($modelConfig->explainability ?? 'gradcam_pose') === 'gradcam_pose' ? 'selected' : '' }}>Grad-CAM + Pose attention (recommended)</option>
                    <option value="gradcam"      {{ ($modelConfig->explainability ?? '') === 'gradcam'      ? 'selected' : '' }}>Grad-CAM only</option>
                    <option value="none"         {{ ($modelConfig->explainability ?? '') === 'none'         ? 'selected' : '' }}>None</option>
                </select>
            </div>

            <div class="form-group" style="margin-bottom:0;">
                <label class="form-label" for="model_version">Active model version</label>
                <select class="form-input" id="model_version" name="model_version">
                    <option value="v1.0">v1.0 – Multimodal (Face + Pose + Flow)</option>
                    <option value="v0.9">v0.9 – Face + Pose only</option>
                    <option value="v0.7">v0.7 – Face only (baseline)</option>
                </select>
            </div>

            <button type="submit" class="btn btn-sm btn-primary mt-14">
                <i class="ti ti-device-floppy"></i> Save configuration
            </button>
        </form>
    </div>

    {{-- =====================================================
         NOTIFICATIONS
    ====================================================== --}}
    <div class="card">
        <div class="section-title">Notifications</div>
        <form method="POST" action="{{ route('settings.notifications') }}">
            @csrf
            @method('PATCH')

            @php
                $notifs = $notifications ?? (object)[
                    'email_on_complete'  => true,
                    'low_confidence'     => true,
                    'weekly_report'      => false,
                    'review_reminders'   => false,
                    'escalation_alerts'  => true,
                ];
            @endphp

            <div class="toggle">
                <div class="toggle-track {{ $notifs->email_on_complete  ? 'on' : '' }}" data-name="email_on_complete">
                    <div class="toggle-thumb"></div>
                </div>
                <div class="toggle-label">Email when analysis is complete</div>
                <input type="hidden" name="email_on_complete" class="toggle-input" value="{{ $notifs->email_on_complete ? '1' : '0' }}" />
            </div>

            <div class="toggle">
                <div class="toggle-track {{ $notifs->low_confidence ? 'on' : '' }}" data-name="low_confidence">
                    <div class="toggle-thumb"></div>
                </div>
                <div class="toggle-label">Alert on low-confidence result</div>
                <input type="hidden" name="low_confidence" class="toggle-input" value="{{ $notifs->low_confidence ? '1' : '0' }}" />
            </div>

            <div class="toggle">
                <div class="toggle-track {{ $notifs->weekly_report ? 'on' : '' }}" data-name="weekly_report">
                    <div class="toggle-thumb"></div>
                </div>
                <div class="toggle-label">Weekly summary report</div>
                <input type="hidden" name="weekly_report" class="toggle-input" value="{{ $notifs->weekly_report ? '1' : '0' }}" />
            </div>

            <div class="toggle">
                <div class="toggle-track {{ $notifs->review_reminders ? 'on' : '' }}" data-name="review_reminders">
                    <div class="toggle-thumb"></div>
                </div>
                <div class="toggle-label">Clinician review reminders</div>
                <input type="hidden" name="review_reminders" class="toggle-input" value="{{ $notifs->review_reminders ? '1' : '0' }}" />
            </div>

            <div class="toggle">
                <div class="toggle-track {{ $notifs->escalation_alerts ? 'on' : '' }}" data-name="escalation_alerts">
                    <div class="toggle-thumb"></div>
                </div>
                <div class="toggle-label">Flagged case escalation alerts</div>
                <input type="hidden" name="escalation_alerts" class="toggle-input" value="{{ $notifs->escalation_alerts ? '1' : '0' }}" />
            </div>

            <button type="submit" class="btn btn-sm btn-primary mt-14">
                <i class="ti ti-device-floppy"></i> Save preferences
            </button>
        </form>
    </div>

    {{-- =====================================================
         DATA & PRIVACY
    ====================================================== --}}
    <div class="card">
        <div class="section-title">Data & privacy</div>
        <div style="font-size:12px; color:#8faac8; line-height:1.9; margin-bottom:16px;">
            <div>· All vEEG data is de-identified before storage</div>
            <div>· IRB approval: MUMHREC 2025/018 · HUKM EC/1028</div>
            <div>· Data retention: 5 years per Monash data policy</div>
            <div>· Encryption: AES-256 at rest, TLS 1.3 in transit</div>
            <div>· Audit log retained for all data access events</div>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <a href="{{ route('settings.irb') }}" class="btn btn-sm">
                <i class="ti ti-file-certificate"></i> View IRB documents
            </a>
            <a href="{{ route('settings.audit') }}" class="btn btn-sm">
                <i class="ti ti-file-description"></i> Download audit log
            </a>
            <form method="POST" action="{{ route('settings.delete-request') }}" style="display:inline;"
                  onsubmit="return confirm('Are you sure you want to request data deletion? This cannot be undone.')">
                @csrf
                <button type="submit" class="btn btn-sm btn-danger">
                    <i class="ti ti-trash"></i> Request deletion
                </button>
            </form>
        </div>
    </div>

</div>

@endsection

@push('scripts')
<script>
    // Toggle switches — update hidden input value when toggled
    document.querySelectorAll('.toggle-track').forEach(function(track) {
        track.addEventListener('click', function() {
            this.classList.toggle('on');
            // Find the sibling hidden input in the same .toggle wrapper
            const input = this.closest('.toggle').querySelector('.toggle-input');
            if (input) input.value = this.classList.contains('on') ? '1' : '0';
        });
    });
</script>
@endpush
