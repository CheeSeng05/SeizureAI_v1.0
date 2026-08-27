@extends('layouts.app')

@section('title', 'Analysis Results')
@section('page-title', 'Analysis Results')

@section('content')

<div class="page-header">
    <div class="flex-between">
        <div>
            <div class="page-title">
                Analysis Results · {{ $case->patient_id ?? 'P-0047' }}
            </div>
            <div class="page-sub">
                Session recorded {{ isset($case) ? \Carbon\Carbon::parse($case->recording_date)->format('d M Y') : '12 May 2026' }}
                · {{ $case->duration ?? 18 }} min
                · {{ $case->ward ?? 'HUKM Ward 7B' }}
            </div>
        </div>
        <div style="display:flex; gap:8px;">
            <a href="{{ route('results.export', $case->id ?? 1) }}" class="btn btn-sm">
                <i class="ti ti-download"></i> Export PDF
            </a>
            <form method="POST" action="{{ route('results.validate', $case->id ?? 1) }}" style="display:inline;">
                @csrf
                @method('PATCH')
                <button type="submit" class="btn btn-sm btn-primary">
                    <i class="ti ti-check"></i> Validate
                </button>
            </form>
        </div>
    </div>
</div>

<div class="flex-row" style="align-items:flex-start;">

    {{-- =====================================================
         LEFT COLUMN — Timeline, Waveform, Findings
    ====================================================== --}}
    <div style="flex:1; display:flex; flex-direction:column; gap:16px;">

        {{-- Seizure timeline --}}
        <div class="card">
            <div class="section-title">Seizure timeline</div>
            <div style="display:flex; gap:16px; margin-bottom:8px; font-size:11px; color:#3a5070;">
                <span style="display:inline-flex;align-items:center;gap:5px;">
                    <span style="width:10px;height:10px;background:#0f2a40;border-radius:2px;display:inline-block;"></span>Normal
                </span>
                <span style="display:inline-flex;align-items:center;gap:5px;">
                    <span style="width:10px;height:10px;background:#2a1800;border-radius:2px;display:inline-block;"></span>Pre-ictal
                </span>
                <span style="display:inline-flex;align-items:center;gap:5px;">
                    <span style="width:10px;height:10px;background:#2a0a0a;border-radius:2px;display:inline-block;"></span>Ictal
                </span>
            </div>
            <div class="timeline-track">
                <div class="timeline-seg seg-normal"   style="left:0%;  width:35%;"></div>
                <div class="timeline-seg seg-preictal" style="left:35%; width:22%;"></div>
                <div class="timeline-seg seg-ictal"    style="left:57%; width:28%;">SEIZURE</div>
                <div class="timeline-seg seg-normal"   style="left:85%; width:15%;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#3a5070; margin-top:5px;">
                <span>0:00</span>
                <span>{{ $case->preictal_onset ?? '4:20' }} Pre-ictal</span>
                <span>{{ $case->ictal_onset ?? '8:18' }} Ictal</span>
                <span>{{ $case->ictal_offset ?? '13:12' }} Offset</span>
                <span>{{ $case->duration ?? 18 }}:00</span>
            </div>
        </div>

        {{-- Feature signal waveform --}}
        <div class="card">
            <div class="section-title">Feature signal — facial AUs + pose</div>
            <div class="waveform" id="waveform"></div>
            <div style="display:flex; gap:16px; margin-top:8px; font-size:11px; color:#3a5070;">
                <span style="color:#fbbf24;">■ Pre-ictal AU surge</span>
                <span style="color:#f87171;">■ Ictal motor dysregulation</span>
                <span style="color:#1e3a5c;">■ Baseline</span>
            </div>
        </div>

        {{-- AI key findings table --}}
        <div class="card">
            <div class="section-title">Key AI findings</div>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Feature</th>
                        <th>Observation</th>
                        <th>Significance</th>
                    </tr>
                </thead>
                <tbody>
                    @forelse ($findings ?? [] as $finding)
                        <tr>
                            <td>{{ $finding->timestamp }}</td>
                            <td class="td-main">{{ $finding->feature }}</td>
                            <td>{{ $finding->observation }}</td>
                            <td><span class="badge badge-{{ $finding->badge_type }}">{{ $finding->significance }}</span></td>
                        </tr>
                    @empty
                        {{-- Fallback static rows --}}
                        <tr>
                            <td>T−3:12</td>
                            <td class="td-main">AU4 (Brow lowerer)</td>
                            <td>Sustained activation (&gt;2.1 s)</td>
                            <td><span class="badge badge-amber">Pre-ictal marker</span></td>
                        </tr>
                        <tr>
                            <td>T+0:00</td>
                            <td class="td-main">Oro-facial automatism</td>
                            <td>Repetitive lip-smacking detected</td>
                            <td><span class="badge badge-red">Ictal onset</span></td>
                        </tr>
                        <tr>
                            <td>T+0:42</td>
                            <td class="td-main">Dystonic posturing</td>
                            <td>Right arm flexion / stiffening</td>
                            <td><span class="badge badge-red">Lateralisation R</span></td>
                        </tr>
                        <tr>
                            <td>T+4:54</td>
                            <td class="td-main">Pose stabilisation</td>
                            <td>Limb movement amplitude ↓</td>
                            <td><span class="badge badge-blue">Post-ictal onset</span></td>
                        </tr>
                    @endforelse
                </tbody>
            </table>
        </div>

    </div>

    {{-- =====================================================
         RIGHT COLUMN — Confidence, Modality scores, Annotation
    ====================================================== --}}
    <div style="width:230px; flex-shrink:0; display:flex; flex-direction:column; gap:12px;">

        {{-- Confidence ring --}}
        <div class="card-sm" style="text-align:center;">
            <div class="section-title" style="margin-bottom:12px;">Overall confidence</div>
            <div class="conf-ring">{{ $case->confidence ?? 94 }}%</div>
            <div style="font-size:12px; color:#3a5070; margin-top:10px;">Ictal phase detected</div>
            <div style="font-size:11px; color:#3a5070; margin-top:4px;">
                Lateralisation: {{ $case->lateralisation ?? 'Right TLE' }}
            </div>
        </div>

        {{-- Modality scores --}}
        <div class="card-sm">
            <div class="section-title">Modality scores</div>
            <div class="chart-row" style="margin-bottom:8px;">
                <div class="chart-label" style="width:80px; font-size:11px;">Face</div>
                <div class="chart-track"><div class="chart-fill" style="width:{{ $case->face_score ?? 93 }}%; background:#3b82f6;"></div></div>
                <div class="chart-val" style="font-size:11px;">{{ $case->face_score ?? 93 }}%</div>
            </div>
            <div class="chart-row" style="margin-bottom:8px;">
                <div class="chart-label" style="width:80px; font-size:11px;">Pose</div>
                <div class="chart-track"><div class="chart-fill" style="width:{{ $case->pose_score ?? 89 }}%; background:#8b5cf6;"></div></div>
                <div class="chart-val" style="font-size:11px;">{{ $case->pose_score ?? 89 }}%</div>
            </div>
            <div class="chart-row">
                <div class="chart-label" style="width:80px; font-size:11px;">Flow</div>
                <div class="chart-track"><div class="chart-fill" style="width:{{ $case->flow_score ?? 82 }}%; background:#06b6d4;"></div></div>
                <div class="chart-val" style="font-size:11px;">{{ $case->flow_score ?? 82 }}%</div>
            </div>
        </div>

        {{-- Clinician annotation --}}
        <div class="card-sm">
            <div class="section-title">Clinician annotation</div>
            <form method="POST" action="{{ route('results.annotate', $case->id ?? 1) }}">
                @csrf
                @method('PATCH')
                <textarea class="form-input" name="annotation" rows="4"
                          placeholder="Add clinical notes and observations…">{{ $case->annotation ?? '' }}</textarea>
                <button type="submit" class="btn btn-sm w-full justify-center" style="margin-top:8px;">
                    <i class="ti ti-device-floppy"></i> Save note
                </button>
            </form>
        </div>

        {{-- Agreement --}}
        <div class="card-sm">
            <div class="section-title">Agreement</div>
            <div style="font-size:12px; color:#8faac8; line-height:1.8;">
                <div>Cohen's κ: <strong style="color:#34d399;">{{ $case->kappa ?? '0.84' }}</strong></div>
                <div>Clinician: <strong style="color:#b0ccee;">{{ $case->clinician_label ?? 'Ictal – R TLE' }}</strong></div>
                <div>AI output: <strong style="color:#b0ccee;">{{ $case->ai_label ?? 'Ictal – R TLE' }}</strong></div>
            </div>
        </div>

    </div>
</div>

@endsection

@push('scripts')
<script>
    // Render waveform bars
    (function() {
        const wf = document.getElementById('waveform');
        if (!wf) return;
        for (let i = 0; i < 90; i++) {
            const bar = document.createElement('div');
            bar.className = 'wf-bar';
            let h = 10 + Math.random() * 25;
            if (i >= 30 && i < 48) {
                h = 18 + Math.random() * 22;
                bar.style.background = '#d97706';
            } else if (i >= 48 && i < 72) {
                h = 22 + Math.random() * 28;
                bar.style.background = '#ef4444';
            }
            bar.style.height = h + 'px';
            wf.appendChild(bar);
        }
    })();
</script>
@endpush
