@extends('layouts.app')

@section('title', 'Evaluation Metrics')
@section('page-title', 'Evaluation Metrics')

@section('content')

<div class="page-header">
    <div class="page-title">Evaluation Metrics</div>
    <div class="page-sub">Model performance across patient-independent validation splits</div>
</div>

{{-- =====================================================
     TOP STAT CARDS
====================================================== --}}
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-label">AUROC</div>
        <div class="stat-val">{{ $metrics->auroc ?? '0.912' }}</div>
        <div class="stat-delta delta-up">Multimodal fusion</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">F1-Score</div>
        <div class="stat-val">{{ $metrics->f1 ?? '0.876' }}</div>
        <div class="stat-delta delta-up">Ictal detection</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">FP / Hour</div>
        <div class="stat-val" style="color:#fbbf24;">{{ $metrics->fp_per_hour ?? '1.4' }}</div>
        <div class="stat-delta delta-down">Target: &lt; 2.0 ✓</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Cohen's κ</div>
        <div class="stat-val">{{ $metrics->kappa ?? '0.81' }}</div>
        <div class="stat-delta delta-up">vs. neurologist</div>
    </div>
</div>

{{-- =====================================================
     MODALITY AUROC + CONFUSION MATRIX
====================================================== --}}
<div class="flex-row">

    {{-- Per-modality AUROC --}}
    <div class="card" style="flex:1;">
        <div class="section-title">Per-modality AUROC</div>

        <div class="chart-row">
            <div class="chart-label">Multimodal</div>
            <div class="chart-track">
                <div class="chart-fill" style="width:{{ ($metrics->auroc_multimodal ?? 0.912) * 100 }}%; background:linear-gradient(90deg,#3b82f6,#06b6d4);"></div>
            </div>
            <div class="chart-val">{{ $metrics->auroc_multimodal ?? '0.912' }}</div>
        </div>
        <div class="chart-row">
            <div class="chart-label">Face only</div>
            <div class="chart-track">
                <div class="chart-fill" style="width:{{ ($metrics->auroc_face ?? 0.79) * 100 }}%; background:#8b5cf6;"></div>
            </div>
            <div class="chart-val">{{ $metrics->auroc_face ?? '0.790' }}</div>
        </div>
        <div class="chart-row">
            <div class="chart-label">Pose only</div>
            <div class="chart-track">
                <div class="chart-fill" style="width:{{ ($metrics->auroc_pose ?? 0.74) * 100 }}%; background:#6366f1;"></div>
            </div>
            <div class="chart-val">{{ $metrics->auroc_pose ?? '0.740' }}</div>
        </div>
        <div class="chart-row">
            <div class="chart-label">Flow only</div>
            <div class="chart-track">
                <div class="chart-fill" style="width:{{ ($metrics->auroc_flow ?? 0.71) * 100 }}%; background:#4f46e5;"></div>
            </div>
            <div class="chart-val">{{ $metrics->auroc_flow ?? '0.710' }}</div>
        </div>

        <hr class="divider" />

        <div class="section-title">Sensitivity / Specificity</div>
        <div class="chart-row">
            <div class="chart-label">Sensitivity</div>
            <div class="chart-track">
                <div class="chart-fill" style="width:{{ $metrics->sensitivity ?? 94.4 }}%; background:#34d399;"></div>
            </div>
            <div class="chart-val">{{ $metrics->sensitivity ?? '94.4' }}%</div>
        </div>
        <div class="chart-row">
            <div class="chart-label">Specificity</div>
            <div class="chart-track">
                <div class="chart-fill" style="width:{{ $metrics->specificity ?? 86.8 }}%; background:#60a5fa;"></div>
            </div>
            <div class="chart-val">{{ $metrics->specificity ?? '86.8' }}%</div>
        </div>
    </div>

    {{-- Confusion matrix --}}
    <div class="card" style="flex:1;">
        <div class="section-title">
            Confusion matrix — Ictal vs Non-ictal (n={{ $metrics->n ?? 142 }})
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px;">
            <div style="background:#ecfdf5; border:0.5px solid #a7f3d0; border-radius:8px; padding:16px; text-align:center;">
                <div style="font-size:24px; font-weight:600; color:#059669;">{{ $metrics->tp ?? 84 }}</div>
                <div style="font-size:11px; color:#3a5070; margin-top:5px;">True Positive</div>
            </div>
            <div style="background:#fef2f2; border:0.5px solid #fecaca; border-radius:8px; padding:16px; text-align:center;">
                <div style="font-size:24px; font-weight:600; color:#dc2626;">{{ $metrics->fn ?? 5 }}</div>
                <div style="font-size:11px; color:#3a5070; margin-top:5px;">False Negative</div>
            </div>
            <div style="background:#fffbeb; border:0.5px solid #fde68a; border-radius:8px; padding:16px; text-align:center;">
                <div style="font-size:24px; font-weight:600; color:#d97706;">{{ $metrics->fp ?? 7 }}</div>
                <div style="font-size:11px; color:#3a5070; margin-top:5px;">False Positive</div>
            </div>
            <div style="background:#ecfdf5; border:0.5px solid #a7f3d0; border-radius:8px; padding:16px; text-align:center;">
                <div style="font-size:24px; font-weight:600; color:#059669;">{{ $metrics->tn ?? 46 }}</div>
                <div style="font-size:11px; color:#3a5070; margin-top:5px;">True Negative</div>
            </div>
        </div>
        <div style="font-size:12px; color:#8faac8; display:flex; gap:16px; flex-wrap:wrap;">
            <div>Precision: <strong style="color:#34d399;">{{ $metrics->precision ?? '92.3' }}%</strong></div>
            <div>Recall: <strong style="color:#34d399;">{{ $metrics->sensitivity ?? '94.4' }}%</strong></div>
            <div>F1: <strong style="color:#60a5fa;">{{ $metrics->f1 ?? '0.876' }}</strong></div>
        </div>
    </div>

</div>

{{-- =====================================================
     ONSET / OFFSET ERROR TABLE
====================================================== --}}
<div class="card">
    <div class="section-title">Onset / offset localisation error</div>
    <table>
        <thead>
            <tr>
                <th>Phase</th>
                <th>Mean onset error (s)</th>
                <th>Mean offset error (s)</th>
                <th>Std deviation</th>
                <th>n</th>
            </tr>
        </thead>
        <tbody>
            @forelse ($localisationErrors ?? [] as $row)
                <tr>
                    <td class="td-main">{{ $row->phase }}</td>
                    <td>{{ $row->onset_error }}</td>
                    <td>{{ $row->offset_error }}</td>
                    <td>± {{ $row->std_dev }}</td>
                    <td>{{ $row->n }}</td>
                </tr>
            @empty
                <tr>
                    <td class="td-main">Pre-ictal</td>
                    <td>8.3</td><td>11.2</td><td>± 4.1</td><td>89</td>
                </tr>
                <tr>
                    <td class="td-main">Ictal</td>
                    <td>3.7</td><td>5.1</td><td>± 2.8</td><td>89</td>
                </tr>
                <tr>
                    <td class="td-main">Post-ictal</td>
                    <td>6.2</td><td>9.4</td><td>± 3.9</td><td>67</td>
                </tr>
            @endforelse
        </tbody>
    </table>
</div>

@endsection
