@extends('layouts.app')

@section('title', 'Dashboard')
@section('page-title', 'Main Dashboard')

@section('content')

<div class="page-header">
    <div class="page-title">Overview</div>
    <div class="page-sub">Real-time system activity and recent analysis summary</div>
</div>

{{-- =====================================================
     STAT CARDS
====================================================== --}}
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-label">Total Cases</div>
        <div class="stat-val">{{ $totalCases ?? 142 }}</div>
        <div class="stat-delta delta-up"><i class="ti ti-arrow-up"></i> 12 this month</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Seizures Detected</div>
        <div class="stat-val">{{ $seizuresDetected ?? 89 }}</div>
        <div class="stat-delta delta-up"><i class="ti ti-arrow-up"></i> 94.2% recall</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Avg AUROC</div>
        <div class="stat-val">{{ $avgAuroc ?? '0.91' }}</div>
        <div class="stat-delta delta-up"><i class="ti ti-arrow-up"></i> +0.03 vs baseline</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Pending Review</div>
        <div class="stat-val" style="color:#fbbf24;">{{ $pendingReview ?? 7 }}</div>
        <div class="stat-delta text-muted">Awaiting clinician sign-off</div>
    </div>
</div>

{{-- =====================================================
     BOTTOM ROW — TABLE + CHARTS
====================================================== --}}
<div class="flex-row">

    {{-- Recent analyses table --}}
    <div class="card" style="flex:2;">
        <div class="section-title">Recent analyses</div>
        <table>
            <thead>
                <tr>
                    <th>Patient ID</th>
                    <th>Duration</th>
                    <th>Phase Detected</th>
                    <th>Confidence</th>
                    <th>Status</th>
                    <th></th>
                </tr>
            </thead>
            <tbody>
                {{-- Loop over recent cases from controller --}}
                @forelse ($recentCases ?? [] as $case)
                    <tr>
                        <td class="td-main">{{ $case->patient_id }}</td>
                        <td>{{ $case->duration }} min</td>
                        <td>
                            <span class="phase-pill phase-{{ strtolower(str_replace('-', '', $case->phase)) }}">
                                {{ $case->phase }}
                            </span>
                        </td>
                        <td>{{ $case->confidence }}%</td>
                        <td>
                            @if ($case->status === 'Validated')
                                <span class="badge badge-green">Validated</span>
                            @elseif ($case->status === 'Pending')
                                <span class="badge badge-amber">Pending</span>
                            @else
                                <span class="badge badge-red">Flagged</span>
                            @endif
                        </td>
                        <td>
                            <a href="{{ route('results', $case->id) }}" class="btn btn-sm">View</a>
                        </td>
                    </tr>
                @empty
                    {{-- Fallback static rows for UI development --}}
                    <tr>
                        <td class="td-main">P-0047</td>
                        <td>18 min</td>
                        <td><span class="phase-pill phase-ictal"><i class="ti ti-bolt"></i> Ictal</span></td>
                        <td>94%</td>
                        <td><span class="badge badge-green">Validated</span></td>
                        <td><a href="{{ route('results') }}" class="btn btn-sm">View</a></td>
                    </tr>
                    <tr>
                        <td class="td-main">P-0051</td>
                        <td>32 min</td>
                        <td><span class="phase-pill phase-preictal"><i class="ti ti-clock"></i> Pre-ictal</span></td>
                        <td>78%</td>
                        <td><span class="badge badge-amber">Pending</span></td>
                        <td><a href="{{ route('results') }}" class="btn btn-sm">View</a></td>
                    </tr>
                    <tr>
                        <td class="td-main">P-0039</td>
                        <td>9 min</td>
                        <td><span class="phase-pill phase-postictal"><i class="ti ti-wave-sine"></i> Post-ictal</span></td>
                        <td>87%</td>
                        <td><span class="badge badge-green">Validated</span></td>
                        <td><a href="{{ route('results') }}" class="btn btn-sm">View</a></td>
                    </tr>
                    <tr>
                        <td class="td-main">P-0063</td>
                        <td>24 min</td>
                        <td><span class="phase-pill phase-ictal"><i class="ti ti-bolt"></i> Ictal</span></td>
                        <td>91%</td>
                        <td><span class="badge badge-red">Flagged</span></td>
                        <td><a href="{{ route('results') }}" class="btn btn-sm">View</a></td>
                    </tr>
                @endforelse
            </tbody>
        </table>
    </div>

    {{-- Charts column --}}
    <div class="card" style="flex:1;">
        <div class="section-title">Phase detection breakdown</div>
        <div class="chart-row">
            <div class="chart-label">Pre-ictal</div>
            <div class="chart-track"><div class="chart-fill" style="width:55%; background:#fbbf24;"></div></div>
            <div class="chart-val">55%</div>
        </div>
        <div class="chart-row">
            <div class="chart-label">Ictal</div>
            <div class="chart-track"><div class="chart-fill" style="width:30%; background:#f87171;"></div></div>
            <div class="chart-val">30%</div>
        </div>
        <div class="chart-row">
            <div class="chart-label">Post-ictal</div>
            <div class="chart-track"><div class="chart-fill" style="width:15%; background:#60a5fa;"></div></div>
            <div class="chart-val">15%</div>
        </div>

        <hr class="divider" />

        <div class="section-title">System health</div>
        <div class="chart-row">
            <div class="chart-label">GPU Utilisation</div>
            <div class="chart-track"><div class="chart-fill" style="width:72%; background:#3b82f6;"></div></div>
            <div class="chart-val">72%</div>
        </div>
        <div class="chart-row">
            <div class="chart-label">Queue</div>
            <div class="chart-track"><div class="chart-fill" style="width:20%; background:#34d399;"></div></div>
            <div class="chart-val">3</div>
        </div>
        <div class="chart-row">
            <div class="chart-label">Storage used</div>
            <div class="chart-track"><div class="chart-fill" style="width:48%; background:#8b5cf6;"></div></div>
            <div class="chart-val">48%</div>
        </div>
    </div>

</div>

@endsection
