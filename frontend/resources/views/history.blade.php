@extends('layouts.app')

@section('title', 'Historical Data')
@section('page-title', 'Historical Data Search')

@section('content')

<div class="page-header">
    <div class="page-title">Historical Data Search</div>
    <div class="page-sub">Search and filter past vEEG analyses</div>
</div>

{{-- =====================================================
     SEARCH & FILTER BAR
====================================================== --}}
<form method="GET" action="{{ route('history') }}">
    <div class="card mb-16" style="margin-bottom:16px;">
        <div class="grid-3" style="align-items:flex-end;">

            <div class="form-group" style="margin:0;">
                <label class="form-label">Patient ID</label>
                <input class="form-input" type="text" name="patient_id"
                       value="{{ request('patient_id') }}" placeholder="Search by ID…" />
            </div>

            <div class="form-group" style="margin:0;">
                <label class="form-label">Phase detected</label>
                <select class="form-input" name="phase">
                    <option value="">All phases</option>
                    <option value="preictal"  {{ request('phase') === 'preictal'  ? 'selected' : '' }}>Pre-ictal</option>
                    <option value="ictal"     {{ request('phase') === 'ictal'     ? 'selected' : '' }}>Ictal</option>
                    <option value="postictal" {{ request('phase') === 'postictal' ? 'selected' : '' }}>Post-ictal</option>
                </select>
            </div>

            <div class="form-group" style="margin:0;">
                <label class="form-label">Date from</label>
                <input class="form-input" type="date" name="date_from" value="{{ request('date_from') }}" />
            </div>

            <div class="form-group" style="margin:0;">
                <label class="form-label">Confidence ≥</label>
                <select class="form-input" name="confidence">
                    <option value="">Any</option>
                    <option value="70" {{ request('confidence') === '70' ? 'selected' : '' }}>70%</option>
                    <option value="80" {{ request('confidence') === '80' ? 'selected' : '' }}>80%</option>
                    <option value="90" {{ request('confidence') === '90' ? 'selected' : '' }}>90%</option>
                </select>
            </div>

            <div class="form-group" style="margin:0;">
                <label class="form-label">Status</label>
                <select class="form-input" name="status">
                    <option value="">All</option>
                    <option value="validated" {{ request('status') === 'validated' ? 'selected' : '' }}>Validated</option>
                    <option value="pending"   {{ request('status') === 'pending'   ? 'selected' : '' }}>Pending</option>
                    <option value="flagged"   {{ request('status') === 'flagged'   ? 'selected' : '' }}>Flagged</option>
                </select>
            </div>

            <button type="submit" class="btn btn-primary" style="align-self:flex-end;">
                <i class="ti ti-search"></i> Search
            </button>

        </div>
    </div>
</form>

{{-- =====================================================
     RESULTS TABLE
====================================================== --}}
<div class="card">
    <div class="flex-between" style="margin-bottom:14px;">
        <div class="section-title" style="margin:0;">
            {{ 142 }} records found
        </div>
        <a href="{{ route('history.export', request()->query()) }}" class="btn btn-sm">
            <i class="ti ti-download"></i> Export CSV
        </a>
    </div>

    <table>
        <thead>
            <tr>
                <th>Patient ID</th>
                <th>Date</th>
                <th>Duration</th>
                <th>Phase</th>
                <th>Confidence</th>
                <th>Modalities</th>
                <th>Status</th>
                <th></th>
            </tr>
        </thead>
        <tbody>
            @forelse ($cases ?? [] as $case)
                <tr>
                    <td class="td-main">{{ $case->patient_id }}</td>
                    <td>{{ \Carbon\Carbon::parse($case->recording_date)->format('d M Y') }}</td>
                    <td>{{ $case->duration }} min</td>
                    <td>
                        @php $phase = strtolower(str_replace(['-',' '], '', $case->phase)); @endphp
                        <span class="phase-pill phase-{{ $phase }}">{{ $case->phase }}</span>
                    </td>
                    <td>{{ $case->confidence }}%</td>
                    <td><span class="badge badge-gray">{{ $case->modalities }}</span></td>
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
                        <a href="{{ route('results', $case->id) }}" class="btn btn-sm">Open</a>
                    </td>
                </tr>
            @empty
                {{-- Fallback static rows for UI development --}}
                <tr>
                    <td class="td-main">P-0047</td><td>12 May 2026</td><td>18 min</td>
                    <td><span class="phase-pill phase-ictal">Ictal</span></td>
                    <td>94%</td>
                    <td><span class="badge badge-gray">Face · Pose · Flow</span></td>
                    <td><span class="badge badge-green">Validated</span></td>
                    <td><a href="{{ route('results') }}" class="btn btn-sm">Open</a></td>
                </tr>
                <tr>
                    <td class="td-main">P-0051</td><td>10 May 2026</td><td>32 min</td>
                    <td><span class="phase-pill phase-preictal">Pre-ictal</span></td>
                    <td>78%</td>
                    <td><span class="badge badge-gray">Face · Pose</span></td>
                    <td><span class="badge badge-amber">Pending</span></td>
                    <td><a href="{{ route('results') }}" class="btn btn-sm">Open</a></td>
                </tr>
                <tr>
                    <td class="td-main">P-0039</td><td>8 May 2026</td><td>9 min</td>
                    <td><span class="phase-pill phase-postictal">Post-ictal</span></td>
                    <td>87%</td>
                    <td><span class="badge badge-gray">Face · Flow</span></td>
                    <td><span class="badge badge-green">Validated</span></td>
                    <td><a href="{{ route('results') }}" class="btn btn-sm">Open</a></td>
                </tr>
                <tr>
                    <td class="td-main">P-0063</td><td>6 May 2026</td><td>24 min</td>
                    <td><span class="phase-pill phase-ictal">Ictal</span></td>
                    <td>91%</td>
                    <td><span class="badge badge-gray">Face · Pose · Flow</span></td>
                    <td><span class="badge badge-red">Flagged</span></td>
                    <td><a href="{{ route('results') }}" class="btn btn-sm">Open</a></td>
                </tr>
                <tr>
                    <td class="td-main">P-0028</td><td>2 May 2026</td><td>41 min</td>
                    <td><span class="phase-pill phase-preictal">Pre-ictal</span></td>
                    <td>82%</td>
                    <td><span class="badge badge-gray">All modalities</span></td>
                    <td><span class="badge badge-green">Validated</span></td>
                    <td><a href="{{ route('results') }}" class="btn btn-sm">Open</a></td>
                </tr>
            @endforelse
        </tbody>
    </table>

    {{-- Pagination --}}
    <div style="display:flex; justify-content:center; align-items:center; gap:8px; margin-top:18px;">
    <button class="btn btn-sm"><i class="ti ti-arrow-left"></i></button>
    <span class="badge badge-blue" style="padding:5px 16px; font-size:12px;">Page 1 of 12</span>
    <button class="btn btn-sm"><i class="ti ti-arrow-right"></i></button>
</div>

</div>

@endsection
