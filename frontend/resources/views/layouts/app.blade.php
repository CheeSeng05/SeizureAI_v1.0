<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <title>SeizureAI – @yield('title', 'Dashboard')</title>

    {{-- Google Fonts --}}
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
    {{-- Tabler Icons --}}
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.44.0/tabler-icons.min.css" />
    {{-- Main stylesheet --}}
    <link rel="stylesheet" href="{{ asset('css/seizureai.css') }}" />

    {{-- Page-specific styles --}}
    @stack('styles')
</head>
<body>

<div class="app">

    {{-- =====================================================
         SIDEBAR
    ====================================================== --}}
    <aside class="sidebar">

        {{-- Logo --}}
        <div class="logo">
            <div class="logo-mark">
                <div class="logo-icon">
                    <i class="ti ti-brain" style="color:#fff; font-size:17px;"></i>
                </div>
                <div>
                    <div class="logo-text">SeizureAI</div>
                    <div class="logo-sub">vEEG Analysis</div>
                </div>
            </div>
        </div>

        {{-- Navigation --}}
        <nav class="nav">
            <div class="nav-section">Core</div>

            <a href="{{ route('dashboard') }}"
               class="nav-item {{ request()->routeIs('dashboard') ? 'active' : '' }}">
                <i class="ti ti-layout-dashboard"></i> Dashboard
            </a>

            <a href="{{ route('upload') }}"
               class="nav-item {{ request()->routeIs('upload') ? 'active' : '' }}">
                <i class="ti ti-upload"></i> Upload Data
            </a>

            <a href="{{ route('history') }}"
               class="nav-item {{ request()->routeIs('history') ? 'active' : '' }}">
                <i class="ti ti-database-search"></i> Historical Data
            </a>

            <div class="nav-section">Analysis</div>

            <a href="{{ route('results') }}"
               class="nav-item {{ request()->routeIs('results') ? 'active' : '' }}">
                <i class="ti ti-report-analytics"></i> Results
            </a>

            <a href="{{ route('metrics') }}"
               class="nav-item {{ request()->routeIs('metrics') ? 'active' : '' }}">
                <i class="ti ti-chart-bar"></i> Eval Metrics
            </a>

            <div class="nav-section">System</div>

            <a href="{{ route('settings') }}"
               class="nav-item {{ request()->routeIs('settings') ? 'active' : '' }}">
                <i class="ti ti-settings"></i> Settings
            </a>
        </nav>

        {{-- User footer --}}
        <div class="sidebar-footer">
            <div class="user-row">
                <div class="avatar">
                    {{-- Show initials from auth user name --}}
                    {{ strtoupper(substr(auth()->user()->name ?? 'U', 0, 2)) }}
                </div>
                <div style="flex:1; min-width:0;">
                    <div class="user-name">{{ auth()->user()->name ?? 'Guest' }}</div>
                    <div class="user-role">{{ auth()->user()->role ?? 'Researcher' }}</div>
                </div>
                <form method="POST" action="{{ route('logout') }}" style="display:inline;">
                    @csrf
                    <button type="submit" style="background:none; border:none; cursor:pointer; padding:0;">
                        <i class="ti ti-logout" style="font-size:15px; color:#3a5070;"></i>
                    </button>
                </form>
            </div>
        </div>

    </aside>

    {{-- =====================================================
         MAIN CONTENT AREA
    ====================================================== --}}
    <div class="main">

        {{-- Top bar --}}
        <header class="topbar">
            <div class="topbar-title">@yield('page-title', 'Dashboard')</div>
            <div class="topbar-badge">
                <i class="ti ti-circle-dot"></i> Model v1.0 Ready
            </div>
            <button class="btn btn-sm" style="margin-left:8px;">
                <i class="ti ti-bell"></i>
            </button>
        </header>

        {{-- Page content injected here --}}
        <div class="page-container">
            @yield('content')
        </div>

    </div>
</div>

{{-- =====================================================
     GLOBAL SCRIPTS
====================================================== --}}
<script>
    // Toggle switch handler
    document.querySelectorAll('.toggle-track').forEach(function(track) {
        track.addEventListener('click', function() {
            this.classList.toggle('on');
        });
    });
</script>

{{-- Page-specific scripts --}}
@stack('scripts')

</body>
</html>
