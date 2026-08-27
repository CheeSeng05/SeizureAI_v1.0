<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <title>SeizureAI – Sign In</title>

    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.44.0/tabler-icons.min.css" />
    <link rel="stylesheet" href="{{ asset('css/seizureai.css') }}" />
</head>
<body>

<div class="login-wrap">
    <div class="login-card">

        {{-- Logo --}}
        <div class="login-logo-wrap">
            <div class="login-logo-icon">
                <i class="ti ti-brain" style="font-size:24px; color:#fff;"></i>
            </div>
            <div class="login-title">SeizureAI</div>
            <div class="login-sub">Monash University Malaysia · FYP 2026</div>
        </div>

        {{-- Validation errors --}}
        @if ($errors->any())
            <div style="background:#200a0a; border:0.5px solid #3a1010; border-radius:8px; padding:10px 14px; margin-bottom:16px; font-size:12px; color:#f87171;">
                @foreach ($errors->all() as $error)
                    <div>{{ $error }}</div>
                @endforeach
            </div>
        @endif

        {{-- Session status (e.g. after password reset) --}}
        @if (session('status'))
            <div style="background:#062010; border:0.5px solid #0a3a1a; border-radius:8px; padding:10px 14px; margin-bottom:16px; font-size:12px; color:#34d399;">
                {{ session('status') }}
            </div>
        @endif

        {{-- Login form --}}
        <form method="POST" action="{{ route('login') }}">
            @csrf

            <div class="form-group">
                <label class="form-label" for="email">Email address</label>
                <input
                    class="form-input"
                    type="email"
                    id="email"
                    name="email"
                    value="{{ old('email') }}"
                    placeholder="clinician@monash.edu"
                    required
                    autofocus
                />
            </div>

            <div class="form-group">
                <label class="form-label" for="password">Password</label>
                <input
                    class="form-input"
                    type="password"
                    id="password"
                    name="password"
                    placeholder="••••••••"
                    required
                />
            </div>

            <div class="flex-between" style="margin-bottom:20px;">
                <label style="display:flex; align-items:center; gap:8px; font-size:12px; color:#7090b0; cursor:pointer;">
                    <input type="checkbox" name="remember" style="accent-color:#3b82f6;" />
                    Remember me
                </label>
                @if (Route::has('password.request'))
                    <a href="{{ route('password.request') }}" style="font-size:12px; color:#3b82f6; text-decoration:none;">
                        Forgot password?
                    </a>
                @endif
            </div>

            <button type="submit" class="btn btn-primary w-full justify-center">
                <i class="ti ti-login"></i> Sign in
            </button>
        </form>

        <div class="login-footer">
            Authorised personnel only<br/>
            MUMHREC 2025/018 · HUKM EC/1028 approved
        </div>

    </div>
</div>

</body>
</html>
