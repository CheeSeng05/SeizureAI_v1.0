<?php

use Illuminate\Support\Facades\Route;

/*
| SeizureAI Web Routes
*/

// Redirect root to login
Route::get('/', function () {
    return redirect()->route('login');
});

// Login page
Route::get('/login', function () {
    return view('auth.login');
})->name('login');

// Temporary placeholder POST for login form
Route::post('/login', function () {
    return redirect()->route('dashboard');
})->name('login.post');

// Logout
Route::post('/logout', function () {
    return redirect()->route('login');
})->name('logout');

// Main pages
Route::get('/dashboard', function () { return view('dashboard'); })->name('dashboard');
Route::get('/upload',    function () { return view('upload'); })->name('upload');
Route::get('/history',   function () { return view('history'); })->name('history');
Route::get('/results',   function () { return view('results'); })->name('results');
Route::get('/metrics',   function () { return view('metrics'); })->name('metrics');
Route::get('/settings',  function () { return view('settings'); })->name('settings');

// Placeholder POST routes (to avoid errors from forms)
Route::post('/upload',            function () { return redirect()->route('upload'); })->name('upload.store');
Route::get('/history/export',     function () { return redirect()->route('history'); })->name('history.export');
Route::patch('/results/{id}/validate', function () { return redirect()->route('results'); })->name('results.validate');
Route::patch('/results/{id}/annotate', function () { return redirect()->route('results'); })->name('results.annotate');
Route::get('/results/{id}/export',     function () { return redirect()->route('results'); })->name('results.export');
Route::patch('/settings/account',       function () { return redirect()->route('settings'); })->name('settings.account');
Route::patch('/settings/model',         function () { return redirect()->route('settings'); })->name('settings.model');
Route::patch('/settings/notifications', function () { return redirect()->route('settings'); })->name('settings.notifications');
Route::get('/settings/irb',             function () { return redirect()->route('settings'); })->name('settings.irb');
Route::get('/settings/audit',           function () { return redirect()->route('settings'); })->name('settings.audit');
Route::post('/settings/delete-request', function () { return redirect()->route('settings'); })->name('settings.delete-request');
