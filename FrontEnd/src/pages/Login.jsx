import { useState } from 'react';
import { Brain } from 'lucide-react';

export default function Login({ onSignIn }) {
  const [email, setEmail] = useState('clinician@monash.edu');
  const [password, setPassword] = useState('');

  const submit = (e) => {
    e.preventDefault();
    onSignIn();
  };

  return (
    <div className="grid h-full place-items-center bg-gradient-to-br from-sky-50 to-blue-50 p-6">
      <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-9 shadow-sm">
        <div className="mb-7 text-center">
          <div className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-sky-400 to-blue-600">
            <Brain className="h-7 w-7 text-white" />
          </div>
          <h1 className="text-2xl font-semibold text-slate-900">SeizureAI</h1>
          <p className="mt-1 text-sm text-slate-500">Monash University Malaysia · FYP 2026</p>
        </div>

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label htmlFor="email" className="mb-1.5 block text-sm text-slate-600">Email</label>
            <input
              id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm"
            />
          </div>
          <div>
            <label htmlFor="password" className="mb-1.5 block text-sm text-slate-600">Password</label>
            <input
              id="password" type="password" value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm"
            />
          </div>
          <button
            type="submit"
            className="w-full rounded-lg bg-blue-600 py-3 font-medium text-white transition hover:bg-blue-700"
          >
            Sign in
          </button>
        </form>

        <p className="mt-6 text-center text-xs leading-relaxed text-slate-400">
          Research prototype · not for clinical use
          <br />
          Sign-in is a placeholder; add real authentication before any deployment.
        </p>
      </div>
    </div>
  );
}