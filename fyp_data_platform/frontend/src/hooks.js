import { useCallback, useEffect, useState } from 'react';

export function useResource(loader, dependencies = [], { pollInterval = 0, shouldPoll = () => true } = {}) {
  const [version, setVersion] = useState(0);
  const [state, setState] = useState({ data: null, loading: true, error: null });
  const refresh = useCallback(() => setVersion((value) => value + 1), []);
  useEffect(() => {
    let alive = true;
    let timer;
    setState((current) => ({ ...current, loading: true }));
    const load = () => loader().then((data) => {
      if (!alive) return;
      setState({ data, loading: false, error: null });
      if (pollInterval && shouldPoll(data)) timer = setTimeout(load, pollInterval);
    }).catch((error) => {
      if (!alive) return;
      setState((current) => ({ ...current, loading: false, error }));
      if (pollInterval) timer = setTimeout(load, pollInterval);
    });
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, [...dependencies, version, pollInterval]);
  return { ...state, refresh };
}
