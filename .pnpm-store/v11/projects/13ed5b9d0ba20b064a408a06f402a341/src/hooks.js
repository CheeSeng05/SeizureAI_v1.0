import { useCallback, useEffect, useState } from 'react';

export function useResource(loader, dependencies = []) {
  const [version, setVersion] = useState(0);
  const [state, setState] = useState({ data: null, loading: true, error: null });
  const refresh = useCallback(() => setVersion((value) => value + 1), []);
  useEffect(() => {
    let alive = true;
    setState((current) => ({ ...current, loading: true }));
    loader().then((data) => alive && setState({ data, loading: false, error: null }))
      .catch((error) => alive && setState({ data: null, loading: false, error }));
    return () => { alive = false; };
  }, [...dependencies, version]);
  return { ...state, refresh };
}
