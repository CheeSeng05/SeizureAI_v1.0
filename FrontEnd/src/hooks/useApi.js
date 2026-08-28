import { useState, useEffect, useRef } from "react";

/** Fetch on mount / when deps change. Returns {data, error, loading}. */
export function useApi(fn, deps = []) {
    const [state, setState] = useState({ data:null, error: null, loading: true });
    useEffect(() => {
        let alive = true;
        setState((s) => ({...s, loading: true}));
        fn()
            .then((data) => alive && setState({ data, error: null, loading: false}))
            .catch((error) => alive && setState({ data: null, error, loading: false}));
        return () => {alive = false};

    }, deps);
    return state;
}

/**
 * Debounced scoring. A slider fires dozens of change events per drag; without
 * this the backend gets hammered and responses arrive out of order.
 */
export function useScoring(api, recordingId, threshold, minConsecutive, delay = 120) {
  const [state, setState] = useState({ data: null, loading: false, error: null });
  const seq = useRef(0);
 
  useEffect(() => {
    if (!recordingId) return;
    const mine = ++seq.current;
    setState((s) => ({ ...s, loading: true }));
    const timer = setTimeout(() => {
      api.score(recordingId, threshold, minConsecutive)
        .then((data) => {
          if (mine === seq.current) setState({ data, loading: false, error: null });
        })
        .catch((error) => {
          if (mine === seq.current) setState({ data: null, loading: false, error });
        });
    }, delay);
    return () => clearTimeout(timer);
  }, [recordingId, threshold, minConsecutive]);
 
  return state;
}