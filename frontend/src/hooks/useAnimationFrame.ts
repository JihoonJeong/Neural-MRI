import { useCallback, useEffect, useRef } from 'react';

export function useAnimationFrame(callback: (time: number) => void, active = true) {
  const rafRef = useRef<number>(0);
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  const animate = useCallback((time: number) => {
    callbackRef.current(time);
    rafRef.current = requestAnimationFrame(animate);
  }, []);

  useEffect(() => {
    if (active) {
      rafRef.current = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(rafRef.current);
    }
  }, [active, animate]);
}
