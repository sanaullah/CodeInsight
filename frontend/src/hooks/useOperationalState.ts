import { useCallback, useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { CapabilitiesResponse, HealthResponse } from "../api/contracts";

export interface OperationalState {
  health: HealthResponse | null;
  capabilities: CapabilitiesResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useOperationalState(): OperationalState {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    void revision;
    const controller = new AbortController();
    setLoading(true);
    Promise.all([apiClient.health(controller.signal), apiClient.capabilities(controller.signal)])
      .then(([nextHealth, nextCapabilities]) => {
        setHealth(nextHealth);
        setCapabilities(nextCapabilities);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "API unavailable");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [revision]);

  return { health, capabilities, loading, error, refresh };
}
