import { useCallback, useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { AnalysisIntelligence, AnalysisRun } from "../api/contracts";

const terminal = new Set(["succeeded", "failed", "cancelled", "needs_attention"]);

export function useRun(runId: string) {
  const [run, setRun] = useState<AnalysisRun | null>(null);
  const [intelligence, setIntelligence] = useState<AnalysisIntelligence | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [intelligenceError, setIntelligenceError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    void revision;
    let active = true;
    let timer: number | undefined;
    const controller = new AbortController();

    const load = async () => {
      try {
        const nextRun = await apiClient.getRun(runId, controller.signal);
        if (!active) return;
        setRun(nextRun);
        setError(null);
        setLoading(false);
        try {
          const nextIntelligence = await apiClient.getIntelligence(runId, controller.signal);
          if (!active) return;
          setIntelligence(nextIntelligence);
          setIntelligenceError(null);
        } catch (reason) {
          if (!active || controller.signal.aborted) return;
          setIntelligenceError(
            reason instanceof Error
              ? reason.message
              : "Review intelligence is temporarily unavailable",
          );
        }
        if (!terminal.has(nextRun.status)) {
          timer = window.setTimeout(load, 1_500);
        }
      } catch (reason) {
        if (!active || controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "Unable to load review");
        setLoading(false);
      }
    };

    void load();
    return () => {
      active = false;
      controller.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, [runId, revision]);

  return { run, intelligence, loading, error, intelligenceError, refresh, setRun };
}
