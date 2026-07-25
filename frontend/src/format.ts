export function formatLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function formatDuration(start: string | null, end?: string | null): string {
  if (!start) return "Not started";
  const seconds = Math.max(
    0,
    Math.round((new Date(end ?? Date.now()).getTime() - new Date(start).getTime()) / 1000),
  );
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

export function shortId(value: string): string {
  return value.length <= 12 ? value : `${value.slice(0, 8)}…${value.slice(-4)}`;
}

export function formatTokens(value: number): string {
  return `${value.toLocaleString()} tokens`;
}
