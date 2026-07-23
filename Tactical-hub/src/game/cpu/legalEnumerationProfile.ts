export type LegalProfileSink = (category: string, milliseconds: number) => void;

let activeSink: LegalProfileSink | undefined;

export function isLegalProfilingEnabled() { return activeSink !== undefined; }

export function withLegalProfileSink<T>(sink: LegalProfileSink | undefined, operation: () => T): T {
  const previous = activeSink;
  activeSink = sink;
  try { return operation(); }
  finally { activeSink = previous; }
}

export function measureLegalSegment<T>(category: string, operation: () => T): T {
  if (!activeSink) return operation();
  const started = performance.now();
  const result = operation();
  activeSink(category, performance.now() - started);
  return result;
}

export function recordLegalSegment(category: string, milliseconds: number) {
  activeSink?.(category, milliseconds);
}
