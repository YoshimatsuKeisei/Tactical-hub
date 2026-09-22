/**
 * Opt-in wall-clock timings for PPO smoke. Never changes actions or checkpoints.
 * Node's RPC duration includes pack + IPC + Python prepare + PyTorch inference.
 */
export class PpoTimingProfiler {
  readonly enabled = process.env.PPO_PROFILE === "1";
  private readonly totals = new Map<string, { count: number; totalMs: number }>();

  private add(stage: string, elapsedMs: number) {
    const item = this.totals.get(stage) ?? { count: 0, totalMs: 0 };
    item.count += 1;
    item.totalMs += elapsedMs;
    this.totals.set(stage, item);
  }

  measure<T>(stage: string, operation: () => T): T {
    if (!this.enabled) return operation();
    const start = performance.now();
    try { return operation(); } finally { this.add(stage, performance.now() - start); }
  }

  async measureAsync<T>(stage: string, operation: () => Promise<T>): Promise<T> {
    if (!this.enabled) return operation();
    const start = performance.now();
    try { return await operation(); } finally { this.add(stage, performance.now() - start); }
  }

  report(label: string) {
    if (!this.enabled) return;
    const stages = Object.fromEntries([...this.totals].map(([key, value]) => [key, {
      count: value.count, totalMs: Math.round(value.totalMs * 100) / 100,
      avgMs: Math.round(value.totalMs / value.count * 1000) / 1000,
    }]));
    process.stderr.write(`[PPO profile node] ${JSON.stringify({ label, stages })}\n`);
  }
}
