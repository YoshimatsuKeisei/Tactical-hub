import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { afterEach, describe, expect, it } from "vitest";

const temporaryDirectories: string[] = [];
afterEach(() => { while (temporaryDirectories.length) rmSync(temporaryDirectories.pop()!, { recursive: true, force: true }); });

function runCli(args: string[]) {
  return spawnSync(process.execPath, [resolve("node_modules/vite-node/vite-node.mjs"), resolve("src/game/cpu/headlessCli.ts"), ...args], { cwd: process.cwd(), encoding: "utf8", timeout: 30_000 });
}

describe("Headless CLI argument initialization", () => {
  it.each([{ name: "normal", extra: [] as string[] }, { name: "profile", extra: ["--profile"] }])("initializes matchCount before $name execution", ({ extra }) => {
    const result = runCli(["--participants", "4", "--matches", "1", "--max-turns", "0", "--workers", "1", ...extra]);
    expect(result.status, result.stderr).toBe(0);
    expect(result.stderr).not.toContain("Cannot access 'matchCount' before initialization");
    expect(JSON.parse(result.stdout)).toMatchObject({ requestedWorkerCount: 1, effectiveWorkerCount: 1 });
  });

  it("initializes matchCount and traceFile before the trace execution path", () => {
    const directory = mkdtempSync(join(tmpdir(), "tactical-headless-cli-"));
    temporaryDirectories.push(directory);
    const tracePath = join(directory, "trace.jsonl");
    const result = runCli(["--participants", "4", "--matches", "1", "--max-turns", "0", "--workers", "1", "--trace-file", tracePath]);
    expect(result.status, result.stderr).toBe(0);
    expect(existsSync(tracePath)).toBe(true);
  });

  it("compares traces without parsing unrelated match arguments", () => {
    const directory = mkdtempSync(join(tmpdir(), "tactical-headless-compare-"));
    temporaryDirectories.push(directory);
    const left = join(directory, "left.jsonl"), right = join(directory, "right.jsonl");
    writeFileSync(left, "", "utf8"); writeFileSync(right, "", "utf8");
    const result = runCli(["--compare-traces", left, right, "--participants", "invalid", "--matches", "invalid"]);
    expect(result.status, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({ equal: true });
  });
});
