import { afterEach, describe, expect, it, vi } from "vitest";
import { createInitialGameState } from "../initialState";
import { HttpBrowserBcInferenceClient, type BrowserBcInferenceClient, type BrowserBcInferenceRequest } from "../cpu/browserBcClient";
import { advanceVisualCpuOneStepWithBc, createBcInferenceRequest } from "../cpu/browserBcPolicy";
import { createVisualCpuPolicyRouter, getVisualCpuActorTeamId, isCpuController } from "../cpu/cpuPolicyRouter";
import { buildRlObservation } from "../cpu/rlEnvironment";
import { RL_ACTION_TYPES } from "../cpu/rlActionEncoder";
import { RL_UNIT_TYPES } from "../cpu/rlObservationEncoder";
import { createCpuRuntime, type CpuTeamSettings } from "../cpu/types";
import type { Unit } from "../types";

const settings = (teamOne: CpuTeamSettings[string] = "bc_cpu"): CpuTeamSettings => ({
  "team-1": teamOne,
  "team-2": "human",
  "team-3": "random_cpu",
  "team-4": "heuristic_cpu",
});

const clientReturning = (actionKey: (request: BrowserBcInferenceRequest) => string): BrowserBcInferenceClient => ({
  health: async () => true,
  infer: async (request) => ({ actionKey: actionKey(request) }),
});

describe("browser BC CPU", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("accepts bc_cpu and leaves Random/Heuristic routing separate", () => {
    const state = createInitialGameState();
    const runtime = createCpuRuntime(7);
    expect(isCpuController("bc_cpu")).toBe(true);
    expect(getVisualCpuActorTeamId(state, runtime, settings())).toBe("team-1");
    expect(createVisualCpuPolicyRouter()(state, runtime, settings())).toBeUndefined();
    expect(createVisualCpuPolicyRouter()(state, runtime, settings("human"))).toBeUndefined();
  });

  it("uses the existing observation/action encoders and applies a returned legal actionKey", async () => {
    const state = createInitialGameState();
    const runtime = createCpuRuntime(8);
    const prepared = createBcInferenceRequest(state, runtime, settings())!;
    expect(prepared.request).toBeDefined();
    const request = prepared.request!;
    expect(request.featureSpec.schemaVersion).toBe(1);
    expect(request.observation.schemaVersion).toBe(1);
    expect(request.legalActions.schemaVersion).toBe(1);
    expect(request.legalActions.actionKeys).toHaveLength(request.legalActions.actions.length);
    const selectedKey = request.legalActions.actionKeys[0];
    const result = await advanceVisualCpuOneStepWithBc(state, runtime, settings(), createVisualCpuPolicyRouter(), clientReturning(() => selectedKey));
    expect(result.applied).toBe(true);
    expect(result.runtime.logs.at(-1)).toMatchObject({ teamId: "team-1", action: "BC selected", detail: selectedKey });
  });

  it("does not apply illegal actionKeys or fall back when the server is unavailable", async () => {
    const state = createInitialGameState();
    const runtime = createCpuRuntime(9);
    const illegal = await advanceVisualCpuOneStepWithBc(state, runtime, settings(), createVisualCpuPolicyRouter(), clientReturning(() => "illegal"));
    expect(illegal.applied).toBe(false);
    expect(illegal.state).toBe(state);
    expect(illegal.runtime.stoppedReason).toContain("illegal or stale actionKey");

    const unavailable: BrowserBcInferenceClient = { health: async () => false, infer: async () => { throw new TypeError("Failed to fetch"); } };
    const failed = await advanceVisualCpuOneStepWithBc(state, runtime, settings(), createVisualCpuPolicyRouter(), unavailable);
    expect(failed.applied).toBe(false);
    expect(failed.runtime.stoppedReason).toBe("BC inference server unavailable");
  });

  it("drops a stale response without applying it", async () => {
    const state = createInitialGameState();
    const runtime = createCpuRuntime(10);
    let resolve!: (value: { actionKey: string }) => void;
    const client: BrowserBcInferenceClient = { health: async () => true, infer: (request) => new Promise((done) => { resolve = done; void request; }) };
    let current = true;
    const pending = advanceVisualCpuOneStepWithBc(state, runtime, settings(), createVisualCpuPolicyRouter(), client, () => current);
    const key = createBcInferenceRequest(state, runtime, settings())!.request!.legalActions.actionKeys[0];
    current = false;
    resolve({ actionKey: key });
    const result = await pending;
    expect(result).toEqual({ state, runtime, applied: false });
  });

  it("deduplicates an in-flight HTTP request for the same decision", async () => {
    let finish!: () => void;
    const fetchMock = vi.fn(() => new Promise<Response>((resolve) => {
      finish = () => resolve(new Response(JSON.stringify({ decisionKey: "same", actionKey: "a" }), { status: 200, headers: { "content-type": "application/json" } }));
    }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new HttpBrowserBcInferenceClient("http://test");
    const request = { decisionKey: "same", featureSpec: {}, observation: {}, legalActions: { actions: [[1]], actionKeys: ["a"] } } as unknown as BrowserBcInferenceRequest;
    const first = client.infer(request);
    const second = client.infer(request);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    finish();
    await expect(first).resolves.toEqual({ actionKey: "a" });
    await expect(second).resolves.toEqual({ actionKey: "a" });
  });

  it("uses team-scoped observation and keeps legacy v1 schemas free of heavy merge actions", () => {
    const state = createInitialGameState();
    const hidden: Unit = { id: "hidden-water", ownerTeamId: "team-2", type: "ninja", hp: 1, position: { kind: "water", x: 4, y: 2 }, statuses: [] };
    const own: Unit = { ...hidden, id: "own-water", ownerTeamId: "team-1", position: { kind: "water", x: 5, y: 2 } };
    state.units.push(hidden, own);
    const observation = buildRlObservation(state, "team-1", "team-1");
    expect(observation.units.map((unit) => unit.id)).toContain(own.id);
    expect(observation.units.map((unit) => unit.id)).not.toContain(hidden.id);
    expect(RL_UNIT_TYPES).not.toContain("heavy_infantry");
    expect(RL_ACTION_TYPES.some((kind) => /merge|heavy/i.test(kind))).toBe(false);
  });
});
