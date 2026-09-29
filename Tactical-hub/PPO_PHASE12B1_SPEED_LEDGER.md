# Tactical-hub PPO Phase 12B-1 Speed Ledger

Updated: 2026-09-28

## Goal and fixed semantics
- Original 50k wall: ~1131.09 sec.
- 20x target: <=56.5545 sec / 50k decisions (~884.1 decisions/sec).
- Keep exactly 8 environments, same-policy collection, one PPO update after 8 episodes.
- Keep round_major_env_index merge order and deterministic resume/update behavior.
- Do not change game rules, Observation/Action semantics, reward, adjudication, PPO loss,
  checkpoint schema, trajectory semantics, seed management, or update/episode counters.

## Current verified baseline
- V5 branch: experiment/ppo-fast-batch-v5-compact-rows
- V5 commit: e0f832b874cc373abb776fd9c221d904f08e37fa
- V5 40k actual wall: 227.197 sec.
- V5 50k ~282.65 sec is projection only, not an actual 50k run.
- V5 stage timings at 40k:
  - batchAct 59.438 sec
  - retention replay 59.382 sec
  - observation encode 36.737 sec
  - game step 34.664 sec
  - action encode 28.380 sec
- V5 exactness: model/optimizer/CPU RNG/CUDA RNG/counters/rollout hashes/workload/
  retention/validation/all-episodes-learnable all PASS.
## Compact Action evidence
- Probe-01: 99.5798% Action zeros; sparse round-trip byte exact.
- Probe-02: direct sparse encoder byte exact; dense encode 1960.009 ms,
  direct sparse encode 1665.775 ms (1.17663x).
- Probe-03A:
  - dense Action bytes 1,012,480
  - sparse Action bytes 5,032
  - raw reduction 99.5030%
  - deflate reduction 65.3853%
  - CPU/GPU restored actions and actionMask exact
  - model outputs exact
  - sampled action/logProbability/value exact
  - CPU RNG and CUDA RNG after sampling exact
  - isolated dense transfer 33.919 ms
  - sparse transfer + GPU restore 16.062 ms (2.11181x)

## Closed / failed routes
- torch.compile: microprobe faster, actual 5k slower; do not retry casually.
- strict multistream: exact but insufficient speedup.
- exact state branch cache: ~1.1x only.
- grouped Replay H2D empty-skip: changed model/optimizer; no-skip exact but replay worsened.
- V4 Node worker pool: exact at 8k but wall worsened 58.253 -> 60.613 sec.
- Do not change 8 env to 16/32 without explicit approval.

## V6-A verified result
- Branch: experiment/ppo-fast-batch-v6-sparse-action-transport
- Candidate tested: 62d1cb57dbe4d46d5ced507aa6f3ab12c0d2d494
- 8 env, seeds 9-16, 100 rounds; 720 timed decisions/path after warmup.
- allExact=true; mismatchCount=0.
- parameter/optimizer/RNG/gradient hashes exact.
- final state hashes exact.
- Dense Action bytes 1,012,480 -> sparse 7,872 (99.2225% reduction).
- Whole packed payload 1,677,776 -> 673,168 (59.8774% reduction).
- Diagnostic round-trip: 1.657566 -> 1.521167 ms/decision (1.0897x).
- This timing is transport inference diagnostic only, NOT full PPO throughput.
- Direct sparse Action encoder is still NOT integrated; packer scans dense encoded rows.
## V6-B verified result
- Branch: experiment/ppo-fast-batch-v6-sparse-action-retention
- Candidate tested: 88dd57fcc5d047d081f6b05e7e1d84455e9750fb
- Workload: 8 env x safety-action-limit 500 = 4,000 decisions.
- semanticExact=true; checkpointCoreExact=true; allExact=true.
- model/optimizer/CPU RNG/CUDA RNG/counters/FeatureSpec/hyperparameters/seed all exact.
- retention structure exact and fully drained after replay.
- Dense retained raw bytes: 599,328,088.
- Sparse retained raw bytes: 340,830,768 (43.1312% reduction).
- Dense retained compressed bytes: 24,598,400.
- Sparse retained compressed bytes: 23,168,575 (5.8127% reduction).
- External wall: 31.577 sec -> 26.167 sec (1.2067x).
- Internal total: 30.080 sec -> 24.679 sec.
- Rollout: 16.368 sec -> 15.990 sec.
- Replay: 5.948 sec -> 5.075 sec.
- This is a 4k diagnostic smoke, not a 40k/50k throughput claim.

## V6-B 8k profiled verification
- Candidate tested: 085216a1342da2ea313305f4276eda61e7f4797f
- Workload: 8 env x 1,000 decisions = 8,000 decisions.
- allExact=true; semanticExact=true; checkpointCoreExact=true.
- V5 wall 54.480 sec -> V6-B wall 45.946 sec (1.1857x).
- Internal total 52.950 sec -> 44.480 sec.
- Rollout 32.743 sec -> 31.294 sec (~4.4% reduction).
- Replay 11.744 sec -> 9.330 sec (~20.6% reduction).
- batchAct 12.669 sec -> 12.067 sec (~4.8% reduction).
- Dense retained raw 1,331,131,880 -> sparse 687,613,680 bytes (48.3437% reduction).
- Dense compressed 50,668,397 -> sparse 47,070,103 bytes (7.1017% reduction).
- Python PPO_PROFILE has a pre-existing name-shadowing bug: nested record() is overwritten by
  loop variable record during retention replay, causing TypeError at finishUpdate when PPO_PROFILE=1.
  Node profiling is unaffected and was used for the successful 8k comparison.

## V6-C direct sparse Action encoder — 4k verified result
- Branch: experiment/ppo-fast-batch-v6-direct-sparse-actions
- Candidate tested: 1d7468a1015402932f42b82a1c1689b361850162
- Baseline: V6-B 085216a1342da2ea313305f4276eda61e7f4797f
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semanticExact=true; checkpointCoreExact=true.
- model/optimizer/CPU RNG/CUDA RNG/counters/FeatureSpec/hyperparameters/seed all exact.
- retention structure exact, fully drained, and retention bytes exactly equal.
- Retained raw bytes: 340,830,768 on both V6-B and V6-C.
- Retained compressed bytes: 23,168,575 on both V6-B and V6-C.
- Action encode: 2,771.44 ms -> 1,008.67 ms (~63.6% reduction, ~2.75x).
- batchAct: 7,620.58 ms -> 6,552.42 ms.
- Replay: 5,544.59 ms -> 5,066.12 ms.
- External wall: 34.313 sec -> 25.048 sec (1.3699x diagnostic).
- The 4k wall comparison shows substantial run-to-run variance; do not treat 1.3699x
  as the stable full-PPO speedup until an 8k comparison confirms it.
- Direct sparse encoder output and the prior dense-scan sparse transport payload were
  verified byte-identical before the Kaggle smoke.

## V6-C direct sparse Action encoder — 8k verified result
- Same implementation commit: 1d7468a1015402932f42b82a1c1689b361850162
- Baseline: V6-B 085216a1342da2ea313305f4276eda61e7f4797f
- Workload: 8 env x 1,000 decisions = 8,000 decisions.
- allExact=true; semanticExact=true; checkpointCoreExact=true.
- model/optimizer/CPU RNG/CUDA RNG/counters/FeatureSpec/hyperparameters/seed all exact.
- retention structure exact, fully drained, and retained payload bytes exactly equal.
- Retained raw bytes: 687,613,680 on both V6-B and V6-C.
- Retained compressed bytes: 47,070,103 on both V6-B and V6-C.
- Action encode: 4,951.94 ms -> 1,777.92 ms (~64.1% reduction, ~2.79x).
- batchAct: 12,996.03 ms -> 11,250.24 ms (~13.4% reduction).
- Replay: 9,046.05 ms -> 8,799.60 ms (~2.7% reduction; likely mostly run variance because retention bytes are identical).
- Rollout: 31.365 sec -> 25.982 sec (~17.2% reduction).
- Internal total: 49.580 sec -> 38.422 sec (~22.5% reduction).
- External wall: 51.122 sec -> 39.904 sec (1.2811x).
- The 4k and 8k runs both show the intended structural reduction; V6-C is now the
  fastest verified Action generation path under exact semantics.

## V6-D direct compact Observation — 4k verified result
- Branch: experiment/ppo-fast-batch-v6-direct-compact-observation
- Candidate tested: 95010f5aa74ffe766fe66d2eab2b643e0df76c5b
- Baseline: V6-C 1d7468a1015402932f42b82a1c1689b361850162
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semanticExact=true; checkpointCoreExact=true.
- model/optimizer/CPU RNG/CUDA RNG/counters/FeatureSpec/hyperparameters/seed all exact.
- retention structure, drain state, raw bytes and compressed bytes all exact.
- Retained raw bytes: 340,830,768 on both V6-C and V6-D.
- Retained compressed bytes: 23,168,575 on both V6-C and V6-D.
- Observation encode: 3,586.56 ms -> 2,656.40 ms (~25.9% reduction).
- Action encode: 982.91 ms -> 905.21 ms.
- batchAct: 6,678.99 ms -> 5,232.32 ms.
- Replay: 5,171.75 ms -> 4,934.78 ms.
- Rollout: 14.552 sec -> 12.041 sec (~17.3% reduction).
- External wall: 29.322 sec -> 22.232 sec (1.3189x diagnostic).
- Packed payload is byte-identical; the gain comes from avoiding padded zero-row
  allocation/copy before the already-existing packed-row compaction.

## V6-D direct compact Observation — 8k verified result
- Same implementation commit: 95010f5aa74ffe766fe66d2eab2b643e0df76c5b
- Baseline: V6-C 1d7468a1015402932f42b82a1c1689b361850162
- Workload: 8 env x 1,000 decisions = 8,000 decisions.
- allExact=true; semanticExact=true; checkpointCoreExact=true.
- model/optimizer/CPU RNG/CUDA RNG/counters/FeatureSpec/hyperparameters/seed all exact.
- retention structure, drain state, raw bytes and compressed bytes all exact.
- Retained raw bytes: 687,613,680 on both V6-C and V6-D.
- Retained compressed bytes: 47,070,103 on both V6-C and V6-D.
- Observation encode: 7,560.28 ms -> 5,186.57 ms (~31.4% reduction).
- Action encode: 2,091.69 ms -> 1,773.67 ms.
- batchAct: 14,532.36 ms -> 10,772.71 ms (~25.9% reduction).
- Replay: 10,667.44 ms -> 10,080.44 ms.
- Rollout: 31.644 sec -> 24.454 sec (~22.7% reduction).
- Internal total: 52.909 sec -> 38.680 sec (~26.9% reduction).
- External wall: 54.619 sec -> 40.395 sec (1.3521x paired diagnostic).
- V6-D is the current fastest verified baseline under exact semantics.

## Python internal profile — verified diagnostic
- Diagnostic candidate: 9cf510348dd81f42d9fa0dea3ae16d76b7efdb81
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; model/optimizer/CPU RNG/CUDA RNG/retention bytes all exact.
- Packed read binary: 97.33 ms.
- Packed prepare H2D/restore: 265.72 ms.
- Act model forward: 630.19 ms.
- Act finite/distribution/sampling/log-prob/host-scalar work: ~274 ms combined.
- Retention compress: 527.67 ms.
- Retained batch decompress/decode: 898.66 ms.
- Retained batch prepare H2D/restore: 102.53 ms.
- Replay model forward: 300.27 ms.
- Replay objective: 165.49 ms.
- Replay backward: 470.07 ms.
- Update finish: 138.76 ms.
- The profiler requires CUDA synchronization and is diagnostic only, not a speed benchmark.

## V6-E compact model rows — CLOSED
- Candidate: 7b097baaa8490e0901a29ba2ea2e6e4ade27960c
- Workload: 4,000 decisions.
- Wall 33.296 -> 25.403 sec; batchAct 7.113 -> 6.076 sec; replay 6.170 -> 5.799 sec.
- retention bytes/counters/RNG remained exact, but modelExact=false and optimizerExact=false.
- semanticDiffKeys were update and diagnostics.
- Conclusion: changing model input tensor shapes changes floating-point execution enough to break
  the project's bit-exact requirement. Do not adopt this route without a new exact-preserving method.

## V6-F raw retention — 4k verified result
- Branch: experiment/ppo-fast-batch-v6-raw-retention
- Candidate: c376f55a67ba5f473192f395bf64d20b2e907bd4
- Baseline: V6-D 5d5c5d13e7ebc846d5fd785ea2d91ea32d04006f
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- Same raw retained payload: 340,830,768 bytes.
- Deflate storage 23,168,575 bytes -> raw storage 340,830,768 bytes.
- Peak retained storage 23,168,575 -> 340,830,768 bytes.
- Kaggle host RAM: 33,659,383,808 bytes.
- 50k raw-retention linear projection: ~4.260 GB (~12.7% of host RAM).
- External wall 28.521 -> 20.790 sec (1.3719x paired diagnostic).
- Internal total 27.014 -> 19.325 sec.
- Rollout 12.027 -> 11.167 sec.
- Replay 5.311 -> 4.473 sec.
- batchAct 5.807 -> 5.239 sec.
- V6-F is the current fastest exact-verified baseline for continued optimization.

## V6-G persistent act H2D — 4k verified result
- Branch: experiment/ppo-fast-batch-v6-persistent-act-h2d
- Candidate: 45d13c43a36de3e0e3e0b5d79f8669c1bfac8e01
- Baseline: V6-F c376f55a67ba5f473192f395bf64d20b2e907bd4
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- Raw retention: 340,830,768 bytes on both paths.
- External wall 25.068 -> 20.749 sec (1.2082x paired diagnostic).
- Internal total 23.565 -> 19.231 sec.
- Rollout 11.666 -> 11.077 sec.
- Replay 4.614 -> 4.518 sec.
- batchAct 5,521.17 -> 5,107.68 ms (~7.5% reduction).
- V6-G is the current fastest exact-verified baseline.

## V6-I persistent Action restore buffers — 4k verified result
- Branch: experiment/ppo-fast-batch-v6-action-restore-workspace
- Candidate: fdb0706627cd8cc6ce2837ae2d83f59fabda58de
- Baseline: V6-G 45d13c43a36de3e0e3e0b5d79f8669c1bfac8e01
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- Raw retention: 340,830,768 bytes on both paths.
- batchAct: 5,741.44 -> 5,288.71 ms (~7.9% reduction).
- Replay: 4,851.97 -> 4,692.79 ms (~3.3% reduction).
- Rollout: 11.891 -> 11.525 sec.
- Internal total: 25.193 -> 19.844 sec.
- External wall: 26.724 -> 21.361 sec (1.2510x paired diagnostic).
- V6-I is the current fastest exact-verified baseline.
- The large wall difference exceeds the directly attributed stage reductions, so treat
  wall speedup as diagnostic; the stage-local batchAct/replay reductions are the trusted effect.

## Next steps
1. Keep V6-I as the current baseline.
2. Inspect exact-shape CUDA Graph capture for act model forward only.
3. Measure exact act-shape/nonempty-signature reuse before relying on graph caching.
4. Keep sampling, RNG consumption, finite checks and host-scalar behavior outside the graph.
5. Never pad or change model input shape solely to increase graph reuse; V6-E/V6-H proved
   shape/execution changes can break bit-exactness.
6. If exact-shape graph reuse is sparse or output exactness fails, close the route quickly.
7. After GPU forward work, return to Node observation/game-step structural cost.
8. Do not run 50k until the remaining gap to the 20x target is materially reduced.

## Source checkpoint
- checkpointKind: ppo_self_play
- schemaVersion: 1
- trainer seed: 7
- updateCount: 2
- episodeCount: 2
- next game seed: 9
- SHA256: 5caa54ac43386e2623a7c2eb6fdbd74e272277963e5030e157f20ffdcce63f72

This file is the persistent handoff for PPO speed work.
Update it whenever the verified fastest path, failed routes, or next experiment changes.


## V6-H inference prefix buffer — CLOSED
- Candidate combined with V6-G: 68059b75999cc45831311a53413891f3d6f86b05
- Baseline: V6-G 45d13c43a36de3e0e3e0b5d79f8669c1bfac8e01
- Workload: 8 env x 500 decisions = 4,000 decisions.
- retention bytes/counters/CPU RNG/CUDA RNG remained exact.
- semanticExact=false; modelExact=false; optimizerExact=false.
- semanticDiffKeys: update, diagnostics.
- batchAct 5,512.16 -> 5,532.09 ms: no isolated structural gain.
- Conclusion: full-shape inference prefix-buffer rewriting still changes floating-point
  execution enough to violate bit-exactness. Do not adopt or extend this route.

## V6-I persistent Action restore buffers — 4k verified result
- Branch: experiment/ppo-fast-batch-v6-action-restore-workspace
- Candidate: fdb0706627cd8cc6ce2837ae2d83f59fabda58de
- Baseline: V6-G 45d13c43a36de3e0e3e0b5d79f8669c1bfac8e01
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- Raw retention: 340,830,768 bytes on both paths.
- External wall 26.724 -> 21.361 sec (1.2510x paired diagnostic).
- Internal total 25.193 -> 19.844 sec.
- Rollout 11.891 -> 11.525 sec.
- Replay 4.871 -> 4.713 sec.
- batchAct 5,741.44 -> 5,288.71 ms (~7.9% reduction).
- The candidate preserves the dense Action shape/value/stride and only reuses
  the zeroed dense-restore buffer plus the GPU index buffer.
- V6-I is the current fastest exact-verified baseline.

## Next steps after V6-I
1. Keep V6-I as the current baseline.
2. Before CUDA Graph implementation, measure exact act input-shape reuse frequency.
3. Do not introduce new bucket/padded shapes solely to increase graph reuse.
4. If exact-shape reuse is too low, close CUDA Graph quickly and return to Node-side
   observation/game-step structural cost.
5. Keep sampling, RNG consumption, finite checks, host-scalar behavior, model input
   shape/value/stride and PPO math unchanged.
6. Do not run 50k until the remaining gap to the 20x target is materially reduced.


## V6-K hot exact-shape CUDA Graph — 4k verified result
- Branch: experiment/ppo-fast-batch-v6k-cuda-graph-hot-clean
- Candidate: 219c8bd00b109b77a5f5a1cf37715cdfa36ad157
- Baseline: V6-I fdb0706627cd8cc6ce2837ae2d83f59fabda58de
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- CUDA Graph policy: exact signature only; minHits=3; maxEntries=12.
- Seen exact signatures: 42.
- Captured graphs: 12.
- Graph replays: 232 / 500 actBatch calls (46.4%).
- Fallback forwards: 268 / 500.
- batchAct: 5,737.97 -> 5,350.23 ms (~6.8% reduction).
- Replay: 4,815.17 -> 4,712.69 ms.
- Rollout: 11.916 -> 11.557 sec.
- Internal total: 24.865 -> 19.914 sec.
- External wall: 26.398 -> 21.419 sec (1.2325x paired diagnostic).
- Graph capture preserves model input information/shape/value/stride and keeps
  sampling/RNG/finite checks/log-prob/host-scalar work outside the graph.
- V6-K is the current fastest exact-verified baseline.

## Next steps after V6-K
1. Keep V6-K as the current baseline.
2. The prior shape audit found 32 repeated exact signatures and 91.6% exact reuse,
   but maxEntries=12 yielded only 46.4% graph replay coverage.
3. Test a representation-identical maxEntries=32 variant once.
4. Keep minHits=3 and all graph semantics unchanged; only expand cache capacity.
5. Measure capture count, replay coverage, batchAct, wall, and GPU stability.
6. If larger cache does not materially improve batchAct or causes memory/capture overhead,
   keep V6-K and close graph-cache scaling.
7. Do not change observation/action information, tensor shapes, PPO math, or RNG semantics.


## V6-L CUDA Graph cache 32 — 4k verified result
- Branch: experiment/ppo-fast-batch-v6l-cuda-graph-32
- Candidate: c0aeb13678c497fd2a41168ec9c99fc087d587da
- Baseline: V6-K 219c8bd00b109b77a5f5a1cf37715cdfa36ad157
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- CUDA Graph minHits remained 3; only maxEntries changed 12 -> 32.
- Captured graphs: 12 -> 29.
- Graph replays: 232 -> 426 / 500 actBatch calls (46.4% -> 85.2%).
- Fallback forwards: 268 -> 74.
- batchAct: 6,019.23 -> 5,536.70 ms (~8.0% reduction).
- Replay: 5,148.87 -> 4,867.77 ms.
- Rollout: 12.397 -> 12.194 sec.
- Internal total: 26.120 -> 20.898 sec.
- External wall: 27.705 -> 22.477 sec (1.2326x paired diagnostic).
- Model input information/shape/value/stride and PPO/RNG semantics remain unchanged.
- V6-L is the current fastest exact-verified baseline.

## Next steps after V6-L
1. Keep V6-L as the current baseline.
2. Close further CUDA Graph cache-capacity scaling for now; 85.2% of act calls already replay graphs.
3. Target replay-side H2D allocation/copy overhead next.
4. Preserve retained sample order, tensor shape/value/stride, gradient accumulation order,
   PPO math, RNG state, and checkpoint schema exactly.
5. Prefer persistent workspace/buffer reuse over changing replay chunk semantics.
6. If replay H2D reuse changes model/optimizer/gradient exactness, close the route immediately.
7. Do not reduce observation/action information or skip decisions.
8. Do not run 50k until the remaining structural gap is materially reduced.


## V6-M persistent replay H2D workspace — 4k verified result
- Branch: experiment/ppo-fast-batch-v6m-replay-h2d-workspace
- Candidate: bb1e41a59cc2170efd21ac0f9a503819e6b4d8a8
- Baseline: V6-L c0aeb13678c497fd2a41168ec9c99fc087d587da
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- CUDA Graph behavior unchanged: 29 captures, 426 replays, 74 fallbacks.
- Replay H2D workspace reuses GPU destination buffers for replay observation tensors,
  sparse Action indices/values, dense Action restore, targets, old log-probabilities,
  advantages and returns.
- Replay stage: 4,991.27 -> 4,834.32 ms (~3.1% reduction).
- Internal replay: 5.012 -> 4.855 sec.
- External wall: 27.032 -> 21.770 sec (1.2417x paired diagnostic; do not attribute
  the whole wall difference to replay H2D workspace).
- batchAct changed 5,803.64 -> 5,529.41 ms even though act path is unchanged;
  treat that difference as run variability, not as V6-M's isolated effect.
- V6-M is the current fastest exact-verified baseline.

## Next steps after V6-M
1. Keep V6-M as the current baseline.
2. Inspect replay H2D transfer count: workspace removes allocations but still performs
   many per-tensor CPU->GPU copy_ operations each replay chunk.
3. Explore grouped replay H2D only if tensor shape/value/stride, sample order, replay
   chunk size, forward/backward order and gradient accumulation remain exactly unchanged.
4. Do not reuse the closed V6-J logical-row restore-buffer route.
5. Reject grouped replay H2D immediately if model/optimizer/gradient/RNG exactness fails.
6. Do not reduce observation/action information or skip decisions.
7. Do not run 50k until the remaining structural gap is materially reduced.


## Replay phase profile before V6-N
- V6-M diagnostic workload: 8 env x 125 decisions = 1,000 decisions.
- Top-level retained replay profile:
  - accumulate_total: 1,152.12 ms (60.01%)
  - decode_records: 554.16 ms (28.86%)
  - prepare_h2d: 130.49 ms (6.80%)
  - combine_cpu: 75.80 ms (3.95%)
  - scalar_h2d: 7.43 ms (0.39%)
- decode_records detail:
  - integrity_hash: 473.38 ms
  - decode_views: 75.82 ms
  - selected_actions: 0.32 ms
- The replay-time SHA256 recheck alone accounted for about 85.4% of decode_records
  and about 24.7% of the profiled replay total.

## V6-N immutable verified raw retention — 4k verified result
- Branch: experiment/ppo-fast-batch-v6n-immutable-raw-retention
- Candidate: 6b4052b5f9c9125568166c15dc8662fa34b7018a
- Baseline: V6-M bb1e41a59cc2170efd21ac0f9a503819e6b4d8a8
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- CUDA Graph behavior exact: 29 captures, 426 replays, 74 fallbacks.
- Raw retention bytes: 340,830,768 on both paths.
- Raw retained payload is copied to immutable Python bytes in the retention worker,
  hashed once after that immutable copy is created, and replay checks length/type while
  skipping the redundant second SHA256 pass only for verified immutable raw records.
- Deflate/non-immutable retention paths keep replay-time SHA256 verification.
- Replay stage: 4,471.66 -> 3,518.28 ms (~21.3% reduction).
- Internal replay: 4.490 -> 3.603 sec.
- Internal total: 23.129 -> 18.608 sec.
- External wall: 24.636 -> 20.127 sec (1.2241x paired diagnostic).
- Observation encode/game step/batchAct differences are treated as run variability;
  they are not attributed to V6-N.
- V6-N is the current fastest exact-verified baseline.

## Next steps after V6-N
1. Keep V6-N as the current baseline.
2. Re-profile V6-N retained replay after removing the redundant replay SHA256 pass.
3. Target the largest newly measured remaining replay component only.
4. If accumulate_total remains dominant, profile forward/objective/backward internally
   before changing update math or kernel structure.
5. Keep sample order, replay chunk size, tensor shape/value/stride, gradient accumulation
   order, PPO math, RNG state and checkpoint schema unchanged.
6. Do not reduce observation/action information or skip decisions.
7. Do not run 50k until the remaining structural gap is materially reduced.


## V6-N replay compute profile after SHA removal
- Diagnostic branch: experiment/ppo-fast-batch-v6n-replay-profile
- Diagnostic commit: 6ad62c02bc05d864368a2f3f07b3164eb08a226e
- Workload: 8 env x 125 decisions = 1,000 decisions.
- Top-level replay profile:
  - accumulate_total: 1,095.96 ms (79.01%)
  - prepare_h2d: 124.31 ms (8.96%)
  - decode_records: 78.81 ms (5.68%)
  - combine_cpu: 75.42 ms (5.44%)
  - scalar_h2d: 12.60 ms (0.91%)
- Immutable raw-retention integrity validation: 0.20 ms total.
- accumulate_total breakdown:
  - backward: 458.97 ms (41.88% of accumulate_total)
  - model forward: 285.96 ms (26.09%)
    - state encoder: 214.48 ms
    - action encoder: 13.12 ms
    - score head: 26.06 ms
    - value head: 14.72 ms
  - objective: 235.28 ms (21.47%)
  - input checks: 31.70 ms
  - finite checks: 31.68 ms
  - selected-action checks: 15.04 ms
- Conclusion: transport/integrity overhead is no longer dominant. The remaining replay
  bottleneck is actual training compute, especially backward + state encoding + objective.

## Next steps after V6-N replay compute profile
1. Keep V6-N as the production baseline; profile-only commits remain diagnostic.
2. Before changing training math or kernels, measure exact replay input-shape reuse.
3. Consider exact-shape CUDA Graph / graphed-callable training only if replay signatures
   repeat enough to amortize capture cost.
4. Do not alter replay chunk size, sample order, tensor shape/value/stride, gradient
   accumulation order, PPO objective, finite checks, RNG state or checkpoint schema.
5. Do not reduce observation/action information.


## V6-N exact replay-shape reuse audit
- Diagnostic branch: experiment/ppo-fast-batch-v6n-replay-shape-audit
- Diagnostic commit: 6143dd725d8a13aed12a50f4d1966a138ce10527
- Workload: 8 env x 500 decisions = 4,000 decisions.
- Replay chunks: 125.
- Unique exact signatures: 33.
- Repeated signatures: 23; singleton signatures: 10.
- Calls on repeated signatures: 115 / 125.
- Reused calls after the first occurrence: 92 / 125 = 73.6%.
- Top exact-signature counts:
  23, 9, 8, 7, 7, 7, 6, 6, 5, 5, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2.
- Signature includes tensor shape/stride/dtype plus _nonempty and _validPrefixCount,
  and all replay tensors including Action/mask/targets/old log-prob/advantages/returns.
- Conclusion: exact replay-shape reuse is materially high enough to justify a
  controlled training CUDA Graph feasibility probe.
- This audit is diagnostic only; V6-N remains the production baseline.

## Next steps after replay-shape audit
1. Keep V6-N as the production baseline.
2. Do not directly graph the production backward path.
3. First build an isolated GPU microprobe on cloned trainer/model state for one hot exact
   replay signature.
4. Compare eager vs graphed-callable outputs and gradients bit-exactly, including
   per-parameter grad None/not-None state and gradient hash.
5. Preserve model input information/shape/value/stride, replay chunk size, sample order,
   PPO objective, RNG, gradient accumulation order and checkpoint schema.
6. If the microprobe cannot preserve exact gradients, close training CUDA Graph immediately.


## V6-O replay graphed forward — 4k gate CLOSED
- Branch: experiment/ppo-fast-batch-v6o-replay-graphed-forward
- Candidate: 1596d823f422a59caa6e3fa9eb6f2b6c828ac7f0
- Baseline: V6-N ea68bbd776c156e3ee0d9c484501d3c3327a35bc
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention exact.
- update equivalence exact, including gradientHash.
- Replay Graph: minHits=3, maxEntries=4, seenSignatures=33,
  capturedGraphs=4, graphReplays=30, fallbackForwards=95.
- Replay stage: 3,953.93 -> 3,982.15 ms (~0.7% regression).
- The prior 1k gate improvement did not reproduce at 4k.
- PyTorch AccumulateGrad stream-mismatch warning persisted.
- External wall 25.772 -> 20.351 sec is not attributed to V6-O because the targeted
  replay stage did not improve.
- Conclusion: exact but no isolated replay speed gain. Do not promote V6-O.
- V6-N remains the current production baseline.

## Next steps after V6-O
1. Keep V6-N as baseline.
2. Close replay CUDA Graph forward for now.
3. Return to Node-side structural cost: game step and observation encoding are the
   next sizeable non-training stages.
4. Profile before changing logic; preserve the same game state, observation values,
   action values, decision cadence, RNG, PPO math and checkpoint schema.
5. Do not reduce information observed by the policy and do not skip decisions.
6. Do not run 50k until another material structural gain is verified.


## V6-N Observation encoder phase profile
- Diagnostic branch: experiment/ppo-fast-batch-v6n-observation-profile
- Diagnostic commit: fac9d7ded5217b847ab7d72b85d1f56782441013
- Workload: 8 env x 125 decisions = 1,000 decisions.
- Top-level Observation encode: 676.53 ms.
- Instrumented Observation encode: 670.291 ms.
- Stage split:
  - map: 275.807 ms (41.15%)
  - bases: 194.694 ms (29.05%)
  - units: 132.339 ms (19.74%)
  - setup: 22.240 ms (3.32%)
  - strategic: 15.753 ms (2.35%)
  - positions: 14.600 ms (2.18%)
  - teamsGlobal: 8.022 ms (1.20%)
  - constructions: 4.687 ms (0.70%)
  - compaction: 2.149 ms (0.32%)
- Node profile on the same 1k diagnostic:
  - game step: 709.91 ms
  - Observation encode: 676.53 ms
  - Action encode: 238.33 ms
  - batchAct: 1,912.54 ms
  - replay: 1,381.91 ms
- Conclusion: map + bases account for about 70.2% of Observation encode.
- Next: optimize map first with exact-value-preserving static row templates; keep bases separate.


## V6-P static map row template cache — 4k verified result
- Branch: experiment/ppo-fast-batch-v6n-map-template-cache
- Candidate: 1a67e421d70293360275ee88ccbeadfa9c076e1b
- Baseline: V6-N 93a909a (same production code as immutable-retention baseline plus docs).
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- Change only caches the immutable static map row template once per episode/team width;
  each decision still creates a fresh output row and writes dynamic bridge/obstacle/base
  ownership/team one-hot values exactly as before.
- Observation encode: 2,368.23 -> 1,771.10 ms (~25.2% reduction).
- Game step: 2,659.46 -> 2,736.88 ms (run variability / unrelated to candidate).
- batchAct: 5,345.57 -> 4,953.24 ms and replay 3,952.23 -> 3,782.24 ms;
  these paths were unchanged and are treated as run variability.
- External wall: 25.274 -> 19.787 sec (1.2773x paired diagnostic; do not attribute
  the full wall difference to map-template caching).
- V6-P is the current fastest exact-verified baseline.

## Next steps after V6-P
1. Keep V6-P as the current baseline.
2. Observation profile showed bases as the next largest encoder component (~29.0% before V6-P).
3. Inspect encodeBase for repeat work that can be cached without changing row values/order/width.
4. Preserve dynamic ownership, slot occupancy, cooldowns and all team/base reference vectors.
5. Keep game step as a separate later bottleneck; do not mix base encoding and game-step changes.
6. Do not reduce observed information or skip decisions.
7. Do not run 50k until another material structural gain is verified.


## V6-Q static base row template cache — 4k verified result
- Branch: experiment/ppo-fast-batch-v6q-base-template-cache
- Candidate: debacad53085bd106de1c60cc8cdd94464005375
- Baseline: V6-P 50abdd3
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters all exact.
- retention structure/raw/stored bytes/drain all exact.
- Observation encode: 1,709.77 -> 1,087.11 ms (~36.4% reduction).
- Static base rows cache centroid/slot-layout zero templates and overwrite only
  dynamic base ownership, occupation priority and occupant team/type fields.
- Policy information, base-slot coverage, row width/order and feature values are unchanged.
- External wall: 24.451 -> 19.100 sec (1.2802x paired diagnostic; do not attribute
  the whole wall difference to base templating).
- V6-Q is the current fastest exact-verified baseline.

## Next steps after V6-Q
1. Keep V6-Q as the current baseline.
2. Observation encode is no longer the dominant Node-side stage.
3. Profile game-step internals next; current 4k game step is 2,747.58 ms vs
   Observation encode 1,087.11 ms.
4. Preserve exact game state transitions, action cadence, RNG, adjudication,
   rewards and all PPO/trajectory semantics.
5. Do not simplify game rules or skip state updates.


## V6-Q neutral battle / attack-distance profiling
- Diagnostic commits:
  - game-step profile: 896ab046fbcd83c816944d0300bb6dadf12bd995
  - battle-stage profile: 0111437a686fe04531a14fdd84e765daff627f07
  - battle event-build profile: 3671d40c66781f6bbdc9343dc7c2b76535f9ab56
  - neutral attack detail profile: 2ec0fa29aee1b177e1780fa654978d5251b0e1ce
- 1k paired diagnostics remained allExact=true.
- Game-step internal profile: apply 468.89 ms, enumerate 325.07 ms, runtime clone 31.65 ms.
- resolve_battle hotspot: 259.39 ms / 24 calls in the first profile.
- Battle-stage split localized nearly all resolve_battle time to event_build.
- Event-build split localized the hotspot to neutral attack candidate generation.
- Neutral candidate detail: 360 getAttackCandidates calls over 24 battles (=15 neutral units/battle).
- neutral candidate search: 212.72 ms total.
- attackRangeDistance: 188.73 ms / 516 calls (~88.7% of neutral candidate search).
- attackPostProcessing/sort: only 0.63 ms total.
- Conclusion: sorting/first-target-only optimization is not the main target. Pairwise road-distance search is the dominant neutral-battle cost.

## V6-R CLOSED — target-based attack distance lookup
- Branch: experiment/ppo-fast-batch-v6r-attack-distance-lookup-clean
- Commit: e9e3a1cb2bd1593a38972a34d11a5b5ae7af4a44
- Change: cache createRoadAttackDistanceLookup() by target unit ID and reuse across attackers.
- Local TypeScript check: PASS.
- Related tests: 45/45 PASS.
- Kaggle 1k paired against V6-Q: allExact=true.
- model/optimizer/CPU RNG/CUDA RNG/counters/retention/semantics: exact.
- game-step: 721.78 -> 1135.69 ms.
- game-step speed ratio: 0.6355x (candidate slower by ~57%).
- Candidate wall was lower, but batchAct/replay noise moved strongly in the opposite direction, so wall is not accepted as evidence of improvement.
- Interpretation: full-graph target lookup construction is too expensive for this workload.
- Decision: V6-R rejected and closed. Do not run 4k or 50k for this route.
- Next candidate: keep pairwise BFS semantics and optimize its queue mechanics only (remove Array.shift without changing visitation order or returned distances).


## V6-S CLOSED — pairwise attack-distance BFS queue indexing
- Branch: experiment/ppo-fast-batch-v6s-road-distance-queue-index
- Commit: 5a4dc500360d6f0d1f70f12c753b63971fda0b75
- Change: replace Array.shift() with index-based FIFO traversal only inside getRoadAttackDistance().
- Local TypeScript check: PASS.
- Related battle/heuristic tests: 45/45 PASS.
- Kaggle 1k paired against V6-Q: allExact=true.
- model/optimizer/CPU RNG/CUDA RNG/counters/retention/semantics: exact.
- Game step: 645.06 -> 638.57 ms (~1.0102x).
- Improvement is only ~6.49 ms / 1k decisions and is below the observed Kaggle timing noise floor.
- External wall changed strongly, but batchAct/replay timing also moved strongly, so wall is not evidence for this two-line queue change.
- Decision: exact but effect insufficient. Close V6-S; do not spend a 4k or 50k run on this route.

## Next step after V6-S closure
1. Keep V6-Q as the exact production baseline.
2. Attack-range distance remains the measured neutral-battle hotspot: 188.73 ms / 516 calls in the 1k diagnostic.
3. Test a dedicated bounded BFS only for attack-candidate range legality.
4. The bounded BFS must return the exact shortest integer distance when the target is reachable within the attacker's exact range.
5. If no target is reachable within that range, it may return Infinity because the existing caller immediately rejects distance > range.
6. Do not replace unrestricted getRoadAttackDistance for other callers.
7. Cache only finite exact distances into AttackEnumerationContext; do not cache bounded Infinity into the unrestricted pair-distance cache.
8. Gate at 1k exact before any 4k run; 50k remains NOT RUN.


## V6-T bounded attack-distance — 1k gate PASS
- Branch: experiment/ppo-fast-batch-v6t-bounded-attack-distance
- Commit: e34aa09e20989d7f5ef8022ebc6a885f773bc8d2
- Baseline: V6-Q fdd9de2c65eec2a95be3a387d30eb36973fe4502
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Local TypeScript static check: PASS.
- Related battle/heuristic tests: 45/45 PASS.
- Change: add a dedicated bounded attack-path BFS for attack-candidate legality only.
- The bounded BFS returns the exact shortest integer distance when reachable within the attacker's exact range; otherwise it returns Infinity.
- Unrestricted getRoadAttackDistance remains unchanged for other callers.
- Only finite exact bounded distances are cached into AttackEnumerationContext; bounded Infinity is not cached as an unrestricted pair distance.
- Game step: 673.14 -> 459.07 ms = 1.4663x speedup (~31.8% reduction).
- External wall: 15.647 -> 10.019 sec, but batchAct/replay also moved materially and wall is not used as isolated evidence.
- Decision: 1k gate passed strongly. Proceed to 4k paired exactness/performance gate before promotion.
- 50k remains NOT RUN.


## V6-T bounded attack-distance — 4k verified result
- Branch: experiment/ppo-fast-batch-v6t-bounded-attack-distance
- Implementation commit: e34aa09e20989d7f5ef8022ebc6a885f773bc8d2
- Baseline: V6-Q fdd9de2c65eec2a95be3a387d30eb36973fe4502
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- CUDA Graph behavior exact: 29 captures, 426 replays, 74 fallbacks.
- Game step: 2,666.43 -> 1,986.34 ms = 1.3424x speedup (~25.5% reduction).
- Observation encode: 1,094.09 -> 1,136.15 ms and Action encode: 845.44 -> 855.51 ms;
  these paths are unchanged and differences are treated as run variability.
- batchAct: 5,402.27 -> 5,193.18 ms; replay: 3,678.81 -> 3,593.25 ms;
  these paths are unchanged and differences are treated as run variability.
- External wall: 23.790 -> 18.511 sec; full wall difference is not attributed solely to V6-T.
- Bounded BFS is used only for attack-candidate range legality.
- It returns the exact shortest distance when reachable within the attacker's exact range;
  otherwise Infinity is sufficient because the existing caller immediately rejects distance > range.
- Unrestricted getRoadAttackDistance remains unchanged for other callers.
- Only finite exact bounded distances are stored in the unrestricted pair-distance cache.
- V6-T is promoted as the current fastest exact-verified baseline.
- 50k remains NOT RUN.

## Next steps after V6-T
1. Keep V6-T as the production baseline.
2. Re-profile game-step internals on V6-T before further game-rule-path changes.
3. Confirm how much neutral attack-distance cost remains after bounded BFS.
4. If game-step is no longer the dominant actionable Node-side stage, compare remaining
   batchAct/replay/encoding structural costs before choosing the next route.
5. Preserve exact legal-action sets/order, game transitions, RNG, rewards, trajectory semantics,
   PPO math, replay chunk size, environment count 8 and checkpoint schema.
6. Do not run 50k until another material structural gap is removed or the remaining gap is
   sufficiently characterized.


## V6-T game-step re-profile after bounded attack-distance
- Diagnostic branch: experiment/ppo-fast-batch-v6t-game-step-profile
- Diagnostic commit: 02e361908751fc914971ecb699864d6aaba797ab
- Baseline: V6-T e34aa09e20989d7f5ef8022ebc6a885f773bc8d2
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Baseline game step: 484.92 ms; instrumented candidate: 492.72 ms.
- Instrumentation overhead is small and candidate remains exact.
- Internal game-step split:
  - apply: 240.71 ms
  - enumerate: 193.86 ms
  - runtime clone: 26.32 ms
  - unaccounted: 31.00 ms
- Apply by phase:
  - movement_input: 152.84 ms
  - attack_input: 83.87 ms
  - strategist_action_input: 2.03 ms
  - strategist_action_resolution: 1.96 ms
- Apply by kind hotspots:
  - movement: 93.80 ms / 344 calls
  - resolve_battle: 82.25 ms / 24 calls
  - submit_team_production: 39.48 ms / 32 calls
  - submit_movement: 18.99 ms / 112 calls
- Enumeration by phase:
  - movement_input: 131.12 ms / 520 calls
  - attack_input: 61.36 ms / 408 calls
  - strategist_action_input: 1.17 ms
  - strategist_action_resolution: 0.20 ms
- Conclusion: after V6-T, movement apply + movement enumeration are the largest remaining
  measured game-step costs. Further battle work is no longer the first Node-side target.

## Next steps after V6-T game-step re-profile
1. Keep V6-T as production baseline.
2. Profile movement enumeration and movement apply separately before changing movement logic.
3. Preserve legal movement set/order, exact destination values, retreat semantics, collision rules,
   bridges/bases/roads, RNG, state transitions and trajectory semantics.
4. Do not combine movement enumeration and movement-apply optimizations in one candidate.


## V6-T movement internal profile
- Diagnostic branch: experiment/ppo-fast-batch-v6t-movement-profile
- Diagnostic commits:
  - 4cdc9ce33853a6a0b3ac4c33d8e965f30b067cf8
  - c0947018d04cc85b15d62aaad476ce9d2252e8e7
- Baseline: V6-T e34aa09e20989d7f5ef8022ebc6a885f773bc8d2
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Movement enumeration:
  - movementVisibleState: 3.76 ms / 352 calls
  - movementRangePathSearch: 91.60 ms / 352 calls
- Movement application:
  - movementApplyValidationPathSearch: 44.70 ms / 203 calls
  - movement action apply total: 97.00 ms / 344 movement decisions
- Therefore roughly 46% of measured movement-action application time is path revalidation.
- The same movement destination was already produced by legal-action enumeration from the current
  state immediately before the selected RL action is applied.
- Existing baseline test note: four legacy saved-intent assertions in movement.test.ts fail on the
  unchanged V6-T baseline e34aa09 as well. The movement-profile changes did not introduce those failures.
  Immediate-movement, movement-rotation and profiler tests remain passing.

## Next candidate: V6-U prevalidated RL movement apply
- Do not remove general movement validation.
- Keep UI/general game callers, legacy replay and externally supplied movement intents unchanged.
- Only a movement action selected from the current RlEnvironment legal-action list may use an
  explicit RL-only prevalidated fast path.
- Preserve hidden-water-ninja collision handling, retreat effects, occupancy/state mutation,
  moved-unit bookkeeping, logs, legal action ordering/content, state transitions and all RNG behavior.
- 1k paired exactness gate first; 4k only if the 1k result is exact and materially faster.


## V6-U prevalidated RL movement — 1k gate PASS
- Branch: experiment/ppo-fast-batch-v6u-prevalidated-movement-clean
- Implementation commit: 1469f256ff1e53199ff537e83992d47c1728bc9d
- Baseline: V6-T e34aa09e20989d7f5ef8022ebc6a885f773bc8d2
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Game step: 477.05 -> 423.79 ms = 1.1257x speedup (~11.2% reduction).
- The optimization is gated by an explicit rlPrevalidatedMovement flag.
- General game/UI movement validation remains unchanged.
- Legacy batched movement remains unchanged.
- Hidden-water-ninja collision handling still executes before the skipped path revalidation.
- A new 300-action deterministic state-identical test for the prevalidated RL path passes.
- Existing ninjaWaterMovement 3 failures and legacy saved-intent movement.test 4 failures were reproduced
  on the unchanged V6-T baseline and are not regressions introduced by V6-U.
- External wall and unchanged batchAct/replay differences are treated as run variability.
- Decision: proceed to 4k paired exactness/performance gate.
- 50k remains NOT RUN.


## V6-U prevalidated RL movement — 4k verified result
- Branch: experiment/ppo-fast-batch-v6u-prevalidated-movement-clean
- Implementation commit: 1469f256ff1e53199ff537e83992d47c1728bc9d
- Baseline: V6-T e34aa09e20989d7f5ef8022ebc6a885f773bc8d2
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- CUDA Graph behavior exact: 29 captures, 426 replays, 74 fallbacks.
- Game step: 1,976.72 -> 1,837.08 ms = 1.0760x speedup (~7.1% reduction).
- Observation encode: 1,095.89 -> 1,095.64 ms and Action encode: 850.06 -> 846.02 ms;
  these paths are unchanged and effectively identical.
- Replay: 3,616.70 -> 3,621.05 ms; unchanged path and effectively identical.
- batchAct: 5,418.96 -> 4,963.28 ms and finishUpdate changed materially between runs;
  these unchanged paths are treated as run variability, not V6-U attribution.
- External wall: 23.116 -> 18.005 sec; full wall difference is not attributed solely to V6-U.
- The optimization only skips duplicate path revalidation for a movement action selected from the
  current RlEnvironment legal-action list, under an explicit RL-only flag.
- General/UI movement validation, legacy replay semantics and externally supplied movement intents remain unchanged.
- Hidden-water-ninja collision handling remains before the skipped validation.
- V6-U is promoted as the current fastest exact-verified baseline.
- 50k remains NOT RUN.

## Next steps after V6-U
1. Re-profile game-step/movement internals on V6-U before changing another game-rule path.
2. Movement legal enumeration path search remains a likely Node-side target, but profile the
   internal getMovementPaths stages first.
3. Do not combine enumeration-path optimization with another apply-path change.
4. Preserve legal movement set/order, exact destination values, visibility, retreat, bridge/base/road
   semantics, state transitions, RNG and trajectory meaning.


## V6-U prevalidated RL movement — 4k verified result
- Branch: experiment/ppo-fast-batch-v6u-prevalidated-movement-clean
- Implementation commit: 1469f256ff1e53199ff537e83992d47c1728bc9d
- Baseline: V6-T e34aa09e20989d7f5ef8022ebc6a885f773bc8d2
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- CUDA Graph behavior exact: 29 captures, 426 replays, 74 fallbacks.
- Game step: 1,965.85 -> 1,829.77 ms = 1.07437x speedup (~6.92% reduction).
- Observation encode: 1,100.86 -> 1,106.18 ms and Action encode: 847.00 -> 834.89 ms;
  unchanged-path differences are treated as run variability.
- batchAct: 5,329.80 -> 5,086.86 ms; replay: 3,613.71 -> 3,803.79 ms;
  these unchanged-path differences are treated as run variability.
- External wall: 22.467 -> 18.232 sec, but full wall difference is not attributed solely to V6-U.
- The general movement validator remains unchanged.
- Only RL movement actions selected from the current legal-action list use the explicit
  rlPrevalidatedMovement fast path.
- Hidden-water collision handling remains before the skipped path revalidation.
- V6-U is promoted as the current fastest exact-verified baseline.
- 50k remains NOT RUN.

## Next steps after V6-U
1. Keep V6-U as production baseline.
2. Re-profile movement enumeration/path-search internals on V6-U.
3. The previous V6-T movement profile showed movementRangePathSearch at 91.60 ms / 352 calls per 1k,
   much larger than movementVisibleState at 3.76 ms.
4. Target redundant representation/allocation/topology work inside getMovementPaths without changing
   candidate destinations, ordering, hidden-information behavior, retreat rules, occupancy rules,
   bridges/bases/roads, or movement semantics.
5. Use a diagnostic-only branch first; do not combine profiling instrumentation with the next optimization.
6. Keep 50k deferred until another material structural gap is removed or remaining costs are characterized.


## V6-U movement-path detail profile
- Diagnostic branch: experiment/ppo-fast-batch-v6u-movement-path-profile
- Diagnostic branch tip: 2463083 (profiling only).
- Equivalent V6-T detail probe commit used for the isolated 1k split: 28578e1207d9d6cb5695411064721bae4442a2d3.
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true.
- movementRangePathSearch: 94.68 ms / 352 calls.
- Dominant internal stage: movementPathLeaveBaseExpansion = 58.86 ms / 219 calls (~62% of path-search time).
- Other measured enumeration costs:
  - movementPathPositionForTile: 8.57 ms / 1,762
  - movementPathBaseConnectivity: 6.78 ms / 262
  - movementPathGroundConnectivity: 3.19 ms / 628
  - movementPathBaseLookup: 1.45 ms / 2,024
  - movementPathBaseDestinationLegal: 1.40 ms / 591
  - movementPathEmptyBasePositions: 0.63 ms / 262
- Code audit: getMovementPaths may enqueue multiple BaseSlot positions for the same base.
- nextGroundPositionsFromBase depends on state, unit and baseId, but not the source slotId.
- Therefore repeated base-slot nodes can recompute the same leave-base destination list inside one path search.

## Next candidate: V6-V per-search leave-base expansion cache
- Cache nextGroundPositionsFromBase results by baseId only within one getMovementPaths invocation.
- Do not share the cache across states or separate legal-enumeration calls.
- Preserve destination values and insertion order exactly.
- Do not change occupancy, visibility, bridge/base/road rules, retreat behavior or RNG.
- Run local tests, then 1k paired exactness gate; 4k only if exact and materially faster.


## V6-V CLOSED — per-search leave-base expansion cache
- Branch: experiment/ppo-fast-batch-v6v-leave-base-cache
- Candidate commit: 815567af070140f95b075308639205b5d6a5d22e
- Baseline: V6-U 321103e
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Game step: 492.94 -> 529.39 ms = 0.9311x (about 7.4% slower).
- Candidate cached nextGroundPositionsFromBase by baseId only within one getMovementPaths call.
- Exactness confirms the cached values/order were semantically safe for this workload.
- Performance result shows the reuse frequency is insufficient to amortize the per-search Map allocation/lookups.
- Decision: CLOSED; do not run 4k and do not retry the same per-search baseId cache with minor parameter changes.
- Keep V6-U as the current fastest exact-verified baseline.
- 50k remains NOT RUN.

## Next direction after V6-V
1. Inspect nextGroundPositionsFromBase itself rather than caching its final result.
2. Separate static base-boundary/topology work from dynamic occupancy/construction checks only if exact ordering
   and all current movement rules can be preserved.
3. Avoid adding a cache unless measured reuse exists.
4. Prefer a structural reduction in repeated geometry scans over another memoization variant.


## V6-W hoist base road sections — 1k gate PASS
- Branch: experiment/ppo-fast-batch-v6w-hoist-base-sections-clean
- Implementation commit: 6904e598fc7be45d910486211580ed33a7427be4
- Baseline: V6-U 835924a
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Game step: 455.12 -> 422.85 ms = 1.0763x speedup (~7.1% reduction).
- Prior leave-base detail profile showed:
  - movementPathLeaveBaseExpansion: 75.52 ms / 219 calls
  - movementPathLeaveBaseConnectivity: 47.49 ms / 1,733 calls
  - movementPathLeaveBasePositionForTile: 13.12 ms / 4,380 calls
  - movementPathLeaveBaseBaseCellCheck: 4.14 ms / 7,008 calls
- Optimization hoists getBaseConnectedRoadSectionIds(state, baseId) once per
  nextGroundPositionsFromBase invocation and preserves the original per-tile roadSectionId membership test.
- Destination values, insertion order, occupancy checks, bridge/base/road rules, visibility,
  retreat behavior and RNG are unchanged.
- Decision: proceed to 4k paired exactness/performance gate.
- 50k remains NOT RUN.


## V6-W hoist base road sections — 4k verified result
- Branch: experiment/ppo-fast-batch-v6w-hoist-base-sections-clean
- Implementation commit: 6904e598fc7be45d910486211580ed33a7427be4
- Baseline: V6-U 835924a
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- CUDA Graph behavior exact: 29 captures, 426 replays, 74 fallbacks.
- Game step: 1,928.24 -> 1,849.77 ms = 1.04242x speedup (~4.07% reduction).
- Observation encode: 1,147.94 -> 1,261.89 ms and Action encode: 881.43 -> 924.26 ms;
  unchanged-path differences are treated as run variability.
- batchAct: 5,703.27 -> 5,565.35 ms; replay: 4,072.61 -> 3,894.64 ms;
  unchanged-path differences are treated as run variability.
- External wall: 24.291 -> 19.360 sec, but full wall difference is not attributed solely to V6-W.
- V6-W hoists the invariant base connected-road-section list once per leave-base expansion.
- Per-tile roadSectionId membership, candidate values, insertion order, visibility, occupancy,
  bridge/base/road rules, retreat behavior and RNG remain unchanged.
- V6-W is promoted as the current fastest exact-verified baseline.
- 50k remains NOT RUN.

## Next steps after V6-W
1. Keep V6-W as production baseline.
2. Re-profile game-step/movement enumeration on V6-W before another optimization.
3. Confirm the residual leave-base expansion cost after hoisting base connected sections.
4. Compare remaining movement enumeration, attack enumeration and battle resolution costs.
5. Do not infer the next target from pre-V6-W profiles alone.
6. Keep 50k deferred until another material structural gap is removed or remaining costs are characterized.


## V6-W game-step + movement-path re-profile
- Diagnostic branch: experiment/ppo-fast-batch-v6w-game-step-profile
- Diagnostic commit: 4121a898c1e9de4c3b1a79e9df7b873c59c23774
- Baseline: V6-W 66f47f5a5ed6059ee085a90d5880d7cdb0e7d744
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Baseline game step without detailed instrumentation: 415.29 ms.
- Instrumented candidate game step: 472.79 ms; this slowdown is profiler overhead and is not a performance regression.
- Instrumented internal split:
  - apply: 207.79 ms
  - enumerate: 203.39 ms
  - runtime clone: 26.76 ms
  - unaccounted: 33.26 ms
- Apply hotspots:
  - movement_input: 120.28 ms
  - attack_input: 83.88 ms
  - movement action: 59.32 ms / 344 calls
  - resolve_battle: 82.30 ms / 24 calls
  - submit_team_production: 42.81 ms / 32 calls
- Enumeration hotspots:
  - movement_input: 122.77 ms
  - attack_input: 79.07 ms
- Movement enumeration internals:
  - movementRangePathSearch: 73.79 ms / 352 calls
  - movementPathLeaveBaseExpansion: 27.22 ms / 219 calls
  - movementPathPositionForTile: 10.56 ms / 1,762 calls
  - movementPathBaseConnectivity: 9.79 ms / 262 calls
  - movementPathGroundConnectivity: 4.66 ms / 628 calls
  - movementVisibleState: 3.82 ms / 352 calls
- Attack enumeration internals:
  - attackTargetSearch: 58.68 ms / 288 calls
  - attackRangeDistance: 20.00 ms / 264 calls
  - attackAcrossBaseBlocking: 16.59 ms / 4,017 calls
  - attackUnitCoordinateBaseSearch: 8.86 ms / 24 calls
- After V6-W, movement path search remains the largest measured enumeration substage.
- Residual leave-base expansion remains 27.22 ms, so profile its post-V6-W internals before another optimization.
- 50k remains NOT RUN.

## Next diagnostic after V6-W
1. Profile current V6-W nextGroundPositionsFromBase internals without reverting the V6-W road-section hoist.
2. Measure base lookup, connected-road-section preparation, base-cell checks, positionForTile,
   road-section lookup and section-membership checks separately.
3. Do not optimize until the post-V6-W residual leave-base cost is localized.


## V6-W leave-base detail profile
- Diagnostic branch: experiment/ppo-fast-batch-v6w-leave-base-detail-profile
- Diagnostic commit: 6947024e27eecc43a2ed12a10708f486f1273912
- Baseline: V6-W 2d1078d5798fa3f259426db6645abc5855d99593
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Detailed instrumentation adds material profiler overhead, so baseline/candidate game-step wall is not used as a speed comparison.
- Residual leave-base expansion breakdown:
  - movementPathLeaveBasePositionForTile: 16.57 ms / 4,380 calls
  - movementPathLeaveBaseConnectedSections: 8.84 ms / 219 calls
  - movementPathLeaveBaseBaseCellCheck: 4.72 ms / 7,008 calls
  - movementPathLeaveBaseRoadSectionLookup: 2.57 ms / 1,733 calls
  - movementPathLeaveBaseDeduplicate: 1.44 ms / 1,733 calls
  - movementPathLeaveBaseSectionMembership: 0.73 ms / 1,733 calls
  - movementPathLeaveBaseBaseLookup: 0.26 ms / 219 calls
  - movementPathLeaveBaseMaterialize: 0.21 ms / 219 calls
- Code audit found positionForTile first calls getTile(state.map.tiles, x, y), then for tile/water
  calls isLegalDestination, which calls getTile again for the same coordinate.
- This duplicate tile lookup changes no information and is a candidate for exact structural removal.

## Next candidate: V6-X known-tile reuse in positionForTile
- Reuse the already fetched Tile only inside positionForTile -> isLegalDestination.
- Keep the exported isLegalDestination behavior unchanged for all existing callers.
- Bridge handling, occupancy, obstacle, water/ninja and terrain checks must remain identical.
- Do not cache across calls or states.
- Run local tests, then 1k paired exactness gate; 4k only if exact and materially faster.
- 50k remains NOT RUN.


## V6-X known-tile reuse — 1k gate PASS
- Branch: experiment/ppo-fast-batch-v6x-known-tile-reuse
- Implementation commit: bc162d6242118242f5ec600c35fe77b10e752a5e
- Baseline: V6-W 9e7357f841f92bd515ef0fbba12074f78cfdceba
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Local TypeScript static check: PASS.
- Relevant profiler/movement tests: 22/22 PASS.
- Change: positionForTile reuses the Tile object it already fetched when running the same
  tile/water legality check, avoiding a duplicate linear getTile scan for that coordinate.
- Exported isLegalDestination behavior and signature are unchanged.
- Bridge, occupancy, obstacle, water/ninja and terrain rules are unchanged.
- Game step: 417.02 -> 407.27 ms = 1.02394x speedup (~2.34% reduction).
- Unchanged batchAct/replay/wall differences are treated as run variability.
- Decision: proceed to a 4k paired exactness/performance gate because the 1k gain is small but
  materially above the previously closed ~1% noise-floor route.
- 50k remains NOT RUN.


## V6-X known-tile reuse — 4k verified result
- Branch: experiment/ppo-fast-batch-v6x-known-tile-reuse
- Implementation commit: bc162d6242118242f5ec600c35fe77b10e752a5e
- Baseline: V6-W 9e7357f841f92bd515ef0fbba12074f78cfdceba
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- CUDA Graph behavior exact: 29 captures, 426 replays, 74 fallbacks.
- Game step: 1,774.76 -> 1,715.63 ms = 1.03447x speedup (~3.33% reduction).
- Observation encode: 1,126.81 -> 1,088.84 ms; Action encode: 858.50 -> 854.66 ms.
- batchAct: 5,292.36 -> 5,059.12 ms; replay: 3,681.35 -> 3,629.67 ms.
  These unchanged-path differences are treated as run variability and are not attributed to V6-X.
- External wall: 22.177 -> 17.790 sec; full wall difference is not attributed solely to V6-X.
- V6-X only reuses the Tile already fetched by positionForTile for its immediately following
  tile/water legality check, removing one duplicate getTile scan.
- Exported isLegalDestination semantics/signature, bridge handling, occupancy, obstacles,
  water/ninja rules, terrain checks, candidate ordering and movement semantics remain unchanged.
- V6-X is promoted as the current fastest exact-verified baseline.
- 50k remains NOT RUN.

## Next steps after V6-X
1. Keep V6-X as production baseline.
2. Re-profile current game-step before choosing V6-Y; do not rely only on V6-W timings.
3. Compare residual movement enumeration, attackTargetSearch and resolve_battle after V6-X.
4. Prefer another measured structural duplicate/repeated scan over a broad cache.
5. Preserve legal action values/order, visibility, movement/attack rules, RNG, trajectory semantics,
   PPO math, 8 environments, replay chunk size 32 and checkpoint schema.


## V6-X game-step + movement-path re-profile
- Diagnostic branch: experiment/ppo-fast-batch-v6x-game-step-profile
- Diagnostic commit: 47208935b8504178b74e0a79b3943f4ee4319f57
- Baseline: V6-X edfaaa4
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Baseline game step without detailed instrumentation: 439.26 ms.
- Instrumented diagnostic game step: 480.84 ms; the slowdown is profiler overhead and is not a regression.
- Instrumented internal split:
  - apply: 213.16 ms
  - enumerate: 204.34 ms
  - runtime clone: 28.34 ms
  - unaccounted: 33.32 ms
- Apply hotspots:
  - movement_input: 118.98 ms
  - attack_input: 90.75 ms
  - movement action: 59.01 ms / 344 calls
  - resolve_battle: 88.95 ms / 24 calls
- Enumeration hotspots:
  - movement_input: 113.79 ms
  - attack_input: 89.16 ms
- Movement enumeration internals:
  - movementRangePathSearch: 64.57 ms / 352 calls
  - movementPathLeaveBaseExpansion: 23.93 ms / 219 calls
  - movementPathBaseConnectivity: 9.28 ms / 262 calls
  - movementPathPositionForTile: 6.68 ms / 1,762 calls
  - movementPathGroundConnectivity: 4.03 ms / 628 calls
  - movementVisibleState: 4.02 ms / 352 calls
- Attack enumeration internals:
  - attackTargetSearch: 66.36 ms / 288 calls
  - attackRangeDistance: 24.21 ms / 264 calls
  - attackAcrossBaseBlocking: 17.27 ms / 4,017 calls
  - attackUnitCoordinateBaseSearch: 9.79 ms / 24 calls
  - attackFinalLegalCheck: 4.18 ms / 4,488 calls
- Next: inspect attackTargetSearch for repeated scans before selecting V6-Y.
- 50k remains NOT RUN.


## V6-Y road-section components — 1k gate PASS
- Branch: experiment/ppo-fast-batch-v6y-road-section-components
- Implementation commit: c087ffb14940f03b6879305895ad8c8d663a10be
- Baseline: V6-X 91f4c6f
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Local TypeScript static check: PASS.
- Related battle/construction tests: 50/50 PASS.
- Change: precompute connected-component IDs from the existing road-section adjacency graph once per RoadAttackTopologyContext.
- Context-backed connectivity checks now compare component IDs instead of repeating BFS for the same static graph.
- Non-context areRoadSectionsDynamicallyConnected remains unchanged.
- Game step: 397.18 -> 390.22 ms = 1.01784x speedup (~1.75% reduction).
- Decision: plausible but modest gain; proceed to 4k paired exactness/performance gate.
- 50k remains NOT RUN.


## V6-Y road-section components — CLOSED after 4k
- Branch: experiment/ppo-fast-batch-v6y-road-section-components
- Implementation commit: c087ffb14940f03b6879305895ad8c8d663a10be
- Baseline: V6-X 91f4c6f
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Game step: 1,661.22 -> 1,642.81 ms = 1.01121x speedup (~1.11% reduction).
- 1k had shown ~1.78%, but the 4k gain collapsed close to the established ~1% noise floor.
- Decision: exact but not a sufficiently material pure improvement; V6-Y is NOT promoted.
- Keep V6-X as the current exact baseline.
- Do not retry road-section connectivity with only minor implementation variations.
- 50k remains NOT RUN.

## Next after V6-Y closure
1. Keep V6-X as production baseline.
2. Profile resolve_battle internals before changing battle resolution.
3. Current V6-X profile measured resolve_battle at 88.95 ms / 24 calls per 1k decisions.
4. Do not alter RNG consumption, attack order, damage rolls, defeat/capture ordering, logs, rewards or state-transition semantics.


## V6-X battle-resolution re-profile
- Diagnostic branch: experiment/ppo-fast-batch-v6x-battle-profile
- Diagnostic commit: f4dc9eb32a28d0d5056a53d23c7292d4353eddc5
- Baseline: V6-X 91f4c6f
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Detailed profiling adds material overhead; candidate game-step wall is not a speed comparison.
- resolve_battle: 83.84 ms / 24 calls.
- Battle-stage breakdown:
  - event_build.neutral_intents: 65.57 ms
  - event_build.encouragement: 3.79 ms
  - event_build.start_positions: 2.24 ms
  - pre_hit_metadata: 1.82 ms
  - turn_flags_and_logs: 1.80 ms
  - status_cleanup: 1.33 ms
  - capture_and_king_resolution: 1.23 ms
  - all other measured stages were below 1 ms each.
- Conclusion: neutral attack-intent generation remains the dominant battle-resolution cost after V6-X.

## V6-X neutral attack-candidate detail profile
- Diagnostic branch: experiment/ppo-fast-batch-v6x-neutral-profile
- Diagnostic commit: bc84af2b724841956f20888dfa33a1574083f083
- Baseline: V6-X 91f4c6f
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- neutral candidate search: 84.34 ms / 360 calls.
- neutral attackTargetSearch: 83.53 ms / 360 calls.
- neutral attackRangeDistance: 55.82 ms / 516 calls.
- neutral attackAcrossBaseBlocking: 6.41 ms / 4,020 calls.
- neutral attackFinalLegalCheck: 2.40 ms / 4,020 calls.
- neutral attackLakeNinjaRule: 1.24 ms / 4,020 calls.
- neutral attackBasicFilter: 1.02 ms / 360 calls.
- Conclusion: bounded road attack-distance calculation accounts for roughly two thirds of neutral target-search cost.
- Do not retry the closed V6-R full-distance-lookup route.

## V6-X bounded attack-distance internal profile
- Diagnostic branch: experiment/ppo-fast-batch-v6x-neutral-distance-profile
- Diagnostic commit: 703e47c1775fb1068ec72d346dae03091be5fb97
- Baseline: V6-X 91f4c6f
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Fine-grained timing instrumentation inflates the profiled attackRangeDistance value, so use the substage split diagnostically rather than as a direct speed comparison.
- Neutral bounded-distance profile:
  - attackRangeDistance: 74.35 ms / 516 calls (instrumented)
  - attackRangeNeighborExpansion: 66.70 ms
  - ground-edge checks: 17.35 ms
  - road/bridge coordinate lookup: 13.07 ms
  - base-connect checks: 10.91 ms
  - base lookup: 5.12 ms
  - coordinate lookup: 1.76 ms
- Active-team enumeration showed the same shape:
  - attackRangeDistance: 28.55 ms / 264 calls
  - neighbor expansion: 24.40 ms
  - ground-edge checks: 8.60 ms
  - road/bridge lookup: 5.07 ms.
- Code audit: attackPathNeighbors is deterministic for a fixed GameState + RoadAttackTopologyContext + position, but is rebuilt on every bounded BFS visit.
- RoadAttackTopologyContext is state-scoped. Neutral resolution creates one context and shares it across all neutral candidate searches; attack-input caching also requires identical units/bases/constructions references.

## Next candidate: V6-Z lazy attack-path-neighbor memoization
- Add a context-local Map keyed by attackPathKey(position) that stores the exact ordered neighbor list after first construction.
- Only use this cache when a RoadAttackTopologyContext is present; context-free public behavior remains unchanged.
- Do not precompute the whole graph and do not revive V6-R full-distance lookup.
- Preserve neighbor ordering and exact UnitPosition values.
- Cache lifetime must remain the existing RoadAttackTopologyContext lifetime; never share across states.
- Run local battle/topology tests, then 1k paired exactness gate; 4k only if exact and materially faster.
- 50k remains NOT RUN.


## V6-Z lazy attack-path-neighbor memoization — 1k gate PASS
- Branch: experiment/ppo-fast-batch-v6z-attack-neighbor-cache
- Implementation commit: 40bc790d9c83fda34ef455b3b16b7cacd2f125ec
- Baseline: V6-X bc6de8002a7137d30eaf58451d1846825862ac21
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Local TypeScript static check: PASS.
- Related battle/heuristic/bridge tests: 47/47 PASS.
- Change: RoadAttackTopologyContext lazily memoizes the exact ordered attackPathNeighbors result by attackPathKey(position).
- No whole-graph precomputation; context-free behavior remains unchanged.
- Cache lifetime is the existing state-scoped RoadAttackTopologyContext lifetime.
- Game step: 402.61 -> 375.51 ms = 1.07217x speedup (~6.73% reduction).
- Unchanged batchAct/replay/external-wall differences are treated as run variability.
- Decision: 1k gate passed materially. Proceed to 4k paired exactness/performance gate.
- 50k remains NOT RUN.


## V6-Z lazy attack-path-neighbor memoization — 4k verified result
- Branch: experiment/ppo-fast-batch-v6z-attack-neighbor-cache
- Implementation commit: 40bc790d9c83fda34ef455b3b16b7cacd2f125ec
- Baseline: V6-X bc6de8002a7137d30eaf58451d1846825862ac21
- Workload: 8 env x 500 decisions = 4,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- CUDA Graph behavior exact: 29 captures, 426 replays, 74 fallbacks.
- Game step: 1,899.97 -> 1,762.10 ms = 1.07824x speedup (~7.26% reduction).
- Observation encode: 1,240.85 -> 1,248.61 ms; Action encode: 926.03 -> 953.24 ms.
- batchAct: 6,501.13 -> 5,678.41 ms; replay: 4,171.24 -> 3,851.72 ms.
  These unchanged-path differences are treated as run variability and are not attributed to V6-Z.
- External wall: 27.313 -> 19.570 sec; full wall difference is not attributed solely to V6-Z.
- Optimization is lazy and context-local: only positions actually visited by attack-path searches are memoized.
- No whole-graph distance precompute; V6-R remains closed.
- Cached neighbor arrays preserve original insertion/order and exact UnitPosition values.
- V6-Z is promoted as the current fastest exact-verified baseline.
- 50k remains NOT RUN.

## Next steps after V6-Z
1. Keep V6-Z as production baseline.
2. Re-profile game-step / neutral battle cost after the neighbor cache before choosing the next route.
3. Re-measure neutral attackRangeDistance and normal attackTargetSearch because both should be affected by V6-Z.
4. Prefer another measured structural duplicate over broad precomputation.
5. Keep 50k deferred until the post-V6-Z residual bottlenecks are characterized.


## V6-Z post-promotion re-profile
- Diagnostic branch: experiment/ppo-fast-batch-v6z-neutral-reprofile
- Diagnostic commit: dc7e1583507afaeea2f23fe0e9a078dbc016aac9
- Baseline: V6-Z a5ef3d225581dd6cf49018cbd1179dee07d65172
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Detailed profiler overhead makes candidate game-step wall non-comparable.
- Residual hotspots:
  - movement enumeration: 123.93 ms
  - attack enumeration: 80.71 ms
  - resolve_battle: 85.72 ms / 24 calls
  - submit_team_production: about 44 ms / 32 calls in subsequent production profiling
  - neutral attackRangeDistance: 35.73 ms / 516 calls
  - neutral attackTargetSearch: 63.69 ms / 360 calls
  - normal attackRangeDistance: 14.93 ms / 264 calls
  - normal attackTargetSearch: 57.52 ms / 288 calls
- V6-Z materially reduced bounded attack-distance work but did not eliminate it.

## V6-Z production-submit detail profile
- Diagnostic branch: experiment/ppo-fast-batch-v6z-production-profile
- Diagnostic commit: b6f4157ffda16f6e1931af678a0fddb41ec905d9
- Baseline: V6-Z a5ef3d225581dd6cf49018cbd1179dee07d65172
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- submit_team_production: 44.44 ms / 32 calls.
- Internal split:
  - structuredClone(state): 41.05 ms
  - choices extraction: 0.25 ms
  - applyProductionChoices: 1.60 ms
  - cleanup: 0.37 ms
- About 92% of submit-team-production time is the full GameState clone.
- Existing production.test.ts test "produces before movement and lets the new unit move immediately"
  fails identically on unchanged V6-Z baseline a5ef3d2; it is not a profiling regression.

## Next candidate: V7-A RL-only in-place production submit
- Branch: experiment/ppo-fast-batch-v7a-inplace-production
- Implementation commit: 0a6c52e4fec9bcdb930f3bb2a962f3790383e8a6
- General submitTeamProduction remains clone-based and unchanged.
- RL fast environment explicitly opts into submitTeamProductionInPlaceForRl.
- The in-place function preserves production-choice order, legality checks, unit ID generation,
  base-slot mutation, unit insertion, logs, action-intent cleanup and completed-team bookkeeping.
- Local TypeScript static check: PASS.
- Direct cloned-submit vs RL-in-place full-state equality test: PASS.
- immediateMovement + profiler tests: 14/14 PASS.
- Kaggle Version 129 1k paired exactness gate: PENDING (currently queued).
- Do not promote V7-A until Kaggle allExact and material game-step improvement are observed.
- 50k remains NOT RUN.


## V7-A RL-only in-place production — 1k gate PASS
- Branch: experiment/ppo-fast-batch-v7a-inplace-production
- Implementation commit: 0a6c52e4fec9bcdb930f3bb2a962f3790383e8a6
- Baseline: V6-Z a5ef3d225581dd6cf49018cbd1179dee07d65172
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Game step: 433.66 -> 403.41 ms = 1.07499x speedup (~6.98% reduction).
- General submitTeamProduction remains clone-based; only the RL fast environment opts into the in-place path.
- Decision: 1k gate passed materially; proceed to 4k paired exactness/performance gate.
- 50k remains NOT RUN.


## V7-A RL-only in-place production — 4k verified result
- Branch: experiment/ppo-fast-batch-v7a-inplace-production
- Implementation commit: 0a6c52e4fec9bcdb930f3bb2a962f3790383e8a6
- Baseline: V6-Z a5ef3d225581dd6cf49018cbd1179dee07d65172
- Workload: 8 env x 500 decisions = 4,000 decisions.
- Kaggle Version 131: COMPLETE.
- allExact=true; semantic/model/optimizer/CPU RNG/CUDA RNG/counters/retention all exact.
- Game step: 1,658.28 -> 1,519.63 ms = 1.09124x speedup (~8.36% reduction).
- Candidate observation encode: 1,189.08 ms; action encode: 915.02 ms; batchAct: 5,393.45 ms; replay restore/accumulate: 3,708.94 ms.
- General submitTeamProduction remains clone-based and unchanged; the RL fast environment uses the exact in-place submit path.
- Decision: V7-A is promoted as the current fastest exact-verified baseline.
- Re-profile residual game-step costs before selecting the next optimization route.
- 50k remains NOT RUN.


## V7-A residual game-step re-profile
- Diagnostic branch: experiment/ppo-fast-batch-v7a-residual-reprofile
- Diagnostic commit: c0ad03403dfdc5b8bd1b17b953bf92403bb17df3
- Baseline: V7-A 0a6c52e4fec9bcdb930f3bb2a962f3790383e8a6
- Workload: 8 env x 125 decisions = 1,000 decisions.
- allExact=true; detailed profiler overhead makes candidate game-step wall non-comparable.
- Baseline uninstrumented game-step: 350.90 ms; instrumented candidate: 416.03 ms.
- Residual top-level costs: movement enumeration 110.24 ms, attack enumeration 72.43 ms, resolve_battle 82.33 ms / 24 calls.
- Production submit is no longer a hotspot: submit_team_production 2.53 ms / 32 calls.
- Movement detail: movementRangePathSearch 63.25 ms / 352 calls; leave-base expansion 23.33 ms / 219 calls.
- Neutral battle detail: attackTargetSearch 62.47 ms / 360 calls; attackRangeDistance 35.97 ms / 516 calls.
- Normal attack detail: attackTargetSearch 51.41 ms / 288 calls; attackRangeDistance 12.99 ms / 264 calls.
- Keep V6-R full-distance lookup and V6-V per-search leave-base cache closed.
- Next route: investigate movement path work that is unnecessary for destination-only legal-action enumeration.
- 50k remains NOT RUN.


## V7-B movement destination-only — CLOSED after 1k
- Branch: experiment/ppo-fast-batch-v7b-movement-destination-only
- Implementation commit: 9afdfea14587d12751c67fde799b2a8a494083bf
- Baseline: V7-A 0a6c52e4fec9bcdb930f3bb2a962f3790383e8a6
- Workload: 8 env x 125 decisions = 1,000 decisions.
- Kaggle Version 133: COMPLETE.
- allExact=true.
- Game step: 322.54 -> 336.79 ms = 0.95769x speedup (~4.42% slower).
- Focused tests: 64 PASS / 1 FAIL; the single cpuCandidateAccess failure reproduces identically on unchanged V7-A baseline and is not a V7-B regression.
- Decision: exact but slower; V7-B is CLOSED and not promoted.
- Keep V7-A as the production baseline.
- Residual profiling and code audit found no remaining exact-preserving structural candidate with a clear material upside; return to the deferred 50k timing validation.
- 50k remains NOT RUN at this point.


## V7-A 50k timing validation — PASS
- Branch: experiment/ppo-fast-batch-v7a-inplace-production
- Production implementation commit: 0a6c52e4fec9bcdb930f3bb2a962f3790383e8a6
- Kaggle Version 134: COMPLETE.
- Probe: ppo_v7a_50k_time_validation.
- Resume source: fixed update2 checkpoint.
- Workload: 8 environments x 6,250 decision cap = 50,000 decisions maximum.
- Actual decisions: 50,000.
- Total samples: 50,000.
- Production timing run used node profiler OFF.
- integrityPass=true.
- CUDA selected: PASS.
- Resume counters: updateCount=2, episodeCount=2.
- Output counters: updateCount=3, episodeCount=10.
- Replay validation: 50,000 / 50,000 samples.
- Replay accumulation: 50,000 / 50,000 samples.
- Retention after replay: currentChunks=0, pendingChunks=0, currentRetainedBytes=0.
- Final retention: currentChunks=0, pendingChunks=0, currentRetainedBytes=0.
- Checkpoint kind, schemaVersion and featureSpec integrity checks: PASS.
- Outcomes: 0 victory, 8 time-limit adjudicated, 0 abnormal truncated.
- Per-environment decisions: [6250, 6250, 6250, 6250, 6250, 6250, 6250, 6250].
- Wall time: 149.462 sec (~2 min 29 sec).
- Internal total: 147,636.095 ms.
- Rollout: 105,848.072 ms = 2.11696 ms/decision.
- Replay/update validation section: 33,359.682 ms = 0.66719 ms/sample.
- Peak raw retention before replay: 4,568,323,408 bytes (~4.57 GB decimal), fully drained after replay.
- Historical 50k manifest recorded 1,131.09 sec on the older single-episode path. The raw wall-time ratio is about 7.57x and the elapsed-time reduction about 86.8%, but this is a historical cross-path comparison, not a paired exact performance gate.
- Exact semantic equivalence of V7-A itself was established separately by the 1k and 4k paired gates; the 50k run is an integrity/timing validation, not another allExact paired run.

## Phase 12B-1 speed-optimization closure
- Current production baseline: V7-A RL-only in-place production submit.
- 50k timing validation from the fixed update2 checkpoint: PASS.
- Required checkpoint/replay/retention/integrity gates passed.
- V7-B was exact but slower and remains CLOSED.
- V6-R full distance lookup, V6-V leave-base per-search cache, V6-Y minor connectivity optimization and V7-B destination-only movement route remain CLOSED; do not revive them with minor variants.
- Residual profiling found movement enumeration, attack enumeration and battle resolution as remaining costs, but no next exact-preserving structural candidate with a clearly material expected gain was identified.
- Decision: stop micro-optimization here for Phase 12B-1. The original deferred 50k validation has now been completed successfully.
- Next work should use V7-A as the fixed baseline and move to the next PPO validation/training objective rather than continue unmeasured micro-tuning.


## V7-C rollout workers 1k exact gate — EXACT BUT SLOWER
- Branch: experiment/ppo-fast-batch-v7c-rollout-workers
- Candidate commit: 1875b0b5e6976f0dc30ea49e2bafdc69685292b1
- Kaggle Version 139: COMPLETE.
- Probe: ppo_v7c_rollout_workers_1k_gate.
- Workload: fixed update2, 8 environments x 125 decisions = 1,000 decisions.
- allExact=true.
- Exact checks passed for semantic summaries, update result, parameter/optimizer/RNG/gradient hashes, model/optimizer checkpoint state, CPU/CUDA RNG state, counters, Feature Spec, hyperparameters, seed and sample counts.
- Baseline V7-A wall: 13.8175 sec.
- Candidate V7-C wall: 14.8610 sec.
- wallSpeedup: 0.92978x (~7.55% slower).
- Baseline rollout: 2,966.08 ms.
- Candidate rollout: 5,583.37 ms.
- rolloutSpeedup: 0.53123x (~1.88x slower).
- Baseline replay: 1,788.22 ms.
- Candidate replay: 1,853.16 ms (roughly neutral/noisy).
- Candidate 4-worker prepare barrier: 3,149.16 ms.
- Candidate apply barrier: 366.69 ms.
- Sum of measured worker CPU work: ~1,626.34 ms across observation/legal-action/encode/game-step.
- Interpretation: worker semantic parallelism is valid, but deep structured-clone/IPC of encoded observations and sparse actions dominates the prepare barrier and more than erases the CPU parallelism gain.
- Decision: do not promote V7-C as-is. Preserve the exact worker architecture as evidence, but next attempt must reduce worker-to-parent transport rather than add more workers.
- Next candidate direction: worker-side packing + transferable binary payloads, while keeping central Python/GPU actBatch ordering, current raw retention/replay, RNG ordering and V7-A semantics unchanged.


## V7-D worker packed transfer 1k exact gate — EXACT, 4k REQUIRED
- Branch: experiment/ppo-fast-batch-v7d-worker-packed-transfer
- Candidate commit: 38018f87082e90cbd0cfdcfb1398b51ab89a00cc
- Kaggle Version 142: COMPLETE.
- Probe: ppo_v7d_worker_packed_transfer_1k_gate.
- Workload: fixed update2, 8 environments x 125 decisions = 1,000 decisions.
- allExact=true.
- Exact checks passed for semantic summaries, update result, parameter/optimizer/RNG/gradient hashes, model/optimizer checkpoint state, CPU/CUDA RNG state, counters, Feature Spec, hyperparameters, seed and sample counts.
- Baseline V7-A wall: 13.6635 sec.
- Candidate V7-D wall: 12.5061 sec.
- wallSpeedup: 1.09255x (~8.47% wall reduction).
- Baseline rollout: 2,876.51 ms.
- Candidate rollout: 3,091.17 ms.
- rolloutSpeedup: 0.93056x (~7.46% rollout regression).
- Baseline replay: 1,816.17 ms.
- Candidate replay: 1,782.60 ms.
- replaySpeedup: 1.01883x (roughly neutral / slightly faster).
- Candidate 4-worker prepare barrier: 982.22 ms.
- Candidate apply barrier: 458.30 ms.
- Candidate worker CPU totals: observation 28.39 ms; legal actions 3.34 ms; observation encode 664.04 ms; action encode 480.09 ms; worker packing 909.31 ms; game-step 760.99 ms.
- Compared with V7-C, prepare barrier improved from 3,149.16 ms to 982.22 ms (~3.21x), confirming transferable packed payloads removed most structured-clone overhead.
- Interpretation: the intended transport bottleneck was materially reduced and exactness is preserved, but 1k rollout remains slower than V7-A; the overall wall improvement may include startup/update noise.
- Decision: do not promote from 1k. Run a 4k paired exact gate before deciding whether V7-D is beneficial.


## V7-D worker packed transfer 1k gate — EXACT / 4k REQUIRED
- Branch: experiment/ppo-fast-batch-v7d-worker-packed-transfer
- Candidate commit: 38018f87082e90cbd0cfdcfb1398b51ab89a00cc
- Kaggle Version 146: COMPLETE.
- Probe: ppo_v7d_worker_packed_transfer_1k_gate.
- Workload: fixed update2, 8 environments x 125 decisions = 1,000 decisions.
- allExact=true.
- Exact checks passed for semantic summaries, update result, parameter/optimizer/RNG/gradient hashes, model/optimizer checkpoint state, CPU/CUDA RNG state, counters, Feature Spec, hyperparameters, seed and sample counts.
- Baseline V7-A wall: 14.7238 sec.
- Candidate V7-D wall: 12.9669 sec.
- wallSpeedup: 1.13549x (~11.93% wall-time reduction).
- Baseline rollout: 2,945.21 ms.
- Candidate rollout: 3,094.39 ms.
- rolloutSpeedup: 0.95179x (~5.07% slower rollout).
- Baseline replay: 1,829.25 ms.
- Candidate replay: 2,117.14 ms.
- Candidate 4-worker prepare barrier: 1,059.83 ms.
- Candidate apply barrier: 449.65 ms.
- V7-C prepare barrier was 3,149.16 ms; worker-side packed transfer cut this by ~66.3%.
- Worker CPU totals included observation 39.08 ms, legal actions 3.07 ms, observation encode 677.68 ms, action encode 540.60 ms, packing 1,003.86 ms and game-step 774.34 ms. These totals overlap across workers and are not wall time.
- Interpretation: worker-side packing/transfer fixes the dominant V7-C structured-clone penalty, but the 1k rollout itself is still slightly slower than V7-A. The end-to-end wall improvement may include run-to-run variation outside rollout/replay.
- Decision: do not promote from 1k alone. Run the same exact-preserving paired gate at 4k before deciding promotion or closure.
