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
