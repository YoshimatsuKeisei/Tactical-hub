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

## Next steps
1. Keep V6-F as the current baseline; do not adopt V6-E.
2. Target transport/H2D and other representation-preserving overhead before changing model math.
3. Inspect current PackedH2dWorkspace / CPU memory path for pinned-memory and avoidable copies.
4. Preserve the exact tensor shapes and values seen by the model.
5. Do not run 50k until the remaining gap to the 20x target is materially reduced.

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
