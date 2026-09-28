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

## Next steps
1. Run an 8k V5 vs V6-B profiled performance comparison.
2. Confirm which stages shrink and whether the 4k speedup scales.
3. If the structural gain holds, integrate the direct sparse Action encoder proven in Probe-02.
4. Then evaluate direct compact Observation generation.
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
