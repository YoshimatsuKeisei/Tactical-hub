import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PROBE = "ppo_u54_newrules_ability_audit_v1"
MARKER = "TTH_U54_NEWRULES_ABILITY_AUDIT_20261019"
COMMIT = "096ad1daff4975db8697f728b8590d94de5b3f88"
SOURCE_ARCHIVE_SHA256 = "19660d33a67d401c7e99453dffcb1433e0b727e9b8352a4577b19bb35bab2dec"
SOURCE_KERNEL = "keisabo/tactical-hub-u52-one-update-balance-trial"
SOURCE_CHECKPOINT_SHA256 = "cce23ec427ee350071244690840e787455baa1651b4d60ce779a7951ae42222a"
SOURCE_UPDATE = 53
SOURCE_EPISODE = 410
TARGET_UPDATE = 54
TARGET_EPISODE = 418
TRAINER_SEED = 7
EXPECTED_GAME_SEEDS = list(range(417, 425))

ROOT = Path("/kaggle/working/Tactical-hub-u54-newrules-ability-audit")
PROJECT = ROOT / "Tactical-hub"
SOURCE_MOUNT = Path("/kaggle/input") / SOURCE_KERNEL.split("/")[-1]
OUTPUT = Path("/kaggle/working/ppo-u54-newrules-ability-audit.pt")
BEST = Path("/kaggle/working/ppo-u54-newrules-ability-audit-best.pt")
REPORT = Path("/kaggle/working/production-update-report-u54-newrules-ability-audit.json")
ABILITY_JSON = Path("/kaggle/working/ability-audit-u54-newrules.json")
ABILITY_MARKDOWN = Path("/kaggle/working/ability-audit-u54-newrules.md")
RETENTION_DIR = Path("/kaggle/working/ppo-retention-u54-newrules")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("TTH_U54_NEWRULES_AUDIT_START", MARKER, flush=True)
source_candidates = sorted(Path("/kaggle/input").rglob("ppo-u53-oneupdate-trial.pt"))
if len(source_candidates) != 1:
    raise RuntimeError(f"Expected exactly one validated u53 checkpoint, got {source_candidates}")
SOURCE = source_candidates[0]
if SOURCE_MOUNT.name not in str(SOURCE):
    raise RuntimeError(f"u53 checkpoint came from wrong kernel source: {SOURCE}")
source_sha_before = sha256_file(SOURCE)
if source_sha_before != SOURCE_CHECKPOINT_SHA256:
    raise RuntimeError(f"u53 SHA mismatch expected={SOURCE_CHECKPOINT_SHA256} actual={source_sha_before}")

if ROOT.exists():
    shutil.rmtree(ROOT)
if RETENTION_DIR.exists():
    shutil.rmtree(RETENTION_DIR)
for path in (OUTPUT, BEST, REPORT, ABILITY_JSON, ABILITY_MARKDOWN):
    if path.exists():
        path.unlink()

subprocess.run(["git", "clone", "--no-checkout", "https://github.com/YoshimatsuKeisei/Tactical-hub.git", str(ROOT)], check=True)
subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", COMMIT], check=True)
subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", COMMIT], check=True)
head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
if head != COMMIT:
    raise RuntimeError(f"Wrong fixed commit: {head}")
archive = subprocess.check_output(["git", "-C", str(ROOT), "archive", "--format=tar", COMMIT, "Tactical-hub"])
archive_sha = hashlib.sha256(archive).hexdigest()
if archive_sha != SOURCE_ARCHIVE_SHA256:
    raise RuntimeError(f"source archive SHA mismatch expected={SOURCE_ARCHIVE_SHA256} actual={archive_sha}")

subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=PROJECT, check=True)
subprocess.run([
    "npx", "vitest", "run",
    "src/game/__tests__/retreat.test.ts",
    "src/game/__tests__/rlConstructionHistoryOverflow.test.ts",
    "src/game/__tests__/rlAbilityAudit.test.ts",
    "src/game/__tests__/rlFeatureSpec.test.ts",
    "src/game/__tests__/rlObservationEncoder.test.ts",
    "src/game/__tests__/rlActionEncoder.test.ts",
    "src/game/__tests__/ninjaTeleport.test.ts",
    "src/game/__tests__/ninjaObstacleMovement.test.ts",
    "src/game/__tests__/ninjaWaterMovement.test.ts",
    "src/game/__tests__/construction.test.ts",
    "src/game/__tests__/teleport.test.ts",
    "src/game/__tests__/encouragement.test.ts",
    "src/game/__tests__/kingEncouragement.test.ts",
    "--reporter=dot",
], cwd=PROJECT, check=True)
subprocess.run(["npx", "tsc", "--noEmit"], cwd=PROJECT, check=True)
subprocess.run([sys.executable, "-m", "py_compile", str(PROJECT / "rl" / "ppo_server.py")], check=True)

source_ckpt = torch.load(SOURCE, map_location="cpu", weights_only=False)
if source_ckpt.get("schemaVersion") != 1 or source_ckpt.get("checkpointKind") != "ppo_self_play":
    raise RuntimeError("u53 checkpoint schema/kind mismatch")
if int(source_ckpt.get("updateCount", -1)) != SOURCE_UPDATE or int(source_ckpt.get("episodeCount", -1)) != SOURCE_EPISODE:
    raise RuntimeError("u53 checkpoint counters mismatch")
if int(source_ckpt.get("seed", -1)) != TRAINER_SEED:
    raise RuntimeError("u53 trainer seed mismatch")
feature_spec = source_ckpt.get("featureSpec")
hyperparameters = source_ckpt.get("hyperparameters")
if not isinstance(feature_spec, dict) or feature_spec.get("schemaVersion") != 2:
    raise RuntimeError("u53 Feature Spec schema mismatch")

probe_ts = PROJECT / "src/game/cpu/u54FeatureSpecPreflightCli.ts"
probe_ts.write_text(
    'import { RlEnvironmentV2 } from "./rlEnvironment";\n'
    'import { createRlFeatureSpecV2 } from "./rlFeatureSpec";\n'
    'const env = new RlEnvironmentV2(undefined, true);\n'
    'console.log("TTH_FEATURE_SPEC=" + JSON.stringify(createRlFeatureSpecV2(env.reset(313, 4))));\n',
    encoding="utf-8",
)
spec_result = subprocess.run(
    ["npx", "vite-node", "src/game/cpu/u54FeatureSpecPreflightCli.ts"],
    cwd=PROJECT, text=True, capture_output=True, check=True,
)
spec_lines = [line.split("TTH_FEATURE_SPEC=", 1)[1] for line in spec_result.stdout.splitlines() if line.startswith("TTH_FEATURE_SPEC=")]
if len(spec_lines) != 1 or json.loads(spec_lines[0]) != feature_spec:
    raise RuntimeError("new-rules Feature Spec differs from u53")
print("TTH_U54_NEWRULES_FEATURE_SPEC_EXACT_PASS", flush=True)

sys.path.insert(0, str(PROJECT))
from rl.ppo_trainer import PpoTrainer

read_only_probe = PpoTrainer(feature_spec, hyperparameters, TRAINER_SEED, "cpu")
resume_state = read_only_probe.resume(str(SOURCE))
if resume_state != {"updateCount": SOURCE_UPDATE, "episodeCount": SOURCE_EPISODE}:
    raise RuntimeError(f"strict u53 resume mismatch: {resume_state}")
if sha256_file(SOURCE) != SOURCE_CHECKPOINT_SHA256:
    raise RuntimeError("u53 checkpoint changed during strict load")

env = os.environ.copy()
env["PPO_RETENTION_STORAGE_MODE"] = "deflate_disk"
env["PPO_RETENTION_DISK_DIR"] = str(RETENTION_DIR)
cmd = [
    "npx", "vite-node", "src/game/cpu/rlPpoFastBatchV7WorkersCli.ts",
    "--seed", str(TRAINER_SEED),
    "--environments", "8",
    "--rollout-workers", "4",
    "--validation-workers", "2",
    "--initial-checkpoint", str(SOURCE),
    "--resume", str(SOURCE),
    "--checkpoint", str(OUTPUT),
    "--best-checkpoint", str(BEST),
    "--safety-max-turns", "1000",
    "--safety-max-actions", "50000",
    "--replay-chunk-size", "32",
    "--memory-log-interval", "5000",
    "--battle-advantage-shaping-beta", "0.02",
    "--ability-audit",
    "--ability-audit-json", str(ABILITY_JSON),
    "--ability-audit-markdown", str(ABILITY_MARKDOWN),
]
proc = subprocess.run(cmd, cwd=PROJECT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if proc.stderr:
    print(proc.stderr, file=sys.stderr, end="")
if proc.returncode != 0:
    print(proc.stdout, flush=True)
    raise RuntimeError(f"u54 new-rules PPO update failed rc={proc.returncode}")
result = json.loads(proc.stdout)

if result.get("selectedDevice") != "cuda":
    raise RuntimeError("CUDA not selected")
if result.get("checkpointStart") != {"updateCount": SOURCE_UPDATE, "episodeCount": SOURCE_EPISODE}:
    raise RuntimeError(f"checkpoint start mismatch: {result.get('checkpointStart')}")
if result.get("seeds") != EXPECTED_GAME_SEEDS:
    raise RuntimeError(f"game seed mismatch: {result.get('seeds')}")
if int(result.get("environmentCount", -1)) != 8:
    raise RuntimeError("environment count mismatch")
total_decisions = int(result.get("totalDecisions", -1))
total_samples = int(result.get("totalSamples", -1))
if not (0 < total_decisions <= 400000) or total_samples != total_decisions:
    raise RuntimeError("decision/sample accounting mismatch")
if int(result.get("replayedSamples", -1)) != total_samples or int(result.get("validatedSamples", -1)) != total_samples:
    raise RuntimeError("Replay/Validation sample count mismatch")
if result.get("validationSkipped") or int(result.get("validationWorkerCount", -1)) != 2:
    raise RuntimeError("validation was not performed with two workers")
summaries = result.get("summaries") or []
if len(summaries) != 8:
    raise RuntimeError("match summary count mismatch")
decision_counts = [int(summary.get("decisionCount", -1)) for summary in summaries]
if any(count <= 0 or count > 50000 for count in decision_counts) or sum(decision_counts) != total_decisions:
    raise RuntimeError(f"per-match decision accounting mismatch: {decision_counts}")
if int(result.get("truncatedEpisodeCount", -1)) != 0:
    raise RuntimeError("truncated episode detected")

update = result.get("update") or {}
if int(update.get("updateCount", -1)) != TARGET_UPDATE or int(update.get("episodeCount", -1)) != TARGET_EPISODE:
    raise RuntimeError(f"target counters mismatch: {update}")
if not update.get("policyParametersChanged") or not update.get("valueParametersChanged"):
    raise RuntimeError("policy/value parameters did not both change")
for key in ("loss", "policyLoss", "valueLoss", "entropy", "gradientNorm"):
    if not math.isfinite(float(update[key])):
        raise RuntimeError(f"non-finite update metric {key}={update[key]}")

balance = result.get("balanceAnalysis") or {}
ability = result.get("abilityAudit") or {}
if ability.get("schemaVersion") != 1 or int(ability.get("matchCount", -1)) != 8:
    raise RuntimeError("ability audit schema/match count mismatch")
production = ability.get("production") or {}
if production.get("counts") != balance.get("productionCounts"):
    raise RuntimeError("ability-audit and balance production counts differ")
production_total = int(production.get("selectionCount", -1))
if production_total <= 0 or sum(int(value) for value in production.get("counts", {}).values()) != production_total:
    raise RuntimeError("production audit accounting mismatch")
for unit_type, count in production["counts"].items():
    if abs(float(production["shares"][unit_type]) - int(count) / production_total) > 1e-12:
        raise RuntimeError(f"production share mismatch for {unit_type}")
if len(ability.get("perMatch") or []) != 8:
    raise RuntimeError("per-match ability audit missing")

for kind in ("bridge", "obstacle"):
    stats = ability["strategist"][kind]
    if int(stats["successCount"]) + int(stats["failureCount"]) != int(stats["instructionCount"]):
        raise RuntimeError(f"{kind} instruction outcome accounting mismatch")
    if int(stats["placedCount"]) != int(stats["successCount"]):
        raise RuntimeError(f"{kind} placement/success mismatch")
teleport = ability["strategist"]["teleport"]
if int(teleport["successCount"]) + int(teleport["failureCount"]) != int(teleport["instructionCount"]):
    raise RuntimeError("strategist teleport outcome accounting mismatch")
encouragement = ability["encouragement"]
if int(encouragement["appliedAttackCount"]) + int(encouragement["withoutEncouragementCount"]) != int(encouragement["totalAttackCount"]):
    raise RuntimeError("encouragement total accounting mismatch")
if int(encouragement["hitCount"]) + int(encouragement["missCount"]) != int(encouragement["appliedAttackCount"]):
    raise RuntimeError("encouragement hit/miss accounting mismatch")
if int(encouragement["bothAppliedAttackCount"]) > min(int(encouragement["strategistAppliedAttackCount"]), int(encouragement["kingAppliedAttackCount"])):
    raise RuntimeError("encouragement overlap accounting mismatch")

if not OUTPUT.is_file() or not ABILITY_JSON.is_file() or not ABILITY_MARKDOWN.is_file():
    raise RuntimeError("expected u54 output missing")
output_ckpt = torch.load(OUTPUT, map_location="cpu", weights_only=False)
if int(output_ckpt.get("updateCount", -1)) != TARGET_UPDATE or int(output_ckpt.get("episodeCount", -1)) != TARGET_EPISODE:
    raise RuntimeError("saved checkpoint counters mismatch")
if output_ckpt.get("featureSpec") != feature_spec or output_ckpt.get("hyperparameters") != hyperparameters or output_ckpt.get("seed") != TRAINER_SEED:
    raise RuntimeError("saved checkpoint compatibility metadata changed")
for name, tensor in output_ckpt["modelStateDict"].items():
    if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite model tensor {name}")
for parameter_id, state in output_ckpt["optimizerStateDict"]["state"].items():
    for state_name, value in state.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite optimizer tensor pid={parameter_id} state={state_name}")

source_sha_after = sha256_file(SOURCE)
if source_sha_after != SOURCE_CHECKPOINT_SHA256:
    raise RuntimeError("u53 checkpoint changed during experiment")
output_sha = sha256_file(OUTPUT)
report = {
    "probe": PROBE,
    "allExact": True,
    "newRulesCommit": COMMIT,
    "sourceArchiveSha256": SOURCE_ARCHIVE_SHA256,
    "source": {
        "kernel": SOURCE_KERNEL,
        "sha256Before": source_sha_before,
        "sha256After": source_sha_after,
        "updateCount": SOURCE_UPDATE,
        "episodeCount": SOURCE_EPISODE,
    },
    "target": {"updateCount": TARGET_UPDATE, "episodeCount": TARGET_EPISODE, "outputSha256": output_sha},
    "workload": {
        "trainerSeed": TRAINER_SEED,
        "gameSeeds": EXPECTED_GAME_SEEDS,
        "environmentCount": 8,
        "maxDecisionsPerMatch": 50000,
        "maxTotalDecisions": 400000,
        "actualTotalDecisions": total_decisions,
        "decisionCountByMatch": decision_counts,
        "maxTurns": 1000,
        "rolloutWorkers": 4,
        "validationWorkers": 2,
        "replayChunkSize": 32,
        "memoryLogInterval": 5000,
        "battleAdvantageShapingBeta": 0.02,
    },
    "featureSpec": feature_spec,
    "hyperparameters": hyperparameters,
    "updateMetrics": {key: update[key] for key in (
        "sampleCount", "loss", "policyLoss", "valueLoss", "entropy", "gradientNorm",
        "policyParametersChanged", "valueParametersChanged",
    )},
    "outcomes": {
        "victory": int(result.get("victoryEpisodeCount", 0)),
        "adjudicated": int(result.get("adjudicatedEpisodeCount", 0)),
        "draw": int(result.get("drawEpisodeCount", 0)),
        "truncated": int(result.get("truncatedEpisodeCount", 0)),
    },
    "balanceAnalysis": balance,
    "abilityAudit": ability,
    "statisticsSchemaVersion": 1,
    "experimentMarker": MARKER,
}
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
shutil.rmtree(ROOT, ignore_errors=True)
shutil.rmtree(RETENTION_DIR, ignore_errors=True)
if BEST.exists():
    BEST.unlink()
print("TTH_U54_NEWRULES_ABILITY_AUDIT_SUCCESS", MARKER, flush=True)
print(json.dumps(report, indent=2), flush=True)
