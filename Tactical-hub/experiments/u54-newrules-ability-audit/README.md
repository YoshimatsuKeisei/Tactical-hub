# u54 new-rules ability audit

Prepared-only Kaggle experiment. It resumes the validated u53 checkpoint for exactly one PPO update on game seeds 417–424. No push or training is performed by preparation.

Fixed game/audit commit: `096ad1daff4975db8697f728b8590d94de5b3f88`. The local 5,000-decision ON/OFF benchmark is recorded in `ability-audit-performance-5000.json`; the read-only UI/PPO gameplay comparison is in `ui-vs-ppo-gameplay-audit.md`.

The Notebook checks the fixed source commit/archive hash, u53 checkpoint hash and counters, exact Feature Spec equality, strict model/optimizer resume, Replay/Validation counts, production composition, and ability-audit accounting before emitting uniquely named u54 artifacts.

Local verification:

```powershell
npx vite-node experiments/u54-newrules-ability-audit/feature_spec_probe.ts experiments/u54-newrules-ability-audit/feature-spec-u53.json
& "<python-with-torch>" experiments/u54-newrules-ability-audit/verify_u53_checkpoint.py "<u53-checkpoint>" experiments/u54-newrules-ability-audit/feature-spec-u53.json
python experiments/u54-newrules-ability-audit/build_notebook.py
```

Safe runner preflight only:

```powershell
& experiments/u54-newrules-ability-audit/run_experiment.ps1
```

After explicit approval, after pushing fixed commit `650483a14c76925c21b3d7e9fd1bcf4085f685b7`, start the real Kaggle run with `-Execute`. The wrapper uses the established local runner, a distinct Kernel slug, and a 720-minute monitoring timeout so a long queue is reported as monitor timeout rather than a learning failure. It never auto-retries the same Notebook.
