# u54 new-rules ability audit

Prepared-only Kaggle experiment. It resumes the validated u53 checkpoint for exactly one PPO update on game seeds 417–424. No push or training is performed by preparation.

Fixed game/audit commit: `e0223f1b0c06465bda43982f8cec43456f6797a8`. The post-fix 5,000-decision ON/OFF benchmark is recorded in `ability-audit-performance-5000.json`, with the pre-fix no_actor result preserved in `ability-audit-performance-5000-before-no-actor-fix.json`; the read-only UI/PPO gameplay comparison is in `ui-vs-ppo-gameplay-audit.md`.

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

After explicit approval, after pushing fixed commit `e0223f1b0c06465bda43982f8cec43456f6797a8`, start the real Kaggle run with `-Execute`. The wrapper uses the established local runner, a distinct Kernel slug, and a 720-minute monitoring timeout so a long queue is reported as monitor timeout rather than a learning failure. It never auto-retries the same Notebook.
