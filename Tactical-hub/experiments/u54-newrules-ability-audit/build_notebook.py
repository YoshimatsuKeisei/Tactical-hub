from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "notebook_main.py").read_text(encoding="utf-8")
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Tactical-hub u53 to u54 new-rules PPO ability audit\n",
                "\n",
                "Retreat mode removed; PPO schemas and u53 model/optimizer state retained.\n",
                "\n",
                "Marker: TTH_U54_NEWRULES_ABILITY_AUDIT_20261019\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        },
    ],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with (HERE / "tth-u54-newrules-ability-audit.ipynb").open(
    "w", encoding="utf-8", newline="\n",
) as handle:
    handle.write(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
