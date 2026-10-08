"""Generate a replayable source notebook from the reviewed training cell."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_HASH = "fbc0f5aee8bf14829e9ffb5485187788aaf9bdca31331b5d968cbfec0d127dec"

if __name__ == "__main__":
    base = (ROOT / "notebooks/colab_training_20261004/train_fold06.py").read_text(encoding="utf-8")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# SleepTCN: complete remaining matched source pairs on CUDA\n",
        "Preserve CPU folds 0–5. Train fresh pairs for folds 6–9; retain the incomplete CPU fold 6 artifact.\n",
        "Upload the four verified source-only ZIPs before running. Checkpoints are saved each epoch in the temporary runtime and exported after each pair.\n",
        "No SHHS access, no target-label selection, no change to epoch/patience settings.\n"]}]
    for fold in range(6, 10):
        manifest = json.loads((ROOT / f"runs/colab_training_20261004/fold_{fold:02d}/bundle_verification.json").read_bytes())
        text = base.replace("Fold06", f"Fold{fold:02d}").replace("fold_06", f"fold_{fold:02d}")
        text = text.replace("fold 6", f"fold {fold}").replace("FOLD 6", f"FOLD {fold}")
        text = text.replace("manifest['fold'] == 6", f"manifest['fold'] == {fold}")
        text = text.replace(BASE_HASH, manifest["archive_sha256"])
        if fold > 6:
            text = text.replace("assert hashlib.sha256(package.read_bytes())",
                "import time\n" +
                f"assert json.loads(pathlib.Path('/content/sleeptcn_training_20261004/results/fold_{fold-1:02d}/verification.json').read_bytes())['status'] == 'passed'\n" +
                "upload_wait_start = time.monotonic()\n" +
                f"while not package.exists() or package.stat().st_size < {manifest['archive_bytes']}:\n" +
                "    assert time.monotonic() - upload_wait_start < 600, 'Upload incomplete; no training started.'\n" +
                "    time.sleep(1)\nassert hashlib.sha256(package.read_bytes())")
        ast.parse(text)
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": text.splitlines(keepends=True)})
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}},
                "cells": cells}
    target = ROOT / "runs/colab_training_20261004/SleepTCN_Training_20261004_Source.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated four reviewed full-source training cells; no executed outputs included.")
