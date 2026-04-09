from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_one(
    python_bin: str,
    run_config: str,
    mode: str,
    interaction_arch: str,
    use_rope: bool,
    out_json: Path,
) -> None:
    cmd: List[str] = [
        python_bin,
        str(ROOT / "scripts" / "train_ctr.py"),
        "--run-config",
        run_config,
        "--interleave-mode",
        mode,
        "--interaction-arch",
        interaction_arch,
        "--save-metrics-path",
        str(out_json),
    ]
    if use_rope:
        cmd.append("--use-rope")

    print("[ablation]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-config", type=str, default="configs/ablation.local.json")
    parser.add_argument("--modes", nargs="*", default=None, help="Optional override list: sole sep n2s s2n int")
    args = parser.parse_args()

    cfg = load_json(args.ablation_config)
    run_config = cfg["run_config"]
    modes = args.modes if args.modes else cfg.get("modes", ["sole", "sep", "n2s", "s2n", "int"])
    interaction_arch = cfg.get("interaction_arch", "dhen")
    use_rope = bool(cfg.get("use_rope", False))
    output_dir = Path(cfg.get("output_dir", "results/ablation"))
    summary_csv = Path(cfg.get("summary_csv", output_dir / "summary.csv"))
    python_bin = cfg.get("python_bin", "python")

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in modes:
        out_json = output_dir / f"{mode}.json"
        run_one(
            python_bin=python_bin,
            run_config=run_config,
            mode=mode,
            interaction_arch=interaction_arch,
            use_rope=use_rope,
            out_json=out_json,
        )
        res = load_json(out_json)
        rows.append(
            {
                "mode": mode,
                "interaction_arch": res.get("interaction_arch", interaction_arch),
                "use_rope": res.get("use_rope", use_rope),
                "train_logloss": res.get("train_logloss", 0.0),
                "val_auc": res.get("val", {}).get("auc", 0.0),
                "val_gauc": res.get("val", {}).get("gauc", 0.0),
                "val_logloss": res.get("val", {}).get("logloss", 0.0),
                "test_auc": (res.get("test") or {}).get("auc", ""),
                "test_gauc": (res.get("test") or {}).get("gauc", ""),
                "test_logloss": (res.get("test") or {}).get("logloss", ""),
                "result_json": str(out_json),
            }
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "interaction_arch",
        "use_rope",
        "train_logloss",
        "val_auc",
        "val_gauc",
        "val_logloss",
        "test_auc",
        "test_gauc",
        "test_logloss",
        "result_json",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ablation] summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
