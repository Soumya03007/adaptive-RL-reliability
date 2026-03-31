import csv
import json
from datetime import datetime
from pathlib import Path

import torch


def prepare_run_dir(base_dir="training/runs", run_name=None):
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_metadata(run_dir, train_config, env_config):
    metadata_path = run_dir / "run_config.json"
    metadata_path.write_text(
        json.dumps(
            {
                "train_config": train_config,
                "env_config": env_config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path


def append_metrics_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, iteration, policy_net, value_net, optimizer, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "policy_state_dict": policy_net.state_dict(),
            "value_state_dict": value_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
