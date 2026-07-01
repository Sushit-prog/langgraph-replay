"""Baseline pinning: store/load which run_id is the pinned baseline."""

import json
import os
from typing import Optional


class BaselineStore:
    """Manages the pinned baseline run ID for a project.

    Stores the baseline in .agenttrace/baseline.json in the working directory,
    matching the convention of local tool config in the project root.
    """

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = os.path.join(os.getcwd(), ".agenttrace")
        self._config_dir = config_dir
        self._baseline_file = os.path.join(config_dir, "baseline.json")

    def set_baseline(self, run_id: str) -> None:
        """Pin a run as the baseline."""
        os.makedirs(self._config_dir, exist_ok=True)
        data = {"baseline_run_id": run_id}
        with open(self._baseline_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_baseline(self) -> Optional[str]:
        """Return the pinned baseline run_id, or None if not set."""
        if not os.path.exists(self._baseline_file):
            return None
        with open(self._baseline_file) as f:
            data = json.load(f)
        return data.get("baseline_run_id")
