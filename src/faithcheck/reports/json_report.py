"""JSON report generator."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from faithcheck.models import FaithReport


class JsonReportGenerator:
    """Generate a machine-readable JSON faithfulness report."""

    @staticmethod
    def generate(report: FaithReport, output_path: Path) -> None:
        """Write the report to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2) + "\n"
        )
