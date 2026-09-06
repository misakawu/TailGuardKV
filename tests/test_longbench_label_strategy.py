from __future__ import annotations

import sys
from pathlib import Path


LABELING_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(LABELING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LABELING_SCRIPTS))


from common import MAINSTREAM_H2O_PROFILES, MAINSTREAM_KIVI_PROFILES
from label_longbench_quality import classify_profile_losses


def _row(request_id: str, *, task: str, risk_family: str, prompt_len: int = 100) -> dict:
    return {
        "request_id": request_id,
        "task": task,
        "prompt": "x" * prompt_len,
        "reference": "ref",
        "metadata": {
            "source": "external_labeling_workspace",
            "source_dataset": f"longbench_{task}",
            "risk_family": risk_family,
        },
    }


def test_tie_sensitive_rows_are_not_eligible_for_sensitive_reserves() -> None:
    losses = {
        "full_gpu": 0.0,
        **{profile: 0.07 for profile in MAINSTREAM_KIVI_PROFILES},
        **{profile: 0.07 for profile in MAINSTREAM_H2O_PROFILES},
    }

    assert classify_profile_losses(losses) == "tie_sensitive"
