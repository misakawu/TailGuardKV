from __future__ import annotations

import pytest

from run_util.canonical_history import (
    CANONICAL_HISTORY_MODE,
    CanonicalHistoryError,
    CanonicalBootstrapManifest,
    build_canonical_fixture,
    canonical_history_hash,
    validate_canonical_history_rows,
)
from run_util.core_types import Request
from run_util.canonical_session import bootstrap_canonical_session
from run_util.data_utils import _with_session_histories


def _rows() -> list[dict[str, object]]:
    return [
        {
            "request_id": "s1_t0",
            "session_id": "s1",
            "turn_index": 0,
            "arrival_index": 0,
            "task": "qa",
            "prompt": "first question",
            "reference": "long reference answer that must not become history",
            "metadata": {"split": "calibration"},
        },
        {
            "request_id": "s1_t1",
            "session_id": "s1",
            "turn_index": 1,
            "arrival_index": 1,
            "task": "qa",
            "prompt": "follow up",
            "reference": "another scoring-only reference",
            "metadata": {"split": "calibration"},
        },
    ]


def test_canonical_fixture_uses_full_gpu_output_in_later_history() -> None:
    fixture = build_canonical_fixture(
        _rows(),
        {"s1_t0": "generated answer"},
        bootstrap_table_hash="bootstrap-table",
        output_token_count=lambda text: len(text.split()),
    )

    assert fixture[1]["metadata"]["canonical_history_mode"] == CANONICAL_HISTORY_MODE
    assert fixture[1]["metadata"]["canonical_history"] == [
        "User: first question",
        "Assistant: generated answer",
    ]
    assert "long reference" not in "\n".join(fixture[1]["metadata"]["canonical_history"])

    requests = [
        Request(
            request_id=str(row["request_id"]),
            task=str(row["task"]),
            prompt=str(row["prompt"]),
            reference=str(row["reference"]),
            session_id=str(row["session_id"]),
            turn_index=int(row["turn_index"]),
            arrival_index=int(row["arrival_index"]),
            metadata=dict(row["metadata"]),
        )
        for row in fixture
    ]
    rendered = _with_session_histories(requests)
    assert rendered[1].history_turns == ("User: first question", "Assistant: generated answer")


def test_canonical_fixture_rejects_missing_or_forged_history_hash() -> None:
    fixture = build_canonical_fixture(
        _rows(),
        {"s1_t0": "generated answer"},
        bootstrap_table_hash="bootstrap-table",
        output_token_count=lambda text: 1,
    )
    fixture[1]["metadata"]["canonical_history_hash"] = "forged"

    with pytest.raises(CanonicalHistoryError, match="canonical_history_mismatch"):
        validate_canonical_history_rows(fixture)


def test_canonical_fixture_rejects_cross_profile_or_turn_binding() -> None:
    fixture = build_canonical_fixture(
        _rows(),
        {"s1_t0": "generated answer"},
        bootstrap_table_hash="bootstrap-table",
        output_token_count=lambda text: 1,
    )
    fixture[1]["metadata"]["canonical_history_source_profile"] = "kivi_4bit_residual32"

    with pytest.raises(CanonicalHistoryError, match="canonical_history_mismatch"):
        validate_canonical_history_rows(fixture)


def test_canonical_fixture_rejects_bootstrap_outputs_over_sixteen_tokens() -> None:
    with pytest.raises(CanonicalHistoryError, match="16"):
        build_canonical_fixture(
            _rows(),
            {"s1_t0": "too many tokens"},
            bootstrap_table_hash="bootstrap-table",
            output_token_count=lambda text: 17,
        )


def test_canonical_fixture_rejects_forged_bootstrap_output_token_count() -> None:
    fixture = build_canonical_fixture(
        _rows(),
        {"s1_t0": "generated answer"},
        bootstrap_table_hash="bootstrap-table",
        output_token_count=lambda text: 2,
    )
    fixture[1]["metadata"]["canonical_history_output_token_counts"] = [17]

    with pytest.raises(CanonicalHistoryError, match="16"):
        validate_canonical_history_rows(fixture)


def test_canonical_history_hash_binds_ordered_history() -> None:
    assert canonical_history_hash(["User: a", "Assistant: b"]) != canonical_history_hash(
        ["Assistant: b", "User: a"]
    )


def test_bootstrap_runner_feeds_generated_output_to_the_next_turn() -> None:
    requests = [
        Request(
            request_id=str(row["request_id"]),
            task=str(row["task"]),
            prompt=str(row["prompt"]),
            reference=str(row["reference"]),
            session_id=str(row["session_id"]),
            turn_index=int(row["turn_index"]),
            arrival_index=int(row["arrival_index"]),
            metadata=dict(row["metadata"]),
        )
        for row in _rows()
    ]
    observed_histories: list[tuple[str, ...]] = []

    fixture, outputs, manifest = bootstrap_canonical_session(
        requests,
        lambda request: observed_histories.append(request.history_turns) or f"generated-{request.turn_index}",
        bootstrap_table_hash="bootstrap-table",
        output_token_count=lambda text: 1,
    )

    assert observed_histories == [(), ("User: first question", "Assistant: generated-0")]
    assert outputs == {"s1_t0": "generated-0", "s1_t1": "generated-1"}
    assert fixture[1]["metadata"]["canonical_history"][-1] == "Assistant: generated-0"
    assert manifest.fixture_hash


def test_canonical_bootstrap_manifest_binds_fixture_table_and_turn_hashes() -> None:
    fixture = build_canonical_fixture(
        _rows(), {"s1_t0": "generated answer"}, bootstrap_table_hash="bootstrap-table", output_token_count=lambda text: 2
    )

    manifest = CanonicalBootstrapManifest.from_fixture(fixture, bootstrap_table_hash="bootstrap-table")

    assert manifest.profile == "full_gpu"
    assert manifest.generation_max_new_tokens == 16
    assert manifest.fixture_hash
    assert manifest.turn_history_hashes["s1_t1"] == fixture[1]["metadata"]["canonical_history_hash"]
    manifest.validate_fixture(fixture)
