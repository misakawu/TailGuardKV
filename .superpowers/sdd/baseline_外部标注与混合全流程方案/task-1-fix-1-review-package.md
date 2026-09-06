32d1d4b fix: 拒绝空 RAGhot 证据
 scripts/validate_trace_quality.py         |  2 +-
 tests/test_pilot_session_trace_dataset.py | 20 ++++++++++++++++++++
 2 files changed, 21 insertions(+), 1 deletion(-)
diff --git a/scripts/validate_trace_quality.py b/scripts/validate_trace_quality.py
index 25173fc..83ee229 100644
--- a/scripts/validate_trace_quality.py
+++ b/scripts/validate_trace_quality.py
@@ -283,21 +283,21 @@ def _quality_record_provenance_failure(fixture_row: dict[str, Any]) -> str | Non
     if str(metadata.get("source", "")).strip() != HYBRID_SOURCE:
         return f"source 必须是 {HYBRID_SOURCE}"
     content_source = str(metadata.get("content_source_dataset", "")).strip().lower()
     source_spec = SOURCE_REGISTRY.get(content_source)
     if source_spec is None:
         return "content_source_dataset 不在来源注册表中"
     if str(metadata.get("source_dataset", "")).strip() != source_spec["hybrid_source_dataset"]:
         return "source_dataset 必须与 content_source_dataset 的来源注册表一致"
     for key in (*INJECTED_TURN_REQUIRED_METADATA, "original_session_id", "hybrid_turn_role", *source_spec["required_metadata"]):
         value = metadata.get(key)
-        if value is None or value == "":
+        if value is None or value == "" or (isinstance(value, (list, tuple, set, dict)) and not value):
             return f"metadata 缺少 {key}"
     try:
         turn_index = int(fixture_row.get("turn_index", -1))
     except (TypeError, ValueError):
         return "turn_index 必须是整数"
     if turn_index not in {2, 3, 4}:
         return f"风险质量记录 turn_index 必须在 2..4: turn_index={turn_index}"
     role = str(metadata["hybrid_turn_role"])
     if role != TURN_ROLES[turn_index]:
         return (
diff --git a/tests/test_pilot_session_trace_dataset.py b/tests/test_pilot_session_trace_dataset.py
index a68d2c8..62157ad 100644
--- a/tests/test_pilot_session_trace_dataset.py
+++ b/tests/test_pilot_session_trace_dataset.py
@@ -280,20 +280,40 @@ def test_validate_trace_quality_rejects_untraceable_eval_quality_record() -> Non
 
     result = validate_trace_quality(measurements, fixture_rows)
 
     assert result.passed is False
     assert len(result.provenance_failures) == 2
     assert any("content_source_request_id" in message for message in result.provenance_failures)
     assert any("foreign-eval" in message for message in result.provenance_failures)
     assert "kivi_sensitive-qa" not in {record["fixture_request_id"] for record in result.quality_records}
 
 
+def test_validate_trace_quality_rejects_empty_raghot_supporting_fact_ids() -> None:
+    fixture_rows, measurements = _passing_risk_gate_inputs()
+    raghot_row = fixture_rows[0]
+    raghot_row["metadata"].update(
+        {
+            "source_dataset": "sharegpt_raghot_qa_hybrid_session",
+            "content_source_dataset": "raghot_qa",
+            "context_pack_hash": "context-pack",
+            "supporting_fact_ids": [],
+            "packing_policy_version": "raghot_support_first_v1",
+        }
+    )
+
+    result = validate_trace_quality(measurements, fixture_rows)
+
+    assert result.passed is False
+    assert any("supporting_fact_ids" in message for message in result.provenance_failures)
+    assert "kivi_sensitive-qa" not in {record["fixture_request_id"] for record in result.quality_records}
+
+
 @pytest.mark.parametrize("quality_loss", [float("nan"), float("inf"), float("-inf")])
 def test_validate_trace_quality_rejects_non_finite_quality_loss(quality_loss: float) -> None:
     fixture_rows, measurements = _passing_risk_gate_inputs()
     object.__setattr__(measurements[0], "quality_loss", quality_loss)
 
     result = validate_trace_quality(measurements, fixture_rows)
 
     assert result.passed is False
     assert any("非有限" in message and "kivi_sensitive-qa" in message for message in result.provenance_failures)
     assert "kivi_sensitive-qa" not in {record["fixture_request_id"] for record in result.quality_records}
