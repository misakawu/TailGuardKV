e1901f1 feat: 增加 baseline 质量信号门禁
 run_util/experiment.py                    |  33 ++++-
 scripts/import_external_fixtures.py       |  57 ++++++--
 scripts/validate_trace_quality.py         | 233 +++++++++++++++++++++++++++---
 tests/test_baseline_quality_gate.py       | 110 ++++++++++++++
 tests/test_external_fixture_import.py     |  58 +++++++-
 tests/test_pilot_session_trace_dataset.py |   3 +-
 6 files changed, 452 insertions(+), 42 deletions(-)
diff --git a/run_util/experiment.py b/run_util/experiment.py
index afb7164..62b802a 100644
--- a/run_util/experiment.py
+++ b/run_util/experiment.py
@@ -45,31 +45,32 @@ from run_util.experiment_common import (
     synthesize_pressure_trace,
     validate_requests_for_experiment_type,
     validate_profile_measurements,
     write_csv,
 )
 from run_util.experiment_summary import summary_rows, write_summary, write_total_policy_summary
 from run_util.build_profile_table import build_profile_table
 from run_util.cli_common import first_number
 from run_util.run_policies import run_policies
 from scripts.generate_pilot_session_trace_requests import build_split_risk_lookup, validate_split_balance
-from scripts.validate_trace_quality import validate_trace_quality
+from scripts.validate_trace_quality import validate_baseline_quality_signal_gate, validate_trace_quality
 from visual.plot_summary import plot_summary
 
 
 PILOT_CONFIG = "configs/pilot.yaml"
 PILOT_PROFILE_OUTPUT = "out/profile_tables/pilot_smoke_measured_profiles.csv"
 PILOT_SESSION_TRACE_OUTPUT = "out/session_traces/pilot_smoke_measured_session_trace.csv"
 PILOT_POLICY_OUTPUT = "out/policy_tables/pilot_smoke_measured_policy.csv"
 PILOT_SUMMARY_OUTPUT = "out/policy_tables/pilot_smoke_measured_summary.csv"
 PILOT_TRACE_SEMANTICS_GATE_OUTPUT = "out/profile_tables/pilot_session_trace_semantics_gate.json"
 PILOT_RISK_SIGNAL_GATE_OUTPUT = "out/profile_tables/pilot_session_risk_signal_gate.json"
+PILOT_BASELINE_QUALITY_SIGNAL_GATE_OUTPUT = "out/profile_tables/pilot_baseline_quality_signal_gate.json"
 PILOT_SPLIT_VALIDATION_OUTPUT = "out/profile_tables/pilot_session_trace_split_validation.html"
 
 def _run_stage(func: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
     stream = io.StringIO()
     with redirect_stdout(stream):
         code = int(func(args))
     raw_output = stream.getvalue().strip()
     if not raw_output:
         return code, {}
     try:
@@ -409,20 +410,26 @@ def _write_json_gate(path: str, payload: dict[str, Any]) -> None:
 def _attach_policy_comparison_status(payload: dict[str, Any], status: str) -> None:
     payload["policy_comparison_status"] = status
     summary = payload.get("summary")
     if not isinstance(summary, dict):
         return
     for metrics in summary.values():
         if isinstance(metrics, dict):
             metrics["policy_comparison_status"] = status
 
 
+def apply_signal_gate_policy_status(payload: dict[str, Any], *, gate_passed: bool) -> str:
+    status = "formally_comparable" if gate_passed else "risk_evidence_insufficient"
+    _attach_policy_comparison_status(payload, status)
+    return status
+
+
 def pilot_smoke_measured(args: argparse.Namespace) -> int:
     config_path = getattr(args, "config", PILOT_CONFIG)
     experiment_type = ""
     run_dir = _resolve_run_dir(getattr(args, "run_dir", None), config_path)
     fallback_summary_output = str(_resolve_run_output(PILOT_SUMMARY_OUTPUT, run_dir))
     fallback_split_validation_output = str(_resolve_run_output(PILOT_SPLIT_VALIDATION_OUTPUT, run_dir))
     explicit_total_summary_output = str(getattr(args, "total_summary_output", "") or "")
     fallback_total_summary_output = str(
         _resolve_run_output(explicit_total_summary_output, run_dir)
         if explicit_total_summary_output
@@ -454,20 +461,26 @@ def pilot_smoke_measured(args: argparse.Namespace) -> int:
             _resolve_run_output(
                 str(
                     outputs.get(
                         "smoke_risk_signal_gate",
                         outputs.get("smoke_trace_quality_gate", PILOT_RISK_SIGNAL_GATE_OUTPUT),
                     )
                 ),
                 run_dir,
             )
         )
+        baseline_quality_signal_gate_output = str(
+            _resolve_run_output(
+                str(outputs.get("smoke_baseline_quality_signal_gate", PILOT_BASELINE_QUALITY_SIGNAL_GATE_OUTPUT)),
+                run_dir,
+            )
+        )
         split_validation_output = str(
             _resolve_run_output(str(outputs.get("smoke_split_validation", PILOT_SPLIT_VALIDATION_OUTPUT)), run_dir)
         )
         configured_total_summary_output = outputs.get("smoke_total_summary")
         if explicit_total_summary_output:
             total_summary_output = str(_resolve_run_output(explicit_total_summary_output, run_dir))
         elif configured_total_summary_output:
             total_summary_output = str(_resolve_run_output(str(configured_total_summary_output), run_dir))
         else:
             total_summary_output = _derive_total_summary_output(summary_output)
@@ -572,20 +585,21 @@ def pilot_smoke_measured(args: argparse.Namespace) -> int:
             "failures": profile_payload.get("failures"),
             "profile": profile_payload,
             "split_validation_output": split_validation_output,
         }
         _print_and_write(payload)
         return 2
 
     sweeps = _policy_sweep_points(config)
     trace_semantics_gate_payload: dict[str, Any] = {}
     risk_signal_gate_payload: dict[str, Any] = {}
+    baseline_quality_signal_gate_payload: dict[str, Any] = {}
     split_validation_payload: dict[str, Any] = {}
     policy_comparison_status = ""
     fixture_rows: list[dict[str, Any]] = []
     if experiment_type == "baseline_session":
         try:
             fixture_rows = _load_fixture_rows(str(config.get("data", {}).get("requests")))
         except (FileNotFoundError, ValueError, KeyError) as exc:
             trace_semantics_gate_payload = {"passed": False, "errors": [str(exc)]}
         try:
             split_result = validate_split_balance(fixture_rows, build_split_risk_lookup(measurements))
@@ -681,20 +695,35 @@ def pilot_smoke_measured(args: argparse.Namespace) -> int:
             if risk_signal_gate_payload.get("passed")
             else "risk_evidence_insufficient"
         )
         risk_signal_gate_payload["policy_comparison_status"] = policy_comparison_status
         _write_json_gate(risk_signal_gate_output, risk_signal_gate_payload)
         for run in policy_runs:
             run["policy_comparison_status"] = policy_comparison_status
             run_payload = run.get("payload")
             if isinstance(run_payload, dict):
                 _attach_policy_comparison_status(run_payload, policy_comparison_status)
+    if experiment_type == "baseline_quality":
+        try:
+            quality_fixture_rows = _load_fixture_rows(str(config.get("data", {}).get("requests")))
+            quality_gate_result = validate_baseline_quality_signal_gate(measurements, quality_fixture_rows)
+            baseline_quality_signal_gate_payload = quality_gate_result.to_json()
+        except (FileNotFoundError, ValueError, KeyError) as exc:
+            baseline_quality_signal_gate_payload = {"passed": False, "errors": [str(exc)]}
+        policy_comparison_status = "formally_comparable" if baseline_quality_signal_gate_payload.get("passed") else "risk_evidence_insufficient"
+        baseline_quality_signal_gate_payload["policy_comparison_status"] = policy_comparison_status
+        _write_json_gate(baseline_quality_signal_gate_output, baseline_quality_signal_gate_payload)
+        for run in policy_runs:
+            run["policy_comparison_status"] = policy_comparison_status
+            run_payload = run.get("payload")
+            if isinstance(run_payload, dict):
+                apply_signal_gate_policy_status(run_payload, gate_passed=bool(baseline_quality_signal_gate_payload.get("passed")))
     payload = {
         "ok": policy_code == 0,
         "return_code": policy_code,
         "step": "complete" if policy_code == 0 else "run_policies",
         "config": config_path,
         "run_dir": str(run_dir),
         "experiment_type": experiment_type,
         "summary_output": summary_output,
         "total_summary_output": total_summary_output,
         "session_trace_output": session_trace_output,
@@ -708,20 +737,22 @@ def pilot_smoke_measured(args: argparse.Namespace) -> int:
         "delta": policy_payload.get("delta"),
         "memory_budget_mib": policy_payload.get("memory_budget_mib"),
         "profile": profile_payload,
         "session_trace": session_trace_payload,
         "split_validation_output": split_validation_output,
         "split_validation": split_validation_payload,
         "trace_semantics_gate_output": trace_semantics_gate_output,
         "trace_semantics_gate": trace_semantics_gate_payload,
         "risk_signal_gate_output": risk_signal_gate_output,
         "risk_signal_gate": risk_signal_gate_payload,
+        "baseline_quality_signal_gate_output": baseline_quality_signal_gate_output if experiment_type == "baseline_quality" else "",
+        "baseline_quality_signal_gate": baseline_quality_signal_gate_payload,
         "policy_comparison_status": policy_comparison_status,
         "policy": policy_payload,
         "policy_runs": policy_runs,
     }
     _print_and_write(payload)
     visual_outputs: list[Path] = []
     if policy_code == 0:
         try:
             visual_outputs = plot_summary(total_summary_output)
         except Exception as exc:  # pragma: no cover - visualization must not fail the experiment.
diff --git a/scripts/import_external_fixtures.py b/scripts/import_external_fixtures.py
index 1eade31..c21b765 100644
--- a/scripts/import_external_fixtures.py
+++ b/scripts/import_external_fixtures.py
@@ -11,36 +11,52 @@ from typing import Any
 REPO_ROOT = Path(__file__).resolve().parent.parent
 DEFAULT_BASELINE_QUALITY_DEST = REPO_ROOT / "data" / "fixtures" / "baseline_quality_external.jsonl"
 DEFAULT_BASELINE_SESSION_DEST = REPO_ROOT / "data" / "fixtures" / "baseline_session_external.jsonl"
 
 QUALITY_REQUIRED_TOP_LEVEL = ("request_id", "task", "prompt", "reference", "metadata")
 SESSION_REQUIRED_TOP_LEVEL = ("request_id", "task", "prompt", "reference", "session_id", "turn_index", "arrival_index", "metadata")
 REQUIRED_METADATA = ("source", "source_dataset", "split", "risk_family")
 ALLOWED_QUALITY_TASKS = {"qa", "summary", "code"}
 REQUIRED_RISK_FAMILIES = {"low_risk"}
 HYBRID_SOURCE = "hybrid_session_builder"
-HYBRID_SOURCE_DATASET = "sharegpt_longbench_hybrid_session"
+SOURCE_REGISTRY = {
+    "longbench": {
+        "hybrid_source_dataset": "sharegpt_longbench_hybrid_session",
+        "required_metadata": (),
+    },
+    "raghot_qa": {
+        "hybrid_source_dataset": "sharegpt_raghot_qa_hybrid_session",
+        "required_metadata": (
+            "context_pack_hash",
+            "supporting_fact_ids",
+            "packing_policy_version",
+        ),
+    },
+}
 HYBRID_REQUIRED_METADATA = (
+    "original_session_id",
+    "hybrid_turn_role",
+)
+INJECTED_TURN_REQUIRED_METADATA = (
     "content_source_dataset",
     "content_source_request_id",
     "content_source_index",
+    "content_payload_hash",
     "injection_template",
-    "original_session_id",
-    "hybrid_turn_role",
 )
 SESSION_RISK_FAMILIES = ("kivi_sensitive", "h2o_sensitive", "low_risk")
 SESSION_TASKS = ("qa", "summary")
 SESSION_SPLITS = ("calibration", "eval")
 SESSION_COUNT = 48
 TURNS_PER_SESSION = 5
 SESSIONS_PER_RISK_TASK_SPLIT = 4
-TURN_ROLES = ("sharegpt_opening", "sharegpt_opening", "longbench_content", "reference_recall", "reference_rewrite")
+TURN_ROLES = ("sharegpt_opening", "sharegpt_opening", "content_query", "reference_recall", "reference_rewrite")
 
 
 def main() -> int:
     parser = argparse.ArgumentParser(description="Validate and import external baseline fixtures into the repo.")
     parser.add_argument("--kind", choices=("baseline_quality", "baseline_session"), required=True)
     parser.add_argument("--input", required=True)
     parser.add_argument("--output", default="")
     parser.add_argument("--validate-only", action="store_true")
     args = parser.parse_args()
 
@@ -182,28 +198,36 @@ def validate_baseline_session_fixture(path: Path) -> dict[str, Any]:
             )
 
         opening_tasks = {str(row["task"]).strip().lower() for row in ordered[:2]}
         injected_tasks = {str(row["task"]).strip().lower() for row in ordered[2:]}
         if opening_tasks != {"chat"}:
             raise ValueError(f"baseline_session turn0/turn1 必须是 chat 开场: session={session_id}")
         if len(injected_tasks) != 1 or not injected_tasks <= set(SESSION_TASKS):
             raise ValueError(f"baseline_session 风险记录只能是 QA/Summary: session={session_id}")
 
         injected_metadata = metadata_rows[2:]
-        if any(str(metadata["content_source_dataset"]).strip().lower() != "longbench" for metadata in injected_metadata):
-            raise ValueError(f"baseline_session 注入 turn 缺少 LongBench provenance: session={session_id}")
-        provenance_keys = ("content_source_request_id", "content_source_index", "injection_template")
+        content_sources = {str(metadata["content_source_dataset"]).strip().lower() for metadata in injected_metadata}
+        if len(content_sources) != 1 or next(iter(content_sources), "") not in SOURCE_REGISTRY:
+            raise ValueError(f"baseline_session 注入 turn 的 content source 不受支持: session={session_id}")
+        content_source = next(iter(content_sources))
+        expected_source_dataset = SOURCE_REGISTRY[content_source]["hybrid_source_dataset"]
+        if {str(metadata["source_dataset"]) for metadata in metadata_rows} != {expected_source_dataset}:
+            raise ValueError(
+                "baseline_session source_dataset 必须与 content source 注册表一致: "
+                f"session={session_id} content_source={content_source}"
+            )
+        provenance_keys = (*INJECTED_TURN_REQUIRED_METADATA, *SOURCE_REGISTRY[content_source]["required_metadata"])
         for key in provenance_keys:
             if len({str(metadata[key]) for metadata in injected_metadata}) != 1:
                 raise ValueError(f"baseline_session 注入 provenance 在 session 内不一致: session={session_id} field={key}")
         if len({str(row["reference"]) for row in ordered[2:]}) != 1:
-            raise ValueError(f"baseline_session turn2/turn3/turn4 必须复用 LongBench reference: session={session_id}")
+            raise ValueError(f"baseline_session turn2/turn3/turn4 必须复用 content reference: session={session_id}")
 
         risk_family = next(iter(session_risks))
         split = next(iter(session_splits))
         task = next(iter(injected_tasks))
         key = f"{risk_family}/{task}/{split}"
         risk_task_split_session_counts[key] = risk_task_split_session_counts.get(key, 0) + 1
 
     expected_counts = {
         f"{risk_family}/{task}/{split}": SESSIONS_PER_RISK_TASK_SPLIT
         for risk_family in SESSION_RISK_FAMILIES
@@ -266,22 +290,20 @@ def _require_metadata(row: dict[str, Any], *, path: Path, row_index: int) -> dic
 
 def _validate_hybrid_session_row(
     row: dict[str, Any],
     metadata: dict[str, Any],
     *,
     turn_index: int,
     row_index: int,
 ) -> None:
     if metadata.get("source") != HYBRID_SOURCE:
         raise ValueError(f"baseline_session source 必须是 {HYBRID_SOURCE}: row={row_index}")
-    if metadata.get("source_dataset") != HYBRID_SOURCE_DATASET:
-        raise ValueError(f"baseline_session source_dataset 必须是 {HYBRID_SOURCE_DATASET}: row={row_index}")
     for key in HYBRID_REQUIRED_METADATA:
         value = metadata.get(key)
         if value is None or value == "":
             raise ValueError(f"baseline_session metadata 缺少 {key}: row={row_index}")
     if str(metadata["split"]) not in SESSION_SPLITS:
         raise ValueError(f"baseline_session split 非法: row={row_index} split={metadata['split']}")
     if str(metadata["risk_family"]) not in SESSION_RISK_FAMILIES:
         raise ValueError(f"baseline_session risk_family 非法: row={row_index} risk_family={metadata['risk_family']}")
     if turn_index < 0 or turn_index >= TURNS_PER_SESSION:
         raise ValueError(f"baseline_session turn_index 必须在 0..4: row={row_index} turn_index={turn_index}")
@@ -289,22 +311,33 @@ def _validate_hybrid_session_row(
         raise ValueError(
             "baseline_session hybrid_turn_role 与 turn_index 不匹配: "
             f"row={row_index} role={metadata['hybrid_turn_role']} turn_index={turn_index}"
         )
     task = str(row["task"]).strip().lower()
     if turn_index < 2 and task != "chat":
         raise ValueError(f"baseline_session turn0/turn1 必须是 chat 开场: row={row_index}")
     if turn_index >= 2:
         if task not in SESSION_TASKS:
             raise ValueError(f"baseline_session 风险记录只能是 QA/Summary: row={row_index}")
-        if str(metadata["content_source_dataset"]).strip().lower() != "longbench":
-            raise ValueError(f"baseline_session 注入 turn 缺少 LongBench provenance: row={row_index}")
+        content_source = str(metadata.get("content_source_dataset", "")).strip().lower()
+        source_spec = SOURCE_REGISTRY.get(content_source)
+        if source_spec is None:
+            raise ValueError(f"baseline_session 注入 turn 的 content source 不受支持: row={row_index}")
+        if metadata.get("source_dataset") != source_spec["hybrid_source_dataset"]:
+            raise ValueError(
+                "baseline_session source_dataset 必须与 content source 注册表一致: "
+                f"row={row_index} content_source={content_source}"
+            )
+        for key in (*INJECTED_TURN_REQUIRED_METADATA, *source_spec["required_metadata"]):
+            value = metadata.get(key)
+            if value is None or value == "" or value == []:
+                raise ValueError(f"baseline_session metadata 缺少 {key}: row={row_index}")
         if not str(row["reference"]).strip():
             raise ValueError(f"baseline_session 注入 turn reference 不能为空: row={row_index}")
 
 
 def _parse_int(value: Any, *, field_name: str, row_index: int) -> int:
     try:
         return int(value)
     except (TypeError, ValueError) as exc:
         raise ValueError(f"{field_name} 必须是整数: row={row_index}") from exc
 
diff --git a/scripts/validate_trace_quality.py b/scripts/validate_trace_quality.py
index 26703ff..25173fc 100644
--- a/scripts/validate_trace_quality.py
+++ b/scripts/validate_trace_quality.py
@@ -11,23 +11,46 @@ from typing import Any
 
 
 REPO_ROOT = Path(__file__).resolve().parent.parent
 if str(REPO_ROOT) not in sys.path:
     sys.path.insert(0, str(REPO_ROOT))
 
 
 from run_util.core_types import ProfileMeasurement
 from run_util.io_utils import read_measurements
 from scripts.generate_pilot_session_trace_requests import MAINSTREAM_H2O_PROFILES, MAINSTREAM_KIVI_PROFILES
+from scripts.import_external_fixtures import (
+    HYBRID_SOURCE,
+    INJECTED_TURN_REQUIRED_METADATA,
+    SOURCE_REGISTRY,
+    TURN_ROLES,
+)
 
 
 QUALITY_GATE_THRESHOLD = 0.02
+BASELINE_QUALITY_PROFILES = (
+    "full_gpu",
+    "kivi_4bit_residual32",
+    "kivi_4bit_residual64",
+    "kivi_2bit_residual32",
+    "kivi_2bit_residual64",
+    "h2o_heavy10_recent10",
+    "h2o_heavy15_recent15",
+    "h2o_heavy20_recent20",
+)
+BASELINE_QUALITY_KIVI_PROFILES = BASELINE_QUALITY_PROFILES[1:5]
+BASELINE_QUALITY_H2O_PROFILES = BASELINE_QUALITY_PROFILES[5:]
+STRICT_RISK_FAMILIES = ("kivi_sensitive", "h2o_sensitive", "low_risk")
+STRICT_GROUP_SIZE = 60
+LOW_RISK_LOSS_THRESHOLD = 0.01
+SENSITIVE_LABEL_THRESHOLD = 0.05
+FAMILY_GAP_THRESHOLD = 0.02
 
 
 @dataclass(frozen=True)
 class TraceQualityValidationResult:
     passed: bool
     errors: list[str] = field(default_factory=list)
     qualifying_profiles: list[str] = field(default_factory=list)
     covered_tasks: set[str] = field(default_factory=set)
     profile_means: dict[str, float] = field(default_factory=dict)
     group_means: dict[str, dict[str, float]] = field(default_factory=dict)
@@ -39,20 +62,35 @@ class TraceQualityValidationResult:
     def to_json(self) -> dict[str, Any]:
         payload = asdict(self)
         payload["covered_tasks"] = sorted(self.covered_tasks)
         payload["task_coverage"] = {
             risk_family: sorted(tasks)
             for risk_family, tasks in sorted(self.task_coverage.items())
         }
         return payload
 
 
+@dataclass(frozen=True)
+class BaselineQualitySignalGateResult:
+    passed: bool
+    errors: list[str] = field(default_factory=list)
+    fixture_group_counts: dict[str, int] = field(default_factory=dict)
+    split_counts: dict[str, int] = field(default_factory=dict)
+    complete_profiles: list[str] = field(default_factory=list)
+    qualifying_profiles: list[str] = field(default_factory=list)
+    group_means: dict[str, dict[str, float]] = field(default_factory=dict)
+    sensitive_control_gaps: dict[str, float | None] = field(default_factory=dict)
+
+    def to_json(self) -> dict[str, Any]:
+        return {"gate": "baseline_quality_signal_gate", **asdict(self)}
+
+
 def main() -> int:
     parser = argparse.ArgumentParser(description="Validate pilot session trace quality coverage before policy sweep.")
     parser.add_argument("--measurements", required=True)
     parser.add_argument("--requests", required=True)
     parser.add_argument("--output", default="")
     args = parser.parse_args()
 
     measurements = read_measurements(Path(args.measurements))
     fixture_rows = _load_fixture_rows(Path(args.requests))
     result = validate_trace_quality(measurements, fixture_rows)
@@ -235,54 +273,43 @@ def _fixture_request_id(row: ProfileMeasurement) -> str:
     if original not in {None, ""}:
         return str(original)
     request_id = str(row.request_id)
     return request_id.split("__pressure", 1)[0]
 
 
 def _quality_record_provenance_failure(fixture_row: dict[str, Any]) -> str | None:
     metadata = fixture_row.get("metadata")
     if not isinstance(metadata, dict):
         return "metadata 必须是对象"
-    required_values = {
-        "source": "hybrid_session_builder",
-        "source_dataset": "sharegpt_longbench_hybrid_session",
-        "content_source_dataset": "longbench",
-    }
-    for key, expected in required_values.items():
-        if str(metadata.get(key, "")).strip().lower() != expected:
-            return f"{key} 必须是 {expected}"
-    for key in (
-        "content_source_request_id",
-        "content_source_index",
-        "injection_template",
-        "original_session_id",
-        "hybrid_turn_role",
-    ):
+    if str(metadata.get("source", "")).strip() != HYBRID_SOURCE:
+        return f"source 必须是 {HYBRID_SOURCE}"
+    content_source = str(metadata.get("content_source_dataset", "")).strip().lower()
+    source_spec = SOURCE_REGISTRY.get(content_source)
+    if source_spec is None:
+        return "content_source_dataset 不在来源注册表中"
+    if str(metadata.get("source_dataset", "")).strip() != source_spec["hybrid_source_dataset"]:
+        return "source_dataset 必须与 content_source_dataset 的来源注册表一致"
+    for key in (*INJECTED_TURN_REQUIRED_METADATA, "original_session_id", "hybrid_turn_role", *source_spec["required_metadata"]):
         value = metadata.get(key)
         if value is None or value == "":
             return f"metadata 缺少 {key}"
     try:
         turn_index = int(fixture_row.get("turn_index", -1))
     except (TypeError, ValueError):
         return "turn_index 必须是整数"
-    expected_roles = {
-        2: "longbench_content",
-        3: "reference_recall",
-        4: "reference_rewrite",
-    }
-    if turn_index not in expected_roles:
+    if turn_index not in {2, 3, 4}:
         return f"风险质量记录 turn_index 必须在 2..4: turn_index={turn_index}"
     role = str(metadata["hybrid_turn_role"])
-    if role != expected_roles[turn_index]:
+    if role != TURN_ROLES[turn_index]:
         return (
             "hybrid_turn_role 与风险 turn_index 不匹配: "
-            f"turn_index={turn_index} role={role} expected={expected_roles[turn_index]}"
+            f"turn_index={turn_index} role={role} expected={TURN_ROLES[turn_index]}"
         )
     if str(fixture_row.get("task", "")).strip().lower() not in {"qa", "summary"}:
         return "质量记录 task 必须是 QA/Summary"
     return None
 
 
 def _quality_record_evidence(
     measurement: ProfileMeasurement,
     fixture_row: dict[str, Any],
 ) -> dict[str, Any]:
@@ -292,22 +319,182 @@ def _quality_record_evidence(
         "fixture_request_id": str(fixture_row["request_id"]),
         "session_id": str(fixture_row.get("session_id", "")),
         "turn_index": int(fixture_row.get("turn_index", 0)),
         "task": str(fixture_row["task"]).strip().lower(),
         "risk_family": str(metadata["risk_family"]),
         "profile": str(measurement.profile),
         "quality_loss": float(measurement.quality_loss),
         "content_source_dataset": str(metadata["content_source_dataset"]),
         "content_source_request_id": str(metadata["content_source_request_id"]),
         "content_source_index": metadata["content_source_index"],
+        "content_payload_hash": str(metadata["content_payload_hash"]),
         "injection_template": str(metadata["injection_template"]),
         "original_session_id": str(metadata["original_session_id"]),
         "hybrid_turn_role": str(metadata["hybrid_turn_role"]),
     }
 
 
+def validate_baseline_quality_signal_gate(
+    measurements: list[ProfileMeasurement],
+    fixture_rows: list[dict[str, Any]],
+) -> BaselineQualitySignalGateResult:
+    errors: list[str] = []
+    fixture_by_id: dict[str, dict[str, Any]] = {}
+    duplicate_ids: set[str] = set()
+    for row in fixture_rows:
+        request_id = str(row.get("request_id", "")).strip()
+        if not request_id:
+            errors.append("strict fixture contains a blank request_id")
+            continue
+        if request_id in fixture_by_id:
+            duplicate_ids.add(request_id)
+        fixture_by_id[request_id] = row
+    if duplicate_ids:
+        errors.append(f"strict fixture contains duplicate request_id values: {sorted(duplicate_ids)}")
+    if len(fixture_rows) != STRICT_GROUP_SIZE * len(STRICT_RISK_FAMILIES):
+        errors.append(f"strict fixture must contain 180 rows, got={len(fixture_rows)}")
+
+    fixture_group_counts = {family: 0 for family in STRICT_RISK_FAMILIES}
+    split_counts: dict[str, int] = {"calibration": 0, "eval": 0}
+    group_splits = {family: set() for family in STRICT_RISK_FAMILIES}
+    for request_id, row in fixture_by_id.items():
+        metadata = row.get("metadata")
+        if not isinstance(metadata, dict):
+            errors.append(f"request_id={request_id} metadata must be an object")
+            continue
+        family = str(metadata.get("risk_family", ""))
+        split = str(metadata.get("split", ""))
+        if family not in fixture_group_counts:
+            errors.append(f"request_id={request_id} has unsupported risk_family={family}")
+            continue
+        fixture_group_counts[family] += 1
+        if split not in split_counts:
+            errors.append(f"request_id={request_id} has unsupported split={split}")
+            continue
+        split_counts[split] += 1
+        group_splits[family].add(split)
+    for family, count in fixture_group_counts.items():
+        if count != STRICT_GROUP_SIZE:
+            errors.append(f"strict fixture requires {STRICT_GROUP_SIZE} rows for {family}, got={count}")
+        if group_splits[family] != {"calibration", "eval"}:
+            errors.append(f"{family} must cover calibration and eval")
+    if split_counts != {"calibration": 90, "eval": 90}:
+        errors.append(f"strict fixture requires 90 calibration and 90 eval rows, got={split_counts}")
+
+    measurement_by_request: dict[str, dict[str, ProfileMeasurement]] = {}
+    duplicate_measurements: set[tuple[str, str]] = set()
+    for measurement in measurements:
+        request_id = _fixture_request_id(measurement)
+        if request_id not in fixture_by_id or measurement.profile not in BASELINE_QUALITY_PROFILES:
+            continue
+        profile_rows = measurement_by_request.setdefault(request_id, {})
+        if measurement.profile in profile_rows:
+            duplicate_measurements.add((request_id, measurement.profile))
+        profile_rows[measurement.profile] = measurement
+    if duplicate_measurements:
+        errors.append(f"complete measurement contains duplicate request/profile rows: {sorted(duplicate_measurements)}")
+
+    complete_profiles: set[str] = set()
+    losses_by_group: dict[str, dict[str, list[float]]] = {
+        family: {profile: [] for profile in BASELINE_QUALITY_PROFILES}
+        for family in STRICT_RISK_FAMILIES
+    }
+    for request_id, fixture_row in fixture_by_id.items():
+        profile_rows = measurement_by_request.get(request_id, {})
+        missing_profiles = set(BASELINE_QUALITY_PROFILES) - set(profile_rows)
+        invalid_profiles = {
+            profile
+            for profile, measurement in profile_rows.items()
+            if not measurement.ok
+            or not measurement.measured
+            or measurement.quality_loss is None
+            or not math.isfinite(float(measurement.quality_loss))
+        }
+        if missing_profiles or invalid_profiles:
+            errors.append(
+                "complete measurement missing or invalid profiles: "
+                f"request_id={request_id} missing={sorted(missing_profiles)} invalid={sorted(invalid_profiles)}"
+            )
+            continue
+        complete_profiles.update(profile_rows)
+        metadata = fixture_row.get("metadata")
+        if not isinstance(metadata, dict):
+            continue
+        family = str(metadata.get("risk_family", ""))
+        if family not in losses_by_group:
+            continue
+        losses = {profile: float(measurement.quality_loss) for profile, measurement in profile_rows.items()}
+        for profile, loss in losses.items():
+            losses_by_group[family][profile].append(loss)
+        derived_family = _derive_final_risk_family(losses)
+        if derived_family != family:
+            detail = "tie" if derived_family == "tie" else derived_family
+            errors.append(
+                "final-form measurement does not support fixture risk label: "
+                f"request_id={request_id} labeled={family} derived={detail}"
+            )
+
+    group_means = {
+        family: {
+            profile: round(sum(values) / len(values), 6)
+            for profile, values in profile_losses.items()
+            if values
+        }
+        for family, profile_losses in losses_by_group.items()
+    }
+    family_profiles = {
+        "kivi_sensitive": BASELINE_QUALITY_KIVI_PROFILES,
+        "h2o_sensitive": BASELINE_QUALITY_H2O_PROFILES,
+    }
+    qualifying_profiles: list[str] = []
+    sensitive_control_gaps: dict[str, float | None] = {}
+    for family, profiles in family_profiles.items():
+        sensitive_values = [group_means.get(family, {}).get(profile) for profile in profiles]
+        low_risk_values = [group_means.get("low_risk", {}).get(profile) for profile in profiles]
+        if any(value is None for value in sensitive_values + low_risk_values):
+            sensitive_control_gaps[family] = None
+            errors.append(f"{family} lacks complete sensitive-vs-low-risk evidence")
+            continue
+        sensitive_peak = max(float(value) for value in sensitive_values if value is not None)
+        low_risk_peak = max(float(value) for value in low_risk_values if value is not None)
+        gap = sensitive_peak - low_risk_peak
+        sensitive_control_gaps[family] = round(gap, 6)
+        qualifying_profiles.extend(
+            profile
+            for profile in profiles
+            if float(group_means[family][profile]) > QUALITY_GATE_THRESHOLD
+        )
+        if sensitive_peak <= QUALITY_GATE_THRESHOLD:
+            errors.append(f"{family} needs a profile group mean quality_loss > {QUALITY_GATE_THRESHOLD:.2f}")
+        if gap <= 0.0:
+            errors.append(f"{family} needs positive sensitive-vs-low-risk evidence, got={gap:.6f}")
+
+    return BaselineQualitySignalGateResult(
+        passed=not errors,
+        errors=errors,
+        fixture_group_counts=dict(sorted(fixture_group_counts.items())),
+        split_counts=dict(sorted(split_counts.items())),
+        complete_profiles=list(BASELINE_QUALITY_PROFILES) if set(BASELINE_QUALITY_PROFILES) <= complete_profiles else [],
+        qualifying_profiles=sorted(set(qualifying_profiles)),
+        group_means=group_means,
+        sensitive_control_gaps=sensitive_control_gaps,
+    )
+
+
+def _derive_final_risk_family(losses: dict[str, float]) -> str:
+    kivi_loss = max(losses[profile] for profile in BASELINE_QUALITY_KIVI_PROFILES)
+    h2o_loss = max(losses[profile] for profile in BASELINE_QUALITY_H2O_PROFILES)
+    if max(kivi_loss, h2o_loss) <= LOW_RISK_LOSS_THRESHOLD:
+        return "low_risk"
+    if kivi_loss >= SENSITIVE_LABEL_THRESHOLD and kivi_loss - h2o_loss >= FAMILY_GAP_THRESHOLD:
+        return "kivi_sensitive"
+    if h2o_loss >= SENSITIVE_LABEL_THRESHOLD and h2o_loss - kivi_loss >= FAMILY_GAP_THRESHOLD:
+        return "h2o_sensitive"
+    return "tie"
+
+
 def _load_fixture_rows(path: Path) -> list[dict[str, Any]]:
     return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
 
 
 if __name__ == "__main__":
     raise SystemExit(main())
diff --git a/tests/test_baseline_quality_gate.py b/tests/test_baseline_quality_gate.py
new file mode 100644
index 0000000..61c11a5
--- /dev/null
+++ b/tests/test_baseline_quality_gate.py
@@ -0,0 +1,110 @@
+from __future__ import annotations
+
+from run_util.core_types import ProfileMeasurement
+
+
+BASELINE_PROFILES = (
+    "full_gpu",
+    "kivi_4bit_residual32",
+    "kivi_4bit_residual64",
+    "kivi_2bit_residual32",
+    "kivi_2bit_residual64",
+    "h2o_heavy10_recent10",
+    "h2o_heavy15_recent15",
+    "h2o_heavy20_recent20",
+)
+KIVI_PROFILES = set(BASELINE_PROFILES[1:5])
+H2O_PROFILES = set(BASELINE_PROFILES[5:])
+
+
+def _strict_fixture_and_measurements() -> tuple[list[dict[str, object]], list[ProfileMeasurement]]:
+    fixture_rows: list[dict[str, object]] = []
+    measurements: list[ProfileMeasurement] = []
+    for risk_family in ("kivi_sensitive", "h2o_sensitive", "low_risk"):
+        for index in range(60):
+            request_id = f"{risk_family}-{index:03d}"
+            split = "calibration" if index < 30 else "eval"
+            fixture_rows.append(
+                {
+                    "request_id": request_id,
+                    "task": "qa" if index % 2 == 0 else "summary",
+                    "prompt": f"final-form prompt {request_id}",
+                    "reference": f"reference {request_id}",
+                    "metadata": {
+                        "source": "external_labeling",
+                        "source_dataset": "longbench_qasper",
+                        "split": split,
+                        "risk_family": risk_family,
+                    },
+                }
+            )
+            for profile in BASELINE_PROFILES:
+                loss = 0.0 if profile == "full_gpu" else 0.005
+                if risk_family == "kivi_sensitive" and profile in KIVI_PROFILES:
+                    loss = 0.06
+                if risk_family == "h2o_sensitive" and profile in H2O_PROFILES:
+                    loss = 0.06
+                measurements.append(
+                    ProfileMeasurement(
+                        request_id=request_id,
+                        profile=profile,
+                        adapter="test",
+                        ok=True,
+                        measured=True,
+                        quality_loss=loss,
+                        extra={"original_request_id": request_id, "split": split},
+                    )
+                )
+    return fixture_rows, measurements
+
+
+def test_baseline_quality_signal_gate_accepts_strict_final_form_evidence() -> None:
+    fixture_rows, measurements = _strict_fixture_and_measurements()
+
+    from scripts.validate_trace_quality import validate_baseline_quality_signal_gate
+
+    result = validate_baseline_quality_signal_gate(measurements, fixture_rows)
+
+    assert result.passed is True
+    assert result.fixture_group_counts == {"h2o_sensitive": 60, "kivi_sensitive": 60, "low_risk": 60}
+    assert result.complete_profiles == list(BASELINE_PROFILES)
+    assert KIVI_PROFILES & set(result.qualifying_profiles)
+    assert H2O_PROFILES & set(result.qualifying_profiles)
+
+
+def test_baseline_quality_signal_gate_rejects_incomplete_profile_measurements() -> None:
+    fixture_rows, measurements = _strict_fixture_and_measurements()
+    measurements.pop()
+
+    from scripts.validate_trace_quality import validate_baseline_quality_signal_gate
+
+    result = validate_baseline_quality_signal_gate(measurements, fixture_rows)
+
+    assert result.passed is False
+    assert any("complete measurement" in error for error in result.errors)
+
+
+def test_baseline_quality_signal_gate_rejects_relabeled_tie() -> None:
+    fixture_rows, measurements = _strict_fixture_and_measurements()
+    for measurement in measurements:
+        if measurement.request_id == "kivi_sensitive-000" and measurement.profile != "full_gpu":
+            object.__setattr__(measurement, "quality_loss", 0.06)
+
+    from scripts.validate_trace_quality import validate_baseline_quality_signal_gate
+
+    result = validate_baseline_quality_signal_gate(measurements, fixture_rows)
+
+    assert result.passed is False
+    assert any("tie" in error for error in result.errors)
+
+
+def test_failed_quality_gate_marks_policy_output_risk_evidence_insufficient() -> None:
+    from run_util.experiment import apply_signal_gate_policy_status
+
+    policy_payload = {"rows": 5, "summary": {"full_lru": {"count": 5.0}}}
+
+    status = apply_signal_gate_policy_status(policy_payload, gate_passed=False)
+
+    assert status == "risk_evidence_insufficient"
+    assert policy_payload["rows"] == 5
+    assert policy_payload["summary"]["full_lru"]["policy_comparison_status"] == status
diff --git a/tests/test_external_fixture_import.py b/tests/test_external_fixture_import.py
index fdfb52c..c1d4313 100644
--- a/tests/test_external_fixture_import.py
+++ b/tests/test_external_fixture_import.py
@@ -81,56 +81,65 @@ def test_validate_baseline_quality_fixture_rejects_missing_required_metadata(tmp
         + "\n",
         encoding="utf-8",
     )
 
     from scripts.import_external_fixtures import validate_baseline_quality_fixture
 
     with pytest.raises(ValueError, match="source_dataset"):
         validate_baseline_quality_fixture(fixture_path)
 
 
-def _hybrid_session_rows() -> list[dict[str, object]]:
+def _hybrid_session_rows(*, content_source_dataset: str = "longbench") -> list[dict[str, object]]:
     sessions: list[list[dict[str, object]]] = []
     session_index = 0
-    roles = ("sharegpt_opening", "sharegpt_opening", "longbench_content", "reference_recall", "reference_rewrite")
+    roles = ("sharegpt_opening", "sharegpt_opening", "content_query", "reference_recall", "reference_rewrite")
     for risk_family in ("kivi_sensitive", "h2o_sensitive", "low_risk"):
         for task in ("qa", "summary"):
             for split in ("calibration", "eval"):
                 for _ in range(4):
                     session_id = f"hybrid-session-{session_index:03d}"
                     content_request_id = f"longbench-{session_index:03d}"
                     reference = f"reference-{session_index:03d}"
                     session_rows: list[dict[str, object]] = []
                     for turn_index in range(5):
                         injected = turn_index >= 2
                         session_rows.append(
                             {
                                 "request_id": f"{session_id}-turn-{turn_index}",
                                 "task": task if injected else "chat",
                                 "prompt": f"prompt {session_index} {turn_index}",
                                 "reference": reference if injected else f"opening {session_index} {turn_index}",
                                 "session_id": session_id,
                                 "turn_index": turn_index,
                                 "metadata": {
                                     "source": "hybrid_session_builder",
-                                    "source_dataset": "sharegpt_longbench_hybrid_session",
-                                    "content_source_dataset": "longbench" if injected else "sharegpt",
+                                    "source_dataset": f"sharegpt_{content_source_dataset}_hybrid_session",
+                                    "content_source_dataset": content_source_dataset if injected else "sharegpt",
                                     "content_source_request_id": content_request_id if injected else f"sharegpt-{session_index}-{turn_index}",
                                     "content_source_index": session_index if injected else session_index * 2 + turn_index,
+                                    "content_payload_hash": f"payload-{session_index:03d}" if injected else "",
                                     "injection_template": "template_a",
                                     "original_session_id": f"sharegpt-session-{session_index:03d}",
                                     "hybrid_turn_role": roles[turn_index],
                                     "split": split,
                                     "risk_family": risk_family,
                                 },
                             }
                         )
+                        if injected and content_source_dataset == "raghot_qa":
+                            session_rows[-1]["metadata"].update(
+                                {
+                                    "context_pack_hash": f"context-{session_index:03d}",
+                                    "supporting_fact_ids": [f"fact-{session_index:03d}"],
+                                    "packing_policy_version": "raghot_support_first_v1",
+                                }
+                            )
                     sessions.append(session_rows)
                     session_index += 1
 
     rows: list[dict[str, object]] = []
     for turn_index in range(5):
         for session_rows in sessions:
             row = dict(session_rows[turn_index])
             row["arrival_index"] = len(rows)
             rows.append(row)
     return rows
@@ -151,20 +160,29 @@ def test_validate_baseline_session_fixture_accepts_exact_hybrid_contract(tmp_pat
 
     report = validate_baseline_session_fixture(fixture_path)
 
     assert report["row_count"] == 240
     assert report["session_count"] == 48
     assert report["turns_per_session"] == 5
     assert set(report["risk_families"]) == {"kivi_sensitive", "h2o_sensitive", "low_risk"}
     assert set(report["risk_task_split_session_counts"].values()) == {4}
 
 
+def test_validate_baseline_session_fixture_accepts_raghot_qa_provenance(tmp_path: Path) -> None:
+    fixture_path = tmp_path / "baseline_session_raghot.jsonl"
+    _write_rows(fixture_path, _hybrid_session_rows(content_source_dataset="raghot_qa"))
+
+    from scripts.import_external_fixtures import validate_baseline_session_fixture
+
+    assert validate_baseline_session_fixture(fixture_path)["row_count"] == 240
+
+
 def test_validate_baseline_session_fixture_rejects_missing_hybrid_metadata(tmp_path: Path) -> None:
     fixture_path = tmp_path / "baseline_session_missing_metadata.jsonl"
     rows = _hybrid_session_rows()
     del rows[0]["metadata"]["original_session_id"]
     _write_rows(fixture_path, rows)
 
     from scripts.import_external_fixtures import validate_baseline_session_fixture
 
     with pytest.raises(ValueError, match="original_session_id"):
         validate_baseline_session_fixture(fixture_path)
@@ -172,21 +190,51 @@ def test_validate_baseline_session_fixture_rejects_missing_hybrid_metadata(tmp_p
 
 def test_validate_baseline_session_fixture_rejects_missing_longbench_provenance(tmp_path: Path) -> None:
     fixture_path = tmp_path / "baseline_session_bad_longbench.jsonl"
     rows = _hybrid_session_rows()
     injected_row = next(row for row in rows if row["turn_index"] == 2)
     injected_row["metadata"]["content_source_dataset"] = "sharegpt"
     _write_rows(fixture_path, rows)
 
     from scripts.import_external_fixtures import validate_baseline_session_fixture
 
-    with pytest.raises(ValueError, match="LongBench"):
+    with pytest.raises(ValueError, match="content source"):
+        validate_baseline_session_fixture(fixture_path)
+
+
+def test_validate_baseline_session_fixture_rejects_missing_raghot_evidence(tmp_path: Path) -> None:
+    fixture_path = tmp_path / "baseline_session_missing_raghot_evidence.jsonl"
+    rows = _hybrid_session_rows(content_source_dataset="raghot_qa")
+    del next(row for row in rows if row["turn_index"] == 2)["metadata"]["context_pack_hash"]
+    _write_rows(fixture_path, rows)
+
+    from scripts.import_external_fixtures import validate_baseline_session_fixture
+
+    with pytest.raises(ValueError, match="context_pack_hash"):
+        validate_baseline_session_fixture(fixture_path)
+
+
+def test_validate_baseline_session_fixture_rejects_forged_source_and_role(tmp_path: Path) -> None:
+    fixture_path = tmp_path / "baseline_session_forged_source.jsonl"
+    rows = _hybrid_session_rows()
+    next(row for row in rows if row["turn_index"] == 2)["metadata"]["content_source_dataset"] = "forged_source"
+    _write_rows(fixture_path, rows)
+
+    from scripts.import_external_fixtures import validate_baseline_session_fixture
+
+    with pytest.raises(ValueError, match="content source"):
+        validate_baseline_session_fixture(fixture_path)
+
+    next(row for row in rows if row["turn_index"] == 2)["metadata"]["content_source_dataset"] = "longbench"
+    next(row for row in rows if row["turn_index"] == 2)["metadata"]["hybrid_turn_role"] = "reference_recall"
+    _write_rows(fixture_path, rows)
+    with pytest.raises(ValueError, match="hybrid_turn_role"):
         validate_baseline_session_fixture(fixture_path)
 
 
 def test_validate_baseline_session_fixture_rejects_chat_as_injected_risk_sample(tmp_path: Path) -> None:
     fixture_path = tmp_path / "baseline_session_chat_risk.jsonl"
     rows = _hybrid_session_rows()
     injected_row = next(row for row in rows if row["turn_index"] == 2)
     injected_row["task"] = "chat"
     _write_rows(fixture_path, rows)
 
diff --git a/tests/test_pilot_session_trace_dataset.py b/tests/test_pilot_session_trace_dataset.py
index 0a9833e..a68d2c8 100644
--- a/tests/test_pilot_session_trace_dataset.py
+++ b/tests/test_pilot_session_trace_dataset.py
@@ -81,23 +81,24 @@ def _risk_template(
         "prompt": f"{task} prompt {request_id}",
         "reference": f"{task} reference {request_id}",
         "session_id": f"session-{request_id}",
         "turn_index": turn_index,
         "metadata": {
             "source": "hybrid_session_builder",
             "source_dataset": "sharegpt_longbench_hybrid_session",
             "content_source_dataset": "longbench" if injected else "sharegpt",
             "content_source_request_id": f"content-{request_id}",
             "content_source_index": 7,
+            "content_payload_hash": f"payload-{request_id}" if injected else "",
             "injection_template": "template_a",
             "original_session_id": f"sharegpt-{request_id}",
-            "hybrid_turn_role": "longbench_content" if injected else "sharegpt_opening",
+            "hybrid_turn_role": "content_query" if injected else "sharegpt_opening",
             "split": split,
             "risk_family": risk_family,
         },
     }
 
 
 def _passing_risk_gate_inputs() -> tuple[list[dict[str, object]], list[ProfileMeasurement]]:
     fixture_rows = [
         _risk_template(f"{family}-{task}", task=task, risk_family=family)
         for family in ("kivi_sensitive", "h2o_sensitive", "low_risk")
