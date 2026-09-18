"""Build the isolated coarse-sweep config and verify reused measured inputs."""
import csv
import json
import sys
from pathlib import Path

import yaml

from run_util.config_loader import config_experiment_type, load_config
from run_util.derive_session_budgets import configured_memory_budgets
from run_util.experiment_common import read_measurements, validate_profile_measurements

root = Path(sys.argv[1]).resolve()
prior = root / 'out/pilot_tight_budget_coarse_batch007_20260911'
base = prior / 'configs/batch007.yaml'
measurements_path = prior / 'batch_outputs/batch007/baseline_wide_sweep/profile_tables/pilot_smoke_measured_profiles_batch007.csv'
fixture = prior / 'fixtures/batch007.jsonl'
output_dir = Path(sys.argv[2]).resolve()
budgets = [15, 20, 25, 30, 35, 50]
policies = {'full_lru', 'static_best', 'static_safe', 'uncalibrated_dynamic', 'utility_dynamic'}
config = load_config(base)
assert set(config['policies']['names']) == policies, 'Baseline policy set changed'
assert fixture.exists(), 'Prior batch007 fixture missing'
assert Path(config['data']['requests']).resolve() == fixture.resolve(), 'Fixture path changed'
with fixture.open(encoding='utf-8') as handle:
    requests = [json.loads(line) for line in handle if line.strip()]
assert len(requests) == 10, f'Expected 10 original requests, found {len(requests)}'
measurements = read_measurements(measurements_path)
validate_profile_measurements(measurements, str(measurements_path), required_profiles=config['profiles']['names'], require_measured=True, experiment_type=config_experiment_type(config))
profile_names = set(config['profiles']['names'])
request_keys = {(row.session_id, row.turn_index, row.request_id) for row in measurements}
assert len(request_keys) == 10, f'Expected 10 measured requests, found {len(request_keys)}'
assert all({row.profile for row in measurements if (row.session_id, row.turn_index, row.request_id) == key} == profile_names for key in request_keys), 'Missing profile for a request'
assert config.get('data', {}).get('diagnostic_only') is True, 'Prior diagnostic provenance missing'
config['pilot']['epsilons'] = [0.1]
config['pilot']['deltas'] = [0.1]
config['pilot']['memory_budgets_mib'] = budgets
config['data']['split_seed'] = 20260906
config['run_dir'] = str(output_dir)
config['output']['run_dir'] = str(output_dir)
config['session_trace']['memory_budgets_mib'] = budgets
config['outputs']['smoke_policy'] = str(output_dir / 'policy_tables/pilot_smoke_measured_policy.csv')
config_path = output_dir / 'coarse_config.yaml'
config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
loaded = load_config(config_path)
assert configured_memory_budgets(loaded['pilot']) == [float(value) for value in budgets]
assert loaded['data']['split_seed'] == 20260906
assert loaded['data']['requests'] == str(fixture)
assert loaded['data']['max_requests'] == 10
assert len(loaded['profiles']['names']) == 8
report = {'config': str(config_path), 'fixture': str(fixture), 'measurements': str(measurements_path), 'request_count': len(request_keys), 'fixture_request_count': len(requests), 'profile_count': len(profile_names), 'measurement_count': len(measurements), 'budgets_mib': budgets, 'epsilon': 0.1, 'delta': 0.1, 'split_seed': 20260906, 'policies': list(config['policies']['names']), 'backend': 'measured_replay', 'diagnostic_only': True}
(output_dir / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
