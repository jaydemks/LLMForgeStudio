import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


class E2EGate:
    def __init__(self, run_dir: Path, output_dir: Path):
        self.run_dir = run_dir
        self.output_dir = output_dir
        self.results: list[CheckResult] = []

    def check_exists(self, rel_path: str, name: str):
        p = self.run_dir / rel_path
        ok = p.exists()
        self.results.append(CheckResult(name, ok, f"{'OK' if ok else 'MISSING'}: {p}"))
        return ok, p

    def check_json_status(self, path: Path, key: str, expected: str, name: str):
        if not path.exists():
            self.results.append(CheckResult(name, False, f"MISSING: {path}"))
            return
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            got = str(data.get(key, ''))
            ok = got.lower() == expected.lower()
            self.results.append(CheckResult(name, ok, f"{'OK' if ok else 'FAIL'}: {key}={got}, expected={expected}"))
        except Exception as ex:
            self.results.append(CheckResult(name, False, f"PARSE ERROR: {path} ({ex})"))

    def run(self):
        # from-scratch chain
        self.check_exists('tokenizer.json', 'Tokenizer trained')
        self.check_exists('model.pt', 'Base model checkpoint')
        self.check_exists('checkpoint_manifest.json', 'Training manifest')
        self.check_exists('train_log.jsonl', 'Training logs')

        # eval chain
        self.check_exists('eval_summary.json', 'Eval summary')
        self.check_exists('release_candidate_scorecard.md', 'Release candidate scorecard')

        # export chain (from-scratch export)
        self.check_exists('exports/ollama/model.gguf', 'From-scratch GGUF export')
        self.check_exists('exports/ollama/Modelfile', 'From-scratch Modelfile')

        # fine-tune chain
        self.check_exists('fine_tuning_ollama/ollama_finetune_manifest.json', 'Fine-tune manifest')
        self.check_exists('fine_tuning_ollama/ollama_finetune_log.jsonl', 'Fine-tune logs')
        ok_handoff, handoff_path = self.check_exists('fine_tuning_ollama/exports/ollama_finetune/ollama_handoff_status.json', 'Fine-tune handoff status')
        self.check_exists('fine_tuning_ollama/exports/ollama_finetune/model.gguf', 'Fine-tune GGUF export')
        self.check_exists('fine_tuning_ollama/exports/ollama_finetune/Modelfile', 'Fine-tune Modelfile')
        self.check_exists('fine_tuning_ollama/exports/ollama_finetune/runtime_environment_snapshot.json', 'Runtime snapshot')
        self.check_exists('fine_tuning_ollama/exports/ollama_finetune/runtime_lock_profile.json', 'Runtime lock profile')

        if ok_handoff:
            self.check_json_status(handoff_path, 'status', 'ready', 'Fine-tune handoff ready state')

        passed = all(r.passed for r in self.results)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            'generatedAtUtc': utc_now(),
            'runDir': str(self.run_dir),
            'passed': passed,
            'totalChecks': len(self.results),
            'passedChecks': sum(1 for r in self.results if r.passed),
            'failedChecks': sum(1 for r in self.results if not r.passed),
            'checks': [r.__dict__ for r in self.results],
        }

        (self.output_dir / 'e2e_gate_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

        lines = [
            '# E2E Gate Result',
            '',
            f"Generated: {report['generatedAtUtc']}",
            f"Run dir: `{report['runDir']}`",
            f"Result: {'PASS' if passed else 'FAIL'}",
            f"Checks: {report['passedChecks']}/{report['totalChecks']} passed",
            '',
            '## Checks',
        ]
        for r in self.results:
            lines.append(f"- [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.message}")

        (self.output_dir / 'e2e_gate_result.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print('PASS' if passed else 'FAIL')
        return 0 if passed else 2


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def run_scenario_pack(pack_path: Path, output_dir: Path) -> int:
    if not pack_path.exists():
        raise FileNotFoundError(f"Scenario pack not found: {pack_path}")

    pack = load_json(pack_path)
    scenarios = pack.get('scenarios', [])
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario pack has no scenarios.")

    aggregate_results: list[dict[str, Any]] = []
    overall_passed = True
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, scenario in enumerate(scenarios, start=1):
        name = str(scenario.get('name', f"scenario_{idx}"))
        run_dir_str = str(scenario.get('runDir', '')).strip()
        scenario_out_rel = str(scenario.get('outputSubdir', name)).strip() or name

        if not run_dir_str:
            aggregate_results.append({
                'name': name,
                'passed': False,
                'error': 'runDir missing in scenario'
            })
            overall_passed = False
            continue

        run_dir = Path(run_dir_str)
        scenario_out = output_dir / scenario_out_rel
        gate = E2EGate(run_dir, scenario_out)
        exit_code = gate.run()
        scenario_passed = exit_code == 0
        if not scenario_passed:
            overall_passed = False

        report_path = scenario_out / 'e2e_gate_report.json'
        details = load_json(report_path) if report_path.exists() else {
            'passed': False,
            'error': f'missing scenario report: {report_path}'
        }
        aggregate_results.append({
            'name': name,
            'runDir': str(run_dir),
            'outputDir': str(scenario_out),
            'passed': scenario_passed and bool(details.get('passed', False)),
            'details': details
        })

    summary = {
        'generatedAtUtc': utc_now(),
        'scenarioPack': str(pack_path),
        'passed': overall_passed and all(r.get('passed', False) for r in aggregate_results),
        'totalScenarios': len(aggregate_results),
        'passedScenarios': sum(1 for r in aggregate_results if r.get('passed', False)),
        'failedScenarios': sum(1 for r in aggregate_results if not r.get('passed', False)),
        'scenarios': aggregate_results,
    }

    (output_dir / 'e2e_suite_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [
        '# E2E Suite Summary',
        '',
        f"Generated: {summary['generatedAtUtc']}",
        f"Scenario pack: `{summary['scenarioPack']}`",
        f"Result: {'PASS' if summary['passed'] else 'FAIL'}",
        f"Scenarios: {summary['passedScenarios']}/{summary['totalScenarios']} passed",
        '',
        '## Scenario Results',
    ]
    for sc in aggregate_results:
        lines.append(f"- [{'PASS' if sc.get('passed', False) else 'FAIL'}] {sc.get('name', 'unknown')} (`{sc.get('runDir', '')}`)")
    (output_dir / 'e2e_suite_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('PASS' if summary['passed'] else 'FAIL')
    return 0 if summary['passed'] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir')
    parser.add_argument('--scenario-pack')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    if args.scenario_pack:
        return run_scenario_pack(Path(args.scenario_pack), Path(args.output_dir))
    if args.run_dir:
        gate = E2EGate(Path(args.run_dir), Path(args.output_dir))
        return gate.run()
    raise ValueError("Specify either --run-dir or --scenario-pack.")


if __name__ == '__main__':
    raise SystemExit(main())
