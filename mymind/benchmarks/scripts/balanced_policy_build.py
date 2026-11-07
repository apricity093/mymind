"""Construct the balanced-policy candidate data and run local, model-free checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from agents.customer_policy.protocol import PublicTask, TerminalResult, VisibleTrajectory, FACT_TYPES
from rl.tasks import PrivateStandard, validate_task_pairs
from rl.grading import GRADING_REVISION, grade_result
from benchmarks.scripts.balanced_policy_design import CLAUSES, KNOWLEDGE, profile_example, fixture, reference, dump
from benchmarks.scripts.balanced_policy_candidates import classify, actual_subtype, value, SUBTYPES
from benchmarks.scripts.balanced_policy_validation import execution_status, check_distribution

REVISION = 'balanced-policy-v1-data-standards-r4-selection'
BASE = 'balanced-policy-v1-standards-v4'
UNKNOWN = '用户明确不清楚'
SPLITS = ('train', 'validation', 'test')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paths():
    root = Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    return root / 'data/agentic_rl/balanced-policy-v1', root / 'runs/balanced-policy-v1/data-build'


def public_states(task):
    from benchmarks.scripts.balanced_policy_events import read_event
    return read_event(task)[0]


def build(data):
    from benchmarks.scripts.balanced_policy_content import build as build_content
    build_content(data)


def run_logged(action):
    data, logs = paths()
    logs.mkdir(parents=True, exist_ok=True)
    for folder in ('public', 'private', 'metadata', 'acceptance'):
        (data / folder).mkdir(exist_ok=True)
    number = len(list(logs.glob('command_*_stdout.txt'))) + 1
    command = [sys.executable, str(Path(__file__).resolve()), action, '--worker']
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    done = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    for stream in ('stdout', 'stderr'):
        (logs / f'command_{number:03}_{stream}.txt').write_text(getattr(done, stream), encoding='utf-8')
    entry = dict(command=command, started=started, finished=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 exit_status=done.returncode, stdout=f'command_{number:03}_stdout.txt', stderr=f'command_{number:03}_stderr.txt')
    with (logs / 'commands.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(entry, ensure_ascii=False)+'\n')
    print(done.stdout, end=''); print(done.stderr, end='', file=sys.stderr)
    return done.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'validate', 'selection'])
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        raise SystemExit(run_logged(args.action))
    data, logs = paths()
    if args.action == 'build':
        build(data)
    elif args.action == 'selection':
        from benchmarks.scripts.balanced_policy_selection_validation import check
        check(data, logs)
    else:
        from benchmarks.scripts.balanced_policy_data_validation import validate
        validate(data, logs)


if __name__ == '__main__':
    main()
