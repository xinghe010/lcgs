#!/usr/bin/env python3

import os
import sys
import re
import json
import time
import subprocess
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_simple import parse_problem_file, extract_used_axioms
from run import find_problem_file, split_list, parse_split

def run_eprover_sine(problem_file, eprover_path, time_limit=60,
                     memory_limit=10000, strategy='satauto'):
    cmd = [eprover_path]

    if strategy == 'auto':
        cmd.append('--auto-schedule')
    elif strategy == 'satauto':
        cmd.append('--satauto-schedule')

    cmd.extend([
        '--sine=Auto',
        '--free-numbers',
        '-s', '-R', '-p',
        f'--cpu-limit={time_limit}',
        f'--memory-limit={memory_limit}',
        '--tstp-format',
        problem_file
    ])

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=time_limit + 30
        )
        elapsed = time.time() - start
        success = '# Proof found!' in result.stdout or '% Proof found!' in result.stdout

        m = re.search(r'[#%] Total time\s*:\s*([\d.]+)', result.stdout)
        cpu_time = float(m.group(1)) if m else elapsed

        return success, result.stdout, cpu_time

    except subprocess.TimeoutExpired:
        return False, 'Timeout', time.time() - start
    except Exception as e:
        return False, f'Error: {e}', 0.0

def cmd_run(args):
    split_id, total_splits = parse_split(args.split)

    if split_id == 'all':
        launch_all_splits(args, total_splits)
        return

    if not os.path.exists(args.eprover_path):
        print(f"ERROR: E-prover not found: {args.eprover_path}")
        sys.exit(1)

    with open(args.test_problems, 'r') as f:
        test_problems = json.load(f)
        if isinstance(test_problems, dict):
            test_problems = sorted(test_problems.keys())

    all_problems = sorted(test_problems)
    my_problems = split_list(all_problems, split_id, total_splits)

    output_file = os.path.join(args.output_dir, f'sine_split_{split_id}of{total_splits}.json')

    print("=" * 60)
    print(f"E-SInE evaluation - split {split_id}/{total_splits}")
    print("=" * 60)
    print(f"Total problems: {len(all_problems)} | this split: {len(my_problems)}")
    print(f"Time limit: {args.time_limit}s | strategy: {args.strategy}")
    print("=" * 60)

    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = set(results.keys())
    print(f"Completed: {len(done)}")

    proved = sum(1 for v in results.values() if v.get('proved'))
    count = len(done)

    for prob_name in my_problems:
        if prob_name in done:
            continue

        prob_file = find_problem_file(prob_name, args.chainy_dir)
        if not prob_file:
            count += 1
            continue

        conj_name, conj_str, local_axioms = parse_problem_file(prob_file)
        num_total_axioms = len(local_axioms) if local_axioms else 0

        success, output, proof_time = run_eprover_sine(
            prob_file, args.eprover_path, args.time_limit,
            args.memory_limit, args.strategy
        )

        used_axioms = extract_used_axioms(output) if success else []

        results[prob_name] = {
            'proved': success,
            'proof_time': round(proof_time, 2),
            'used_axioms': used_axioms,
            'num_used': len(used_axioms),
            'num_total_axioms': num_total_axioms
        }

        count += 1
        if success:
            proved += 1

        print(f"\r[{count}/{len(my_problems)}] proved: {proved} | "
              f"{prob_name} {'OK' if success else 'FAIL'} ({proof_time:.1f}s)     ",
              end='', flush=True)

        if count % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSplit {split_id}/{total_splits} done!")
    print(f"Processed: {len(results)} | proved: {proved}")

def cmd_merge(args):
    print("=" * 60)
    print("Merging E-SInE results")
    print("=" * 60)

    split_files = sorted([
        f for f in os.listdir(args.output_dir)
        if re.match(r'sine_split_\d+of\d+\.json', f)
    ])

    if not split_files:
        print("No split files found (sine_split_*of*.json)")
        return

    print(f"Found {len(split_files)} split files:")

    merged = {}
    for filename in split_files:
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        merged.update(data)
        print(f"  {filename}: {len(data)} problems")

    print(f"\nTotal after merging: {len(merged)} problems")

    proved_problems = {k: v for k, v in merged.items() if v.get('proved')}
    proved_count = len(proved_problems)
    total_count = len(merged)

    spre_sum = 0.0
    sel_sum = 0.0
    total_proof_time = 0.0

    for v in proved_problems.values():
        n_used = v.get('num_used', 0)
        n_axp = v.get('num_total_axioms', 1)

        if n_axp > 0:
            spre_sum += n_used / n_axp
            sel_sum += 1.0

        total_proof_time += v.get('proof_time', 0.0)

    spre = spre_sum / proved_count if proved_count > 0 else 0.0
    sel = sel_sum / proved_count if proved_count > 0 else 0.0
    avg_proof_time = total_proof_time / proved_count if proved_count > 0 else 0.0

    metrics = {
        'total_problems': total_count,
        'proved_count': proved_count,
        'proved_rate': round(proved_count / max(total_count, 1) * 100, 2),
        'avg_proof_time': round(avg_proof_time, 2),
        'Spre': round(spre, 4),
        'Sel': round(sel, 4)
    }

    merged_file = os.path.join(args.output_dir, 'sine_merged.json')
    with open(merged_file, 'w') as f:
        json.dump(merged, f, indent=2)

    metrics_file = os.path.join(args.output_dir, 'sine_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"E-SInE evaluation results")
    print(f"{'='*60}")
    print(f"Total problems: {total_count}")
    print(f"Proved:   {proved_count} ({metrics['proved_rate']:.1f}%)")
    print(f"Average proof time: {avg_proof_time:.2f}s")
    print(f"Spre: {spre:.4f}")
    print(f"Sel:  {sel:.4f}")
    print(f"\nMerged results: {merged_file}")
    print(f"Metrics: {metrics_file}")

def launch_all_splits(args, total_splits):
    print("=" * 60)
    print(f"Launching {total_splits} splits")
    print("=" * 60)

    script_path = os.path.abspath(__file__)
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(1, total_splits + 1):
        cmd = [sys.executable, script_path, 'run', '--split', f'{i}/{total_splits}']

        for key, value in vars(args).items():
            if key in ('command', 'split'):
                continue
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
                continue
            if isinstance(value, list):
                cmd.append(f'--{key}')
                cmd.extend(str(v) for v in value)
                continue
            cmd.extend([f'--{key}', str(value)])

        log_file = os.path.join(args.output_dir, f'log_split_{i}of{total_splits}.txt')
        print(f"  Launching split {i}/{total_splits} -> {log_file}")

        with open(log_file, 'w') as log:
            subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True
            )

    print(f"\nAll {total_splits} splits launched in background!")
    print(f"Check progress: tail -f {args.output_dir}/log_split_*")
    print(f"After all done, run: python3 {script_path} merge --output_dir {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='E-SInE benchmark evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    p1 = subparsers.add_parser('run', help='Run E-SInE evaluation')
    p1.add_argument('--split', type=str, default='1/1')
    p1.add_argument('--chainy_dir', type=str,
                    default='./dataset/MPTP2078-master/chainy')
    p1.add_argument('--test_problems', type=str,
                    default='./data/test_problems.json')
    p1.add_argument('--eprover_path', type=str,
                    default='eprover')
    p1.add_argument('--output_dir', type=str, required=True)
    p1.add_argument('--time_limit', type=int, default=60)
    p1.add_argument('--memory_limit', type=int, default=10000)
    p1.add_argument('--strategy', type=str, default='satauto',
                    choices=['auto', 'satauto', 'none'])

    p2 = subparsers.add_parser('merge', help='Merge split results')
    p2.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'run':
        os.makedirs(args.output_dir, exist_ok=True)

    start = datetime.now()
    print(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'merge':
        cmd_merge(args)

    print(f"\nTotal elapsed: {datetime.now() - start}")

if __name__ == '__main__':
    main()
