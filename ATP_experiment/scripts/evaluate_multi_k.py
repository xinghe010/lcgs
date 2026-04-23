#!/usr/bin/env python3

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_fixed_k import evaluate_fixed_k_single
from evaluate_simple import parse_problem_file
from run import find_problem_file, split_list, parse_split

def main():
    parser = argparse.ArgumentParser(description='Multi-K workflow evaluation')
    parser.add_argument('--premise_rankings', type=str, required=True)
    parser.add_argument('--test_problems', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k_sequence', type=str, default='256,64,128,512,1024',
                       help='K value sequence, comma separated [default: 256,64,128,512,1024]')
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--eprover_path', type=str,
                       default='eprover')
    parser.add_argument('--chainy_dir', type=str,
                       default='./dataset/MPTP2078-master/chainy/')
    parser.add_argument('--strategy', type=str, default='satauto')
    parser.add_argument('--parallel', type=int, default=8, help='Parallel processes')
    args = parser.parse_args()

    k_sequence = [int(k) for k in args.k_sequence.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.premise_rankings) as f:
        rankings = json.load(f)

    with open(args.test_problems) as f:
        test_probs = json.load(f)
    if isinstance(test_probs, dict):
        prob_names = sorted(test_probs.keys())
    else:
        prob_names = sorted(test_probs)

    print("=" * 60)
    print(f"Multi-K workflow evaluation")
    print(f"K sequence: {k_sequence}")
    print(f"Test problems: {len(prob_names)}")
    print(f"Time limit: {args.time_limit}s/K/problem")
    print(f"Parallel: {args.parallel}")
    print("=" * 60)

    all_results = {}
    proved_set = set()
    step_stats = []

    for step_idx, K in enumerate(k_sequence):
        remaining = [p for p in prob_names if p not in proved_set]
        if not remaining:
            print(f"\nAll problems proved!")
            break

        print(f"\n--- Step {step_idx+1}: K={K}, remaining {len(remaining)} ---")
        start = time.time()

        step_dir = os.path.join(args.output_dir, f"step_{step_idx+1}_K{K}")
        os.makedirs(step_dir, exist_ok=True)

        remaining_file = os.path.join(step_dir, "remaining.json")
        with open(remaining_file, 'w') as f:
            json.dump({p: [] for p in remaining}, f)

        processes = []
        n_splits = min(args.parallel, len(remaining))
        for i in range(1, n_splits + 1):
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), 'evaluate_fixed_k.py'),
                'run', '--split', f'{i}/{n_splits}', '--K', str(K),
                '--premise_rankings', args.premise_rankings,
                '--test_problems', remaining_file,
                '--output_dir', step_dir,
                '--time_limit', str(args.time_limit),
                '--strategy', args.strategy,
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p)

        for p in processes:
            p.wait()

        merge_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), 'evaluate_fixed_k.py'),
            'merge', '--output_dir', step_dir,
        ]
        subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        merged_file = os.path.join(step_dir, "fixedk_merged.json")
        if os.path.exists(merged_file):
            with open(merged_file) as f:
                step_results = json.load(f)

            new_proved = 0
            for prob, result in step_results.items():
                if result.get('proved'):
                    proved_set.add(prob)
                    new_proved += 1
                    result['proved_at_K'] = K
                    result['proved_at_step'] = step_idx + 1
                all_results[prob] = result

            elapsed = time.time() - start
            step_stats.append({
                'step': step_idx + 1,
                'K': K,
                'remaining': len(remaining),
                'new_proved': new_proved,
                'cumulative': len(proved_set),
                'time': round(elapsed, 1)
            })
            print(f"  Newly proved: {new_proved}, cumulative: {len(proved_set)}/{len(prob_names)}, elapsed: {elapsed:.0f}s")
        else:
            print(f"  [WARNING] merge failed")

    final_file = os.path.join(args.output_dir, "multi_k_results.json")
    with open(final_file, 'w') as f:
        json.dump({
            'k_sequence': k_sequence,
            'total_problems': len(prob_names),
            'total_proved': len(proved_set),
            'step_stats': step_stats,
            'results': all_results,
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Multi-K final results: {len(proved_set)}/{len(prob_names)} ({len(proved_set)/len(prob_names)*100:.1f}%)")
    print(f"{'=' * 60}")
    for s in step_stats:
        print(f"  Step {s['step']}: K={s['K']:>4}, +{s['new_proved']:>3} newly proved, cumulative {s['cumulative']}, {s['time']:.0f}s")
    print(f"Results: {final_file}")

if __name__ == '__main__':
    main()
