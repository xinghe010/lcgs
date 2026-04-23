import os
import sys
import subprocess
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

def parse_problem_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conj_name = None
    conj_str = None
    local_axioms = {}
    buffer = []

    for line in lines:

        if '%' in line:
            line = line.split('%')[0]

        line = line.strip()

        if not line or line.startswith('#'):
            continue

        buffer.append(line)

        if line.endswith('.'):
            full_fof = " ".join(buffer)
            buffer = []

            if not full_fof.startswith('fof'):
                continue

            try:

                name_start = full_fof.find('(') + 1
                name_end = full_fof.find(',', name_start)
                name = full_fof[name_start:name_end].strip()

                clean_fof_check = full_fof.replace(' ', '')

                if ',conjecture,' in clean_fof_check:
                    conj_name = name
                    conj_str = full_fof
                elif ',axiom,' in clean_fof_check:
                    local_axioms[name] = full_fof
            except Exception:
                continue

    return conj_name, conj_str, local_axioms

def run_eprover(problem_file, eprover_path, time_limit=60, memory_limit=10000, strategy='satauto'):

    eprover_cmd = [eprover_path]

    if strategy == 'auto':
        eprover_cmd.append('--auto-schedule')
    elif strategy == 'satauto':
        eprover_cmd.append('--satauto-schedule')

    eprover_cmd.extend([
        '--free-numbers',
        '-s',
        '-R',
        '-p',
        f'--cpu-limit={time_limit}',
        f'--memory-limit={memory_limit}',
        '--tstp-format',
        problem_file
    ])

    start_time = subprocess.time if hasattr(subprocess, 'time') else None
    import time
    start_time = time.time()

    try:
        result = subprocess.run(
            eprover_cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 10
        )

        end_time = time.time()
        wall_time = end_time - start_time

        success = '# Proof found!' in result.stdout or '% Proof found!' in result.stdout
        output = result.stdout

        time_match = re.search(r'[#%] Total time\s*:\s*([\d.]+)', output)
        if time_match:
            proof_time = float(time_match.group(1))
        else:

            proof_time = wall_time

        return success, output, proof_time

    except subprocess.TimeoutExpired:
        end_time = time.time()
        wall_time = end_time - start_time
        return False, 'Timeout', wall_time
    except Exception as e:
        return False, f'Error: {str(e)}', 0.0

def extract_used_axioms(eprover_output):
    used = set()

    pattern = re.compile(r"file\('[^']*',\s*(\w+)\)")
    for line in eprover_output.split('\n'):
        if 'file(' not in line:
            continue
        match = pattern.search(line)
        if match:
            name = match.group(1)

            if ', conjecture,' not in line and ',conjecture,' not in line:
                used.add(name)
    return sorted(used)

def evaluate_single_problem(prob_file, conj_name, conj_str, local_axioms,
                            eprover_path, k_values, temp_dir, premise_ranking,
                            time_limit=60, memory_limit=10000, strategy='satauto'):
    local_candidates = list(local_axioms.keys())

    if len(local_candidates) == 0:
        return {k: {'success': False, 'time': 0.0, 'used_axioms': [], 'selected': []}
                for k in k_values}

    ranked_premises = [p for p in premise_ranking if p in local_axioms]

    if not ranked_premises:
        return {k: {'success': False, 'time': 0.0, 'used_axioms': [], 'selected': []}
                for k in k_values}

    results = {}

    for k in k_values:
        top_k = min(k, len(ranked_premises))
        selected_premises = ranked_premises[:top_k]

        temp_file = os.path.join(temp_dir, f"{conj_name}_k{k}.p")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(conj_str + "\n")
            for prem in selected_premises:
                if prem in local_axioms:
                    f.write(local_axioms[prem] + "\n")

        success, output, proof_time = run_eprover(
            temp_file, eprover_path, time_limit, memory_limit, strategy
        )

        used_axioms = extract_used_axioms(output) if success else []

        results[k] = {
            'success': success,
            'time': proof_time,
            'used_axioms': used_axioms,
            'num_used': len(used_axioms),
            'num_selected': len(selected_premises),
            'num_total_axioms': len(local_axioms),
            'selected': selected_premises
        }

        try:
            os.remove(temp_file)
        except:
            pass

        if success:
            for larger_k in k_values:
                if larger_k > k and larger_k not in results:
                    results[larger_k] = results[k].copy()
            break

    for k in k_values:
        if k not in results:
            results[k] = {
                'success': False,
                'time': 0.0,
                'used_axioms': [],
                'selected': []
            }

    return results

def main():
    parser = argparse.ArgumentParser(
        description='ATP evaluation script (using precomputed rankings)'
    )

    parser.add_argument('--chainy_dir', type=str, default="./dataset/MPTP2078-master/chainy",
                       help='Chainy problem directory')
    parser.add_argument('--test_problems', type=str, default="./data/test_problems.json",
                       help='Test problem list JSON file')
    parser.add_argument('--premise_rankings', type=str, default="./results/rankings_top512.json",
                       help='Premise ranking JSON file (required)')
    parser.add_argument('--eprover_path', type=str, default="eprover",
                       help='E-prover path (use compiled release version)')
    parser.add_argument('--output_dir', type=str, default="./results/results_top_512",
                       help='Output directory')

    parser.add_argument('--k_values', type=int, nargs='+',
                       default=[512],
                       help='List of K values to test')
    parser.add_argument('--time_limit', type=int, default=60,
                       help='E-prover time limit (seconds)')
    parser.add_argument('--memory_limit', type=int, default=10000,
                       help='E-prover memory limit (MB)')
    parser.add_argument('--strategy', type=str, default='satauto',
                       choices=['auto', 'satauto', 'none'],
                       help='E-prover strategy selection')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    print("="*60)
    print("ATP evaluation (using precomputed rankings)")
    print("="*60)
    print(f"Chainy directory: {args.chainy_dir}")
    print(f"Test problems: {args.test_problems}")
    print(f"Premise ranking: {args.premise_rankings}")
    print(f"E-prover: {args.eprover_path}")
    print(f"K values: {args.k_values}")
    print(f"Time limit: {args.time_limit}s")
    print(f"Memory limit: {args.memory_limit}MB")
    print(f"Strategy: {args.strategy}")
    print("="*60)

    with open(args.test_problems, 'r') as f:
        test_problems = json.load(f)
        if isinstance(test_problems, dict):
            test_problems = list(test_problems.keys())

    with open(args.premise_rankings, 'r') as f:
        premise_rankings = json.load(f)

    print(f"\nTest problems: {len(test_problems)}")
    print(f"Problems with ranking: {len(premise_rankings)}")

    all_results = {}
    success_counts = {k: 0 for k in args.k_values}
    total_times = {k: 0.0 for k in args.k_values}

    for problem in tqdm(test_problems, desc="Evaluation progress"):

        prob_file = None
        candidates = [
            os.path.join(args.chainy_dir, f"{problem}.p"),
        ]

        match = re.match(r't\d+_(.+)', problem)
        if match:
            theory = match.group(1)
            candidates.append(os.path.join(args.chainy_dir, f"{theory}__{problem}.p"))

        for candidate in candidates:
            if os.path.exists(candidate):
                prob_file = candidate
                break

        if not prob_file:
            continue

        conj_name, conj_str, local_axioms = parse_problem_file(prob_file)

        if not conj_str or not local_axioms:
            continue

        if problem not in premise_rankings:
            continue

        results = evaluate_single_problem(
            prob_file, conj_name, conj_str, local_axioms,
            args.eprover_path, args.k_values, temp_dir,
            premise_rankings[problem],
            args.time_limit, args.memory_limit, args.strategy
        )

        all_results[problem] = results

        for k in args.k_values:
            if results[k]['success']:
                success_counts[k] += 1
                total_times[k] += results[k]['time']

    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    metrics = {}
    for k in args.k_values:
        spre_sum = 0.0
        sel_sum = 0.0
        proved_count = 0

        for prob, results in all_results.items():
            r = results[k]
            if r['success']:
                proved_count += 1
                n_uip = r.get('num_used', 0)
                n_sel = r.get('num_selected', 1)
                n_axp = r.get('num_total_axioms', 1)

                if n_sel > 0:
                    spre_sum += n_uip / n_sel
                if n_axp > 0:
                    sel_sum += n_sel / n_axp

        spre = spre_sum / proved_count if proved_count > 0 else 0.0
        sel = sel_sum / proved_count if proved_count > 0 else 0.0
        metrics[k] = {'Spre': spre, 'Sel': sel}

    metrics_file = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print("Evaluation done!")
    print("="*60)
    print(f"Problems evaluated: {len(all_results)}")
    print("\nResults per K:")
    print(f"  {'K':>5}  {'Proved':>6}  {'Rate':>8}  {'AvgTime':>8}  {'Spre':>8}  {'Sel':>8}")
    print(f"  {'-'*52}")
    for k in args.k_values:
        success_rate = (success_counts[k] / len(all_results) * 100) if all_results else 0
        avg_time = (total_times[k] / success_counts[k]) if success_counts[k] > 0 else 0
        spre = metrics[k]['Spre']
        sel = metrics[k]['Sel']
        print(f"  {k:>5}  {success_counts[k]:>5}   {success_rate:>6.1f}%  {avg_time:>7.2f}s  {spre:>8.4f}  {sel:>8.4f}")

    print(f"\nMetric descriptions:")
    print(f"  Spre (selection precision): higher is better, fraction of selected premises that are useful")
    print(f"  Sel  (selectivity):         lower is better, fraction selected from total premises")
    print(f"\nResults saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")

    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass

if __name__ == '__main__':
    main()
