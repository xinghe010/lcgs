#!/usr/bin/env python3

import os, sys, re, json, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_simple import parse_problem_file, run_eprover, extract_used_axioms
from run import find_problem_file, split_list, parse_split

STAGE_K = [32, 64, 128, 256, 512, 1024]
STAGE_TIME = 10

def evaluate_cascade_single(prob_name, conj_name, conj_str, local_axioms,
                            premise_ranking, eprover_path, temp_dir,
                            memory_limit=10000, strategy='satauto'):
    ranked = [p for p in premise_ranking if p in local_axioms]

    for stage_idx, K in enumerate(STAGE_K):
        k = min(K, len(ranked))
        selected = ranked[:k]

        temp_file = os.path.join(temp_dir, f"{prob_name}_s{stage_idx+1}.p")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(conj_str + "\n")
            for prem in selected:
                f.write(local_axioms[prem] + "\n")

        success, output, proof_time = run_eprover(
            temp_file, eprover_path, STAGE_TIME, memory_limit, strategy)

        try:
            os.remove(temp_file)
        except OSError:
            pass

        if success:
            used_axioms = extract_used_axioms(output)
            return {
                'proved': True,
                'proved_stage': stage_idx + 1,
                'proved_K': K,
                'proof_time': round(proof_time, 2),
                'used_axioms': used_axioms,
                'num_used': len(used_axioms),
                'num_selected': len(selected),
            }

    return {
        'proved': False,
        'proved_stage': -1,
        'proved_K': -1,
        'proof_time': 0.0,
        'used_axioms': [],
        'num_used': 0,
        'num_selected': 0,
    }

def cmd_run(args):
    split_id, total_splits = parse_split(args.split)

    if split_id == 'all':
        launch_all(args, total_splits)
        return

    if not os.path.exists(args.eprover_path):
        print(f"ERROR: E-prover not found: {args.eprover_path}")
        sys.exit(1)

    with open(args.premise_rankings) as f:
        rankings = json.load(f)
    with open(args.test_problems) as f:
        test_problems = json.load(f)
        if isinstance(test_problems, dict):
            test_problems = sorted(test_problems.keys())
        elif isinstance(test_problems, list):
            test_problems = sorted(test_problems)

    all_problems = [p for p in test_problems if p in rankings]
    my_problems = split_list(all_problems, split_id, total_splits)

    output_file = os.path.join(args.output_dir, f'cascade_split_{split_id}of{total_splits}.json')
    temp_dir = os.path.join(args.output_dir, f'temp_{split_id}')
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Cascade evaluation - split {split_id}/{total_splits}")
    print(f"Problems: {len(my_problems)} | K={STAGE_K} | {STAGE_TIME}s per stage")

    results = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
    done = set(results.keys())
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
        if not conj_str or not local_axioms:
            count += 1
            continue

        result = evaluate_cascade_single(
            prob_name, conj_name, conj_str, local_axioms,
            rankings[prob_name], args.eprover_path, temp_dir,
            args.memory_limit, args.strategy)

        results[prob_name] = result
        count += 1
        if result['proved']:
            proved += 1

        status = f"K={result['proved_K']}" if result['proved'] else "FAIL"
        print(f"\r[{count}/{len(my_problems)}] proved: {proved} | "
              f"{prob_name} {status}     ", end='', flush=True)

        if count % 5 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    try:
        import shutil
        shutil.rmtree(temp_dir)
    except OSError:
        pass

    print(f"Split {split_id}/{total_splits}: {proved}/{len(results)} proved")

def cmd_merge(args):
    print("Merging cascade results")

    split_files = sorted([
        f for f in os.listdir(args.output_dir)
        if re.match(r'cascade_split_\d+of\d+\.json', f)
    ])

    if not split_files:
        print("No split files found")
        return

    merged = {}
    for fn in split_files:
        with open(os.path.join(args.output_dir, fn)) as f:
            merged.update(json.load(f))
        print(f"  {fn}: {len(json.load(open(os.path.join(args.output_dir, fn))))} problems")

    proved = {k: v for k, v in merged.items() if v.get('proved')}
    total = len(merged)
    proved_count = len(proved)

    stage_counts = {}
    for v in proved.values():
        k = v.get('proved_K', -1)
        stage_counts[k] = stage_counts.get(k, 0) + 1

    spre_sum = sum(v['num_used'] / max(v['num_selected'], 1) for v in proved.values())
    spre = spre_sum / proved_count if proved_count > 0 else 0

    metrics = {
        'total_problems': total,
        'proved_count': proved_count,
        'proved_rate': round(proved_count / max(total, 1) * 100, 2),
        'stage_counts': {str(k): c for k, c in sorted(stage_counts.items())},
        'Spre': round(spre, 4),
    }

    merged_file = os.path.join(args.output_dir, 'cascade_merged.json')
    metrics_file = os.path.join(args.output_dir, 'cascade_metrics.json')
    with open(merged_file, 'w') as f:
        json.dump(merged, f, indent=2)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTotal problems: {total}")
    print(f"Proved: {proved_count} ({metrics['proved_rate']}%)")
    print(f"Proved per stage:")
    for k in STAGE_K:
        c = stage_counts.get(k, 0)
        print(f"  K={k:<5}: {c}")
    print(f"Spre: {spre:.4f}")
    print(f"\nResults: {merged_file}")
    print(f"Metrics: {metrics_file}")

def launch_all(args, total_splits):
    script_path = os.path.abspath(__file__)
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(1, total_splits + 1):
        cmd = [sys.executable, script_path, 'run', '--split', f'{i}/{total_splits}']
        for key, value in vars(args).items():
            if key in ('command', 'split'):
                continue
            if value is None:
                continue
            cmd.extend([f'--{key}', str(value)])
        log_file = os.path.join(args.output_dir, f'log_{i}of{total_splits}.txt')
        with open(log_file, 'w') as log:
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"{total_splits} splits launched")

def main():
    parser = argparse.ArgumentParser(description='Cascade ATP evaluation')
    subparsers = parser.add_subparsers(dest='command')

    p1 = subparsers.add_parser('run')
    p1.add_argument('--split', type=str, default='1/1')
    p1.add_argument('--chainy_dir', type=str,
                    default='./dataset/MPTP2078-master/chainy')
    p1.add_argument('--premise_rankings', type=str, required=True)
    p1.add_argument('--test_problems', type=str, required=True)
    p1.add_argument('--eprover_path', type=str,
                    default='eprover')
    p1.add_argument('--output_dir', type=str, required=True)
    p1.add_argument('--memory_limit', type=int, default=10000)
    p1.add_argument('--strategy', type=str, default='satauto')

    p2 = subparsers.add_parser('merge')
    p2.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == 'run':
        os.makedirs(args.output_dir, exist_ok=True)

    start = datetime.now()
    print(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'merge':
        cmd_merge(args)

    print(f"Elapsed: {datetime.now() - start}")

if __name__ == '__main__':
    main()
