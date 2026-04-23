#!/usr/bin/env python3

import os
import sys
import re
import json
import time
import subprocess
import argparse
from datetime import datetime

def run_eprover(problem_file, eprover_path, time_limit=60, memory_limit=4000, strategy='none'):
    cmd = [eprover_path]

    if strategy == 'auto':
        cmd.append('--auto-schedule')
    elif strategy == 'satauto':
        cmd.append('--satauto-schedule')

    cmd.extend([
        '--free-numbers',
        '-s', '-R',
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

        return success, cpu_time, result.stdout

    except subprocess.TimeoutExpired:
        return False, time.time() - start, 'Timeout'
    except FileNotFoundError:
        return False, 0.0, f'Error: eprover not found at {eprover_path}'
    except Exception as e:
        return False, 0.0, f'Error: {e}'

def check_eprover(eprover_path):
    if not os.path.exists(eprover_path):
        print(f"ERROR: E-prover not found: {eprover_path}")
        print("Please use --eprover_path to specify the correct path")
        sys.exit(1)
    if not os.access(eprover_path, os.X_OK):
        print(f"ERROR: E-prover is not executable: {eprover_path}")
        sys.exit(1)
    print(f"E-prover: {eprover_path} [OK]")

def extract_problem_name(filename):
    name = filename[:-2] if filename.endswith('.p') else filename
    if '__' in name:
        return name.split('__', 1)[1]
    return name

def find_problem_file(problem_name, problem_dir):
    candidate = os.path.join(problem_dir, f"{problem_name}.p")
    if os.path.exists(candidate):
        return candidate

    m = re.match(r't\d+_(.+)', problem_name)
    if m:
        theory = m.group(1)
        candidate = os.path.join(problem_dir, f"{theory}__{problem_name}.p")
        if os.path.exists(candidate):
            return candidate
    return None

def parse_problem_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conj_name = conj_str = None
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
                ns = full_fof.find('(') + 1
                ne = full_fof.find(',', ns)
                name = full_fof[ns:ne].strip()
                clean = full_fof.replace(' ', '')
                if ',conjecture,' in clean:
                    conj_name, conj_str = name, full_fof
                elif ',axiom,' in clean:
                    local_axioms[name] = full_fof
            except Exception:
                continue

    return conj_name, conj_str, local_axioms

def split_list(lst, split_id, total_splits):
    n = len(lst)
    chunk_size = n // total_splits
    remainder = n % total_splits

    start = 0
    for i in range(1, split_id):
        start += chunk_size + (1 if i <= remainder else 0)
    end = start + chunk_size + (1 if split_id <= remainder else 0)

    return lst[start:end]

def parse_split(split_str):
    parts = split_str.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid split format: {split_str}, expected '1/4' or 'all/4'")
    total = int(parts[1])
    if parts[0] == 'all':
        return 'all', total
    split_id = int(parts[0])
    if split_id < 1 or split_id > total:
        raise ValueError(f"Split ID must be between 1 and {total}")
    return split_id, total

def cmd_provable(args):
    split_id, total_splits = parse_split(args.split)

    if split_id == 'all':
        check_eprover(args.eprover_path)
        launch_all_splits('provable', args, total_splits)
        return

    check_eprover(args.eprover_path)

    all_files = sorted([f for f in os.listdir(args.problem_dir) if f.endswith('.p')])
    my_files = split_list(all_files, split_id, total_splits)

    output_file = os.path.join(args.output_dir, f'provable_split_{split_id}of{total_splits}.json')

    print("=" * 60)
    print(f"Filtering provable problems - split {split_id}/{total_splits}")
    print("=" * 60)
    print(f"Total problems: {len(all_files)}")
    print(f"This split: {len(my_files)} problems")
    print(f"Time limit: {args.time_limit}s | memory: {args.memory_limit}MB | strategy: {args.strategy}")
    print(f"Output: {output_file}")
    print("=" * 60)

    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = set(results.keys())
    print(f"Completed: {len(done)}")

    proved = sum(1 for v in results.values() if v['success'])
    count = len(done)

    for i, filename in enumerate(my_files):
        prob_name = extract_problem_name(filename)
        if prob_name in done:
            continue

        prob_file = os.path.join(args.problem_dir, filename)
        success, cpu_time, output = run_eprover(
            prob_file, args.eprover_path, args.time_limit,
            args.memory_limit, args.strategy
        )

        results[prob_name] = {
            'success': success,
            'time': round(cpu_time, 2)
        }

        count += 1
        if success:
            proved += 1

        if not success and count == 1 and output.startswith('Error:'):
            print(f"\nFirst problem failed: {output}")
            print("Please check eprover path and problem file path\n")

        total = len(my_files)
        print(f"\r[{count}/{total}] proved: {proved} | "
              f"{prob_name} {'OK' if success else 'FAIL'} ({cpu_time:.1f}s)     ",
              end='', flush=True)

        if count % 20 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSplit {split_id}/{total_splits} done!")
    print(f"Processed: {len(results)} | provable: {proved}")
    print(f"Results: {output_file}")

def cmd_baseline(args):
    split_id, total_splits = parse_split(args.split)

    if split_id == 'all':
        check_eprover(args.eprover_path)
        launch_all_splits('baseline', args, total_splits)
        return

    check_eprover(args.eprover_path)

    if args.problem_list:
        with open(args.problem_list, 'r') as f:
            all_problems = json.load(f)
            if isinstance(all_problems, dict):
                all_problems = sorted(all_problems.keys())
    else:
        all_problems = sorted([extract_problem_name(f)
                               for f in os.listdir(args.problem_dir) if f.endswith('.p')])

    my_problems = split_list(all_problems, split_id, total_splits)
    output_file = os.path.join(args.output_dir, f'baseline_split_{split_id}of{total_splits}.json')

    print("=" * 60)
    print(f"Chainy Baseline - split {split_id}/{total_splits}")
    print("=" * 60)
    print(f"Total problems: {len(all_problems)} | this split: {len(my_problems)}")
    print(f"Time limit: {args.time_limit}s | memory: {args.memory_limit}MB | strategy: {args.strategy}")
    print("=" * 60)

    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = set(results.keys())
    print(f"Completed: {len(done)}")

    proved = sum(1 for v in results.values() if v['success'])
    count = len(done)

    for prob_name in my_problems:
        if prob_name in done:
            continue

        prob_file = find_problem_file(prob_name, args.problem_dir)
        if not prob_file:
            results[prob_name] = {'success': False, 'time': 0.0, 'error': 'file not found'}
            count += 1
            continue

        success, cpu_time, output = run_eprover(
            prob_file, args.eprover_path, args.time_limit,
            args.memory_limit, args.strategy
        )

        results[prob_name] = {'success': success, 'time': round(cpu_time, 2)}
        count += 1
        if success:
            proved += 1

        print(f"\r[{count}/{len(my_problems)}] proved: {proved} | "
              f"{prob_name} {'OK' if success else 'FAIL'} ({cpu_time:.1f}s)     ",
              end='', flush=True)

        if count % 20 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSplit {split_id}/{total_splits} done! Processed: {len(results)} | provable: {proved}")

def cmd_evaluate(args):
    split_id, total_splits = parse_split(args.split)

    if split_id == 'all':
        check_eprover(args.eprover_path)
        launch_all_splits('evaluate', args, total_splits)
        return

    check_eprover(args.eprover_path)

    with open(args.premise_rankings, 'r') as f:
        rankings = json.load(f)

    if args.test_problems:
        with open(args.test_problems, 'r') as f:
            all_problems = json.load(f)
            if isinstance(all_problems, dict):
                all_problems = sorted(all_problems.keys())
    else:
        all_problems = sorted(rankings.keys())

    all_problems = [p for p in all_problems if p in rankings]

    my_problems = split_list(all_problems, split_id, total_splits)
    output_file = os.path.join(args.output_dir, f'evaluate_split_{split_id}of{total_splits}.json')
    temp_dir = os.path.join(args.output_dir, f'temp_{split_id}')
    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 60)
    print(f"Premise selection evaluation - split {split_id}/{total_splits}")
    print("=" * 60)
    print(f"Total problems: {len(all_problems)} | this split: {len(my_problems)}")
    print(f"K values: {args.k_values}")
    print(f"Time limit: {args.time_limit}s | strategy: {args.strategy}")
    print("=" * 60)

    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = set(results.keys())
    print(f"Completed: {len(done)}")

    count = len(done)
    success_counts = {k: 0 for k in args.k_values}
    for v in results.values():
        for k in args.k_values:
            if v.get(str(k), {}).get('success'):
                success_counts[k] += 1

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

        ranked = [p for p in rankings[prob_name] if p in local_axioms]
        if not ranked:
            count += 1
            continue

        prob_results = {}
        proved = False

        for k in sorted(args.k_values):
            if proved:
                prob_results[str(k)] = prob_results[str(prev_k)].copy()
                continue

            top_k = ranked[:min(k, len(ranked))]

            temp_file = os.path.join(temp_dir, f"{prob_name}_k{k}.p")
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(conj_str + "\n")
                    for prem in top_k:
                        f.write(local_axioms[prem] + "\n")

                success, cpu_time, output = run_eprover(
                    temp_file, args.eprover_path, args.time_limit,
                    args.memory_limit, args.strategy
                )

                prob_results[str(k)] = {
                    'success': success,
                    'time': round(cpu_time, 2),
                    'num_premises': len(top_k)
                }

                if success:
                    proved = True
                    prev_k = k
                    success_counts[k] += 1

            except Exception as e:
                prob_results[str(k)] = {'success': False, 'time': 0.0}
            finally:
                try:
                    os.remove(temp_file)
                except:
                    pass

        results[prob_name] = prob_results
        count += 1

        status = ' '.join(f"K{k}:{'Y' if prob_results.get(str(k),{}).get('success') else 'N'}"
                         for k in args.k_values[:4])
        print(f"\r[{count}/{len(my_problems)}] {status} | {prob_name}     ",
              end='', flush=True)

        if count % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass

    print(f"\nSplit {split_id}/{total_splits} done!")
    for k in args.k_values:
        print(f"  K={k}: {success_counts[k]}/{len(results)}")

def cmd_merge(args):
    print("=" * 60)
    print("Merging split results")
    print("=" * 60)

    split_files = sorted([
        f for f in os.listdir(args.output_dir)
        if re.match(rf'{args.mode}_split_\d+of\d+\.json', f)
    ])

    if not split_files:
        print(f"No split files found (pattern: {args.mode}_split_*of*.json)")
        return

    print(f"Found {len(split_files)} split files:")
    for f in split_files:
        print(f"  {f}")

    merged = {}
    for filename in split_files:
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        merged.update(data)
        print(f"  {filename}: {len(data)} problems")

    print(f"\nTotal after merging: {len(merged)} problems")

    if args.mode in ('provable', 'baseline'):
        proved = sum(1 for v in merged.values() if v.get('success'))
        total_time = sum(v.get('time', 0) for v in merged.values())
        print(f"Provable: {proved} ({proved/max(len(merged),1)*100:.1f}%)")
        print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")

    elif args.mode == 'evaluate':
        for k in args.k_values:
            count = sum(1 for v in merged.values()
                       if v.get(str(k), {}).get('success'))
            rate = count / max(len(merged), 1) * 100
            print(f"  K={k:4d}: {count:4d}/{len(merged)} ({rate:5.1f}%)")

    output_file = os.path.join(args.output_dir, f'{args.mode}_merged.json')
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged results: {output_file}")

    if args.mode == 'provable' and args.train_deps:
        proved_names = sorted([p for p, v in merged.items() if v.get('success')])
        print(f"\nProvable problems: {len(proved_names)}")

        train_set = set()
        with open(args.train_deps, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    train_set.add(line.split(':')[0])

        test_problems = sorted([p for p in proved_names if p not in train_set])

        test_file = os.path.join(args.output_dir, 'test_problems_provable.json')
        with open(test_file, 'w') as f:
            json.dump(test_problems, f, indent=2)

        provable_file = os.path.join(args.output_dir, 'provable_problems.json')
        with open(provable_file, 'w') as f:
            json.dump(proved_names, f, indent=2)

        print(f"Training set: {len(train_set)} | test set: {len(test_problems)}")
        print(f"Test set file: {test_file}")
        print(f"Provable list: {provable_file}")

def launch_all_splits(command, args, total_splits):
    print("=" * 60)
    print(f"Launching {total_splits} splits")
    print("=" * 60)

    script_path = os.path.abspath(__file__)

    for i in range(1, total_splits + 1):

        cmd = [sys.executable, script_path, command, '--split', f'{i}/{total_splits}']

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
    print(f"After all done, run: python3 {script_path} merge --output_dir {args.output_dir} --mode {command}")

def main():
    parser = argparse.ArgumentParser(
        description='TWGNN batched parallel E-prover testing tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    p1 = subparsers.add_parser('provable', help='Filter provable problems (bushy)')
    p1.add_argument('--split', type=str, default='1/1',
                   help='Split: "1/4"=part 1 of 4, "all/4"=launch all in background')
    p1.add_argument('--problem_dir', type=str, required=True)
    p1.add_argument('--eprover_path', type=str, default='eprover')
    p1.add_argument('--output_dir', type=str, required=True)
    p1.add_argument('--train_deps', type=str, default=None,
                   help='Training deps file (only used for merge)')
    p1.add_argument('--time_limit', type=int, default=60)
    p1.add_argument('--memory_limit', type=int, default=4000)
    p1.add_argument('--strategy', type=str, default='none',
                   choices=['auto', 'satauto', 'none'])

    p2 = subparsers.add_parser('baseline', help='Chainy baseline test')
    p2.add_argument('--split', type=str, default='1/1')
    p2.add_argument('--problem_dir', type=str, required=True)
    p2.add_argument('--problem_list', type=str, default=None)
    p2.add_argument('--eprover_path', type=str, default='eprover')
    p2.add_argument('--output_dir', type=str, required=True)
    p2.add_argument('--time_limit', type=int, default=60)
    p2.add_argument('--memory_limit', type=int, default=10000)
    p2.add_argument('--strategy', type=str, default='none',
                   choices=['auto', 'satauto', 'none'])

    p3 = subparsers.add_parser('evaluate', help='Premise selection evaluation')
    p3.add_argument('--split', type=str, default='1/1')
    p3.add_argument('--chainy_dir', type=str, required=True)
    p3.add_argument('--premise_rankings', type=str, required=True)
    p3.add_argument('--test_problems', type=str, default=None)
    p3.add_argument('--eprover_path', type=str, default='eprover')
    p3.add_argument('--output_dir', type=str, required=True)
    p3.add_argument('--k_values', type=int, nargs='+', default=[64, 128, 256, 512])
    p3.add_argument('--time_limit', type=int, default=60)
    p3.add_argument('--memory_limit', type=int, default=10000)
    p3.add_argument('--strategy', type=str, default='none',
                   choices=['auto', 'satauto', 'none'])

    p4 = subparsers.add_parser('merge', help='Merge split results')
    p4.add_argument('--output_dir', type=str, required=True)
    p4.add_argument('--mode', type=str, default='provable',
                   choices=['provable', 'baseline', 'evaluate'],
                   help='Which mode of results to merge')
    p4.add_argument('--train_deps', type=str, default=None,
                   help='Training deps file (provable mode for generating test set)')
    p4.add_argument('--k_values', type=int, nargs='+', default=[64, 128, 256, 512])

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command != 'merge':
        os.makedirs(args.output_dir, exist_ok=True)

    start = datetime.now()
    print(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.command == 'provable':
        cmd_provable(args)
    elif args.command == 'baseline':
        cmd_baseline(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'merge':
        cmd_merge(args)

    print(f"\nTotal elapsed: {datetime.now() - start}")

if __name__ == '__main__':
    main()
