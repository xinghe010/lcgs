#!/usr/bin/env python3

import os
import sys
import json
import pickle
import argparse
from difflib import SequenceMatcher
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_problems import extract_formula_body
from run import find_problem_file
from evaluate_simple import parse_problem_file

def extract_body(fof_str):
    body = extract_formula_body(fof_str)
    return body if body else fof_str

def generate_rankings(args):
    print("=" * 60)
    print("Levenshtein (SequenceMatcher) ranking generation")
    print("=" * 60)

    print("Loading statements.pkl...")
    with open(args.statements, 'rb') as f:
        statements = pickle.load(f)
    print(f"  Total formulas: {len(statements)}")

    print("Extracting formula bodies...")
    bodies = {}
    for name, fof_str in statements.items():
        bodies[name] = extract_body(fof_str)

    print("Loading test problem list...")
    with open(args.test_problems, 'r') as f:
        test_problems = json.load(f)
        if isinstance(test_problems, dict):
            test_problems = sorted(test_problems.keys())
    print(f"  Test problems: {len(test_problems)}")

    print("Generating rankings...")
    rankings = {}
    processed = 0

    for prob_name in test_problems:

        prob_file = find_problem_file(prob_name, args.chainy_dir)
        if not prob_file:
            continue

        conj_name, conj_str, local_axioms = parse_problem_file(prob_file)
        if not conj_str or not local_axioms:
            continue

        conj_body = extract_body(conj_str)

        similarities = []
        for ax_name in local_axioms:
            if ax_name in bodies:
                ax_body = bodies[ax_name]
            else:
                ax_body = extract_body(local_axioms[ax_name])

            ratio = SequenceMatcher(None, conj_body, ax_body).ratio()
            similarities.append((ax_name, ratio))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = getattr(args, 'top_k', 1024)
        rankings[prob_name] = [name for name, _ in similarities[:top_k]]

        processed += 1
        if processed % 100 == 0:
            print(f"\r  Processed: {processed}/{len(test_problems)}     ", end='', flush=True)

    print(f"\r  Processed: {processed}/{len(test_problems)}     ")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(rankings, f, indent=2)

    print(f"\nRankings saved: {args.output}")
    print(f"Problems with ranking: {len(rankings)}")

def main():
    parser = argparse.ArgumentParser(
        description='Levenshtein (SequenceMatcher) premise selection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    p1 = subparsers.add_parser('generate', help='Generate Levenshtein rankings')
    p1.add_argument('--statements', type=str,
                    default='./data/statements.pkl',
                    help='Formula strings file')
    p1.add_argument('--test_problems', type=str,
                    default='./data/test_problems.json',
                    help='Test problem list')
    p1.add_argument('--chainy_dir', type=str,
                    default='./dataset/MPTP2078-master/chainy',
                    help='Chainy problem directory')
    p1.add_argument('--output', type=str,
                    default='./results_levenshtein/rankings_levenshtein.json',
                    help='Output ranking file')
    p1.add_argument('--top_k', type=int, default=1024,
                    help='Save top_k premises [default: 1024]')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    start = datetime.now()
    print(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.command == 'generate':
        generate_rankings(args)

    print(f"\nTotal elapsed: {datetime.now() - start}")

if __name__ == '__main__':
    main()
