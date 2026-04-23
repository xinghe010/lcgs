#!/usr/bin/env python3

import os
import re
import json
import argparse
from tqdm import tqdm

def tokenize(text):
    tokens = []
    i = 0
    n = len(text)
    while i < n:

        if text[i].isspace():
            i += 1

        elif i + 2 < n and text[i:i+3] in ('<=>', '<~>'):
            tokens.append(text[i:i+3])
            i += 3

        elif i + 1 < n and text[i:i+2] in ('=>', '!='):
            tokens.append(text[i:i+2])
            i += 2

        elif text[i] in '()[]~&|,:=!?.':
            tokens.append(text[i])
            i += 1

        elif text[i].isalnum() or text[i] in '_$':
            j = i
            while j < n and (text[j].isalnum() or text[j] in '_$'):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            i += 1
    return tokens

class FOFAnalyzer:

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.literal_count = 0
        self.var_occurrences = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def analyze(self):
        try:
            self._parse_formula()
        except Exception:
            pass
        return self.literal_count, self.var_occurrences

    def _parse_formula(self):
        self._parse_unary()
        while self.peek() in ('=>', '<=>', '&', '|', '<~>'):
            self.consume()
            self._parse_unary()

    def _parse_unary(self):
        tok = self.peek()
        if tok == '~':
            self.consume()
            self._parse_unary()
        elif tok in ('!', '?'):
            self.consume()
            self.consume()

            while self.peek() not in (']', None):
                self.consume()
            self.consume()
            self.consume()
            self._parse_unary()
        elif tok == '(':
            self.consume()
            self._parse_formula()
            if self.peek() == ')':
                self.consume()
        elif tok is not None:
            self._parse_atom()

    def _parse_atom(self):
        self.literal_count += 1
        self._parse_term()

        if self.peek() in ('=', '!='):
            self.consume()
            self._parse_term()

    def _parse_term(self):
        tok = self.consume()
        if tok is None:
            return

        if tok[0].isupper():
            self.var_occurrences += 1

        if self.peek() == '(':
            self.consume()
            if self.peek() != ')':
                self._parse_term()
                while self.peek() == ',':
                    self.consume()
                    self._parse_term()
            if self.peek() == ')':
                self.consume()

def extract_formula_body(fof_line):
    line = fof_line.strip()

    if line.endswith(').'):
        line = line[:-2]
    elif line.endswith(')'):
        line = line[:-1]

    if not line.startswith('fof('):
        return None

    comma_count = 0
    i = 4
    while i < len(line) and comma_count < 2:
        if line[i] == ',':
            comma_count += 1
        i += 1

    if comma_count < 2:
        return None

    return line[i:].strip()

def analyze_problem_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    fof_statements = []
    buffer = []

    for line in content.split('\n'):

        if '%' in line:
            line = line[:line.index('%')]
        line = line.strip()
        if not line:
            continue
        buffer.append(line)
        if line.endswith('.'):
            full = ' '.join(buffer)
            buffer = []
            if full.startswith('fof('):
                fof_statements.append(full)

    num_clauses = len(fof_statements)
    total_literals = 0
    total_var_terms = 0
    max_clause_literals = 0

    for stmt in fof_statements:
        formula = extract_formula_body(stmt)
        if formula is None:
            continue

        tokens = tokenize(formula)
        analyzer = FOFAnalyzer(tokens)
        literals, var_terms = analyzer.analyze()

        total_literals += literals
        total_var_terms += var_terms
        max_clause_literals = max(max_clause_literals, literals)

    return {
        'num_clauses': num_clauses,
        'total_literals': total_literals,
        'total_var_terms': total_var_terms,
        'max_clause_literals': max_clause_literals
    }

def main():
    parser = argparse.ArgumentParser(
        description='MPTP2078 problem statistics analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--chainy_dir', type=str,
                        default='./dataset/MPTP2078-master/chainy',
                        help='Chainy problem directory')
    parser.add_argument('--output', type=str,
                        default='./data/problem_statistics.json',
                        help='Output file path')
    args = parser.parse_args()

    problem_files = sorted([f for f in os.listdir(args.chainy_dir) if f.endswith('.p')])
    print(f"Total {len(problem_files)} problem files")

    all_stats = {}

    for filename in tqdm(problem_files, desc="Analyzing problems"):

        prob_name = filename[:-2]
        if '__' in prob_name:
            short_name = prob_name.split('__', 1)[1]
        else:
            short_name = prob_name

        file_path = os.path.join(args.chainy_dir, filename)
        stats = analyze_problem_file(file_path)
        all_stats[short_name] = stats

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_stats, f, indent=2)

    n = len(all_stats)
    clauses = [s['num_clauses'] for s in all_stats.values()]
    literals = [s['total_literals'] for s in all_stats.values()]
    var_terms = [s['total_var_terms'] for s in all_stats.values()]
    max_lits = [s['max_clause_literals'] for s in all_stats.values()]

    print(f"\n{'='*60}")
    print(f"MPTP2078 problem statistics summary ({n} problems)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Min':>8} {'Max':>8} {'Avg':>10} {'Total':>12}")
    print(f"{'-'*60}")
    print(f"{'Clauses':<18} {min(clauses):>8} {max(clauses):>8} {sum(clauses)/n:>10.1f} {sum(clauses):>12}")
    print(f"{'Literals':<18} {min(literals):>8} {max(literals):>8} {sum(literals)/n:>10.1f} {sum(literals):>12}")
    print(f"{'Variable terms':<16} {min(var_terms):>8} {max(var_terms):>8} {sum(var_terms)/n:>10.1f} {sum(var_terms):>12}")
    print(f"{'Max literals/clause':<14} {min(max_lits):>8} {max(max_lits):>8} {sum(max_lits)/n:>10.1f} {'':>12}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
