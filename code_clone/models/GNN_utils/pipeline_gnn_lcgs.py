import os
import sys
import math
import json
import re
import time
import shutil
import pickle
import hashlib
import argparse
import multiprocessing
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import networkx as nx
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from LCGS_core.normalization import normalize_code_semantics
from LCGS_core.augmentation import transitivity_augmentation
from LCGS_core.tptf import get_type_tiers, TYPE_PARAMS, get_ast_helpers, collect_token_depths
from LCGS_core.cwj import compute_pair_features
from LCGS_core.tptf import compute_corpus_idf

_worker_ast_map = None
_worker_idf_dict = None
_worker_avg_doc_len = None
_worker_lang = None
_worker_use_compat = True
_worker_cwj_alpha = 0.5
_worker_cwj_beta = 0.5

def _init_worker(ast_map, idf_dict, avg_doc_len, lang, use_compat=True,
                 cwj_alpha=0.5, cwj_beta=0.5):
    global _worker_ast_map, _worker_idf_dict, _worker_avg_doc_len
    global _worker_lang, _worker_use_compat, _worker_cwj_alpha, _worker_cwj_beta
    _worker_ast_map = ast_map
    _worker_idf_dict = idf_dict
    _worker_avg_doc_len = avg_doc_len
    _worker_lang = lang
    _worker_use_compat = use_compat
    _worker_cwj_alpha = cwj_alpha
    _worker_cwj_beta = cwj_beta

def _worker_compute_features(args):
    id1, id2 = args
    ast1 = _worker_ast_map.get(int(id1))
    ast2 = _worker_ast_map.get(int(id2))
    return compute_pair_features(
        ast1, ast2, _worker_idf_dict, _worker_avg_doc_len, lang=_worker_lang,
        use_compatibility=_worker_use_compat,
        cwj_alpha=_worker_cwj_alpha, cwj_beta=_worker_cwj_beta
    )

def _make_java_parser(apply_r2=True):
    import javalang

    def parse_program(text):
        try:
            if apply_r2:
                text = normalize_code_semantics(text, lang='java')
            tokens = javalang.tokenizer.tokenize(text)
            parser = javalang.parse.Parser(tokens)
            tree = parser.parse_member_declaration()
            return tree
        except:
            try:
                tokens = javalang.tokenizer.tokenize(text)
                tree = javalang.parser.parse(tokens)
                return tree
            except:
                return None

    return parse_program

def _make_c_parser(apply_r2=True):
    from pycparser import c_parser

    def clean_c_code(code):
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.S)
        return code.strip()

    def parse_program(text):
        parser = c_parser.CParser()
        try:
            if apply_r2:
                text = normalize_code_semantics(text, lang='c')
            text = clean_c_code(text)
            if not text.strip():
                return None
            return parser.parse(text)
        except:
            try:
                text = "void func() { " + text + " }"
                return parser.parse(text)
            except:
                return None

    return parse_program

def create_shadow_dataset(original_path, target_path, aug_limit, no_r1=False):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    print(f"Creating augmented dataset in {target_path}...")

    orig_train = os.path.join(original_path, 'train.csv')
    target_train = os.path.join(target_path, 'train.csv')

    df = pd.read_csv(orig_train, header=None)
    df = df.iloc[:, :3]
    df.columns = ['id1', 'id2', 'label']

    if not no_r1:
        t0 = time.time()
        aug_df = transitivity_augmentation(df, limit_per_type=aug_limit)
        print(f"  R1 augmentation: {time.time() - t0:.2f}s ({len(df)} -> {len(aug_df)} pairs)")
    else:
        print("  Skipping R1 transitivity augmentation (--no_r1)")
        aug_df = df
    aug_df[['id1', 'id2', 'label']].to_csv(target_train, header=False, index=False)

    for f in ['valid.csv', 'test.csv']:
        src = os.path.join(original_path, f)
        dst = os.path.join(target_path, f)
        if os.path.exists(src):
            shutil.copy(src, dst)

    json_src = os.path.join(original_path, 'data.jsonl')
    json_dst = os.path.join(target_path, 'data.jsonl')
    if os.path.exists(json_dst):
        os.remove(json_dst)
    if os.path.exists(json_src):
        try:
            os.symlink(os.path.abspath(json_src), json_dst)
            print(f"Symlinked data.jsonl")
        except OSError:
            shutil.copy(json_src, json_dst)
            print(f"Copied data.jsonl")

def run_pipeline(args):
    source_dir = args.data_path.rstrip('/')
    target_dir = source_dir + "_LCGS"
    lang = args.lang
    no_r1 = getattr(args, 'no_r1', False)
    no_r2 = getattr(args, 'no_r2', False)
    no_r3 = getattr(args, 'no_r3', False)
    cwj_alpha = getattr(args, 'cwj_alpha', 0.5)
    cwj_beta = getattr(args, 'cwj_beta', 0.5)

    print(f"{'=' * 60}")
    print(f"GNN LCGS Pipeline  [lang={lang}]")
    flags = []
    if no_r1: flags.append('no_r1')
    if no_r2: flags.append('no_r2')
    if no_r3: flags.append('no_r3')
    if flags:
        print(f"  Disabled rules: {', '.join(flags)}")
    print(f"{'=' * 60}")

    pipeline_start = time.time()

    print(f"\n[1/5] Creating augmented dataset...")
    create_shadow_dataset(source_dir, target_dir, args.aug_limit, no_r1=no_r1)

    print(f"\n[2/5] Loading and parsing source code...")
    t_parse_start = time.time()
    jsonl_path = os.path.join(target_dir, 'data.jsonl')

    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)

    data_df = pd.DataFrame(records)

    r2_tag = "" if not no_r2 else " [R2 disabled]"
    if lang == 'c':
        parse_fn = _make_c_parser(apply_r2=not no_r2)
        print(f"  Parsing {len(data_df)} functions with pycparser (C) + R2 normalization{r2_tag}...")
    else:
        parse_fn = _make_java_parser(apply_r2=not no_r2)
        print(f"  Parsing {len(data_df)} functions with javalang (Java) + R2 normalization{r2_tag}...")

    data_df['ast'] = data_df['func'].apply(parse_fn)
    before = len(data_df)
    data_df = data_df.dropna(subset=['ast'])
    print(f"  Parsed: {before} -> {len(data_df)} (dropped {before - len(data_df)})")
    print(f"  Parsing time: {time.time() - t_parse_start:.2f}s")

    ast_map = dict(zip(data_df['idx'].astype(int), data_df['ast']))

    print(f"\n[3/5] Computing corpus IDF statistics...")
    sources_df = pd.DataFrame({
        'idx': data_df['idx'].astype(int),
        'func': data_df['ast']
    })
    idf_dict, avg_doc_len = compute_corpus_idf(sources_df, lang=lang)
    print(f"  Vocab: {len(idf_dict)}, Avg doc len: {avg_doc_len:.1f}")

    use_compat = not no_r3
    r3_tag = "" if use_compat else " [R3 disabled]"
    print(f"\n[4/5] Computing CWJ/MCS features{r3_tag}...")
    t_cwj_start = time.time()
    feature_storage = {}
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"  Using {num_cores} cores")

    for split in ['train', 'valid', 'test']:
        csv_path = os.path.join(target_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            continue

        print(f"  Processing {split}...")
        pairs = pd.read_csv(csv_path, header=None)
        pairs.columns = ['id1', 'id2', 'label'][:len(pairs.columns)]

        tasks = [(int(row['id1']), int(row['id2'])) for _, row in pairs.iterrows()]

        try:
            with multiprocessing.Pool(
                processes=num_cores,
                initializer=_init_worker,
                initargs=(ast_map, idf_dict, avg_doc_len, lang, use_compat,
                          cwj_alpha, cwj_beta)
            ) as pool:
                split_features = list(tqdm(
                    pool.imap(_worker_compute_features, tasks),
                    total=len(tasks), desc=f"  {split}"
                ))
        except Exception as e:
            print(f"  Multiprocessing failed ({e}), falling back to serial...")
            _init_worker(ast_map, idf_dict, avg_doc_len, lang, use_compat,
                         cwj_alpha, cwj_beta)
            split_features = [_worker_compute_features(t) for t in tqdm(tasks, desc=f"  {split}")]

        feature_storage[split] = split_features
    print(f"  CWJ/MCS computation: {time.time() - t_cwj_start:.2f}s")

    output_file = os.path.join(target_dir, 'gnn_lcgs_features.pkl')
    print(f"\n[5/5] Saving features to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(feature_storage, f)

    pipeline_total = time.time() - pipeline_start
    print(f"\nPipeline complete! Total time: {pipeline_total:.2f}s")
    print(f"  Original: {source_dir}")
    print(f"  LCGS:     {target_dir}")
    print(f"  Features: {output_file}")

    for split, feats in feature_storage.items():
        cwj_vals = [f[0] for f in feats]
        mcs_vals = [f[1] for f in feats]
        print(f"  {split}: {len(feats)} pairs, "
              f"avg CWJ={sum(cwj_vals)/len(cwj_vals):.4f}, "
              f"MCS=1 count={sum(mcs_vals)}")

DATASET_CONFIG = {
    'bcb': {'lang': 'java'},
    'poj': {'lang': 'c'},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN LCGS Feature Pipeline')
    parser.add_argument("--dataset", default='bcb', choices=['bcb', 'poj'],
                        help="Dataset: bcb (Java) or poj (C)")
    parser.add_argument("--data_path", default=None,
                        help="Path to original dataset directory (e.g., data/bcb/)")
    parser.add_argument("--aug_limit", type=int, default=1000,
                        help="Max augmented samples per label type")
    parser.add_argument("--no_r1", action='store_true',
                        help="Disable R1 transitivity augmentation")
    parser.add_argument("--no_r2", action='store_true',
                        help="Disable R2 semantic normalization")
    parser.add_argument("--no_r3", action='store_true',
                        help="Disable R3 type compatibility in CWJ")
    parser.add_argument("--cwj_alpha", type=float, default=0.5,
                        help="CWJ weight for structural Jaccard (J)")
    parser.add_argument("--cwj_beta", type=float, default=0.5,
                        help="CWJ weight for weighted Jaccard (WJ)")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    args.lang = cfg['lang']

    if args.data_path is None:
        project_root = str(Path(__file__).resolve().parents[3])
        args.data_path = os.path.join(project_root, 'data', args.dataset)

    print(f"Dataset: {args.dataset} ({args.lang})")
    print(f"Data: {args.data_path}")

    run_pipeline(args)
