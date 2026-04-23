import os
import sys
import json
import re
import time
import warnings
import argparse

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from LCGS_core.normalization import normalize_code_semantics
from LCGS_core.augmentation import transitivity_augmentation
from LCGS_core.tptf import compute_tptf_vector, compute_corpus_idf, get_ast_helpers
from LCGS_core.cwj import ast_to_networkx, compute_cwj, check_mcs_equal

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

class PipelineLCGS:

    def __init__(self, ratio, root, bench, dataset_path, lang='java',
                 no_r1=False, no_r2=False, no_r3=False,
                 cwj_alpha=0.5, cwj_beta=0.5):
        self.dataset = dataset_path
        self.ratio = ratio
        self.root = root
        self.bench = bench
        self.lang = lang
        self.no_r1 = no_r1
        self.no_r2 = no_r2
        self.no_r3 = no_r3
        self.cwj_alpha = cwj_alpha
        self.cwj_beta = cwj_beta
        self.sources = None
        self.blocks = None
        self.pairs = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    def jsonl_to_df(self, jf):
        with open(jf, 'r', encoding='utf-8') as json_file:
            json_list = list(json_file)
        x = []
        for json_str in json_list:
            result = json.loads(json_str)
            x.append([result['idx'], result['func']])
        return pd.DataFrame(x)

    def parse_source(self, output_file, option):
        path = self.root + '/' + output_file
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if self.lang == 'c':
            parse_program = _make_c_parser(apply_r2=not self.no_r2)
            msg = "pycparser (C)" + ("" if not self.no_r2 else " [R2 disabled]")
            print(f"Parsing with {msg}...")
        else:
            parse_program = _make_java_parser(apply_r2=not self.no_r2)
            msg = "javalang (Java)" + ("" if not self.no_r2 else " [R2 disabled]")
            print(f"Parsing with {msg}...")

        jf = self.dataset + '/data.jsonl'
        source = self.jsonl_to_df(jf)
        source.columns = ['idx', 'func']
        print(f"  Total: {len(source)} functions")
        source['func'] = source['func'].apply(parse_program)
        before = len(source)
        source = source.dropna(subset=['func'])
        print(f"  Parsed: {before} -> {len(source)} (dropped {before - len(source)})")
        source.to_pickle(path)
        self.sources = source
        return source

    def check_or_create(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def read_data(self):
        dataset_path = self.dataset + '/'
        k = ['id1', 'id2', 'label']
        train = pd.read_csv(dataset_path + 'train.csv', names=k)
        dev = pd.read_csv(dataset_path + 'valid.csv', names=k)
        test = pd.read_csv(dataset_path + 'test.csv', names=k)

        if not self.no_r1:
            print("Applying R1 transitivity augmentation...")
            train = transitivity_augmentation(train, limit_per_type=1000)
        else:
            print("Skipping R1 transitivity augmentation (--no_r1)")
            if 'weight' not in train.columns:
                train['weight'] = 1.0

        self.pairs = pd.concat([train, test, dev])

        data_path = self.root + self.bench + '/'

        train_path = data_path + 'train/'
        self.check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = data_path + 'dev/'
        self.check_or_create(dev_path)
        self.dev_file_path = dev_path + 'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = data_path + 'test/'
        self.check_or_create(test_path)
        self.test_file_path = test_path + 'test_.pkl'
        test.to_pickle(self.test_file_path)

    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        data_path = self.root + self.bench + '/'
        if not input_file:
            input_file = self.train_file_path

        pairs = pd.read_pickle(input_file)
        train_ids = pd.concat([pairs['id1'], pairs['id2']]).unique()
        self.sources['idx'] = self.sources['idx'].astype(int)

        valid_source_ids = set(self.sources['idx'].values)
        filtered_train_ids = [tid for tid in train_ids if tid in valid_source_ids]
        if len(filtered_train_ids) == 0:
            raise ValueError("No valid training IDs found.")

        trees = self.sources.set_index('idx', drop=False).loc[filtered_train_ids]

        embedding_dir = data_path + 'train/embedding'
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)

        if self.lang == 'c':
            from LCGS_core.tptf import get_ast_token_c, get_ast_children_c

            def get_sequence_c(node, sequence):
                sequence.append(get_ast_token_c(node))
                for child in get_ast_children_c(node):
                    get_sequence_c(child, sequence)

            def trans_to_sequences(ast):
                sequence = []

                if hasattr(ast, 'ext'):
                    for ext in ast.ext:
                        get_sequence_c(ext, sequence)
                else:
                    get_sequence_c(ast, sequence)
                return sequence
        else:
            from utils import get_sequence as func

            def trans_to_sequences(ast):
                sequence = []
                func(ast, sequence)
                return sequence

        corpus = trees['func'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['func'] = pd.Series(str_corpus)

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, vector_size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(embedding_dir + '/node_w2v_' + str(size))

    def generate_block_seqs(self):
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(
            self.root + self.bench + '/train/embedding/node_w2v_' + str(self.size)
        ).wv
        vocab = word2vec.key_to_index
        max_token = word2vec.vectors.shape[0]

        get_token, get_children = get_ast_helpers(self.lang)

        if self.lang == 'c':

            def tree_to_index(node):
                token = get_token(node)
                result = [vocab[token] if token in vocab else max_token]
                children = get_children(node)
                for child in children:
                    result.append(tree_to_index(child))
                return result

            def trans2seq(r):
                if r is None:
                    return []

                if hasattr(r, 'ext'):
                    tree = []
                    for ext_node in r.ext:
                        tree.append(tree_to_index(ext_node))
                    return tree if tree else [tree_to_index(r)]
                return [tree_to_index(r)]
        else:

            def tree_to_index(node):
                token = get_token(node)
                if token in ['long', 'short', 'byte']:
                    token = 'int'
                elif token in ['double']:
                    token = 'float'
                elif token in ['protected', 'private']:
                    token = 'public'
                result = [vocab[token] if token in vocab else max_token]
                children = get_children(node)
                for child in children:
                    result.append(tree_to_index(child))
                return result

            from utils import get_blocks_v1 as func

            def trans2seq(r):
                blocks = []
                if r is None:
                    return []
                try:
                    func(r, blocks)
                except Exception:
                    pass
                if len(blocks) == 0:
                    blocks = [r]
                tree = []
                for b in blocks:
                    btree = tree_to_index(b)
                    tree.append(btree)
                return tree

        trees = pd.DataFrame(self.sources, copy=True)
        trees['func'] = trees['func'].apply(trans2seq)
        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    def merge(self, data_path, part):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='idx')
        df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='idx')
        df.drop(['idx_x', 'idx_y'], axis=1, inplace=True)
        df.dropna(inplace=True)
        df.to_pickle(self.root + self.bench + '/' + part + '/blocks.pkl')

    def calculate_lcgs_features(self, split_file, idf_dict, avg_doc_len):
        pairs = pd.read_pickle(split_file)

        if 'cwj' in pairs.columns and 'mcs_equal' in pairs.columns:
            print(f"Skipping {split_file}: Features already exist.")
            return

        print(f"Computing LCGS features for {split_file}...")

        ast_map = self.sources.set_index('idx')['func'].to_dict()
        unique_ids = set(pairs['id1']) | set(pairs['id2'])
        graph_cache = {}

        for uid in tqdm(unique_ids, desc="Building Graphs"):
            if uid in ast_map:
                ast = ast_map[uid]
                weights = compute_tptf_vector(ast, idf_dict, avg_doc_len, lang=self.lang)
                G = ast_to_networkx(ast, weights, lang=self.lang, use_compatibility=not self.no_r3)
                graph_cache[uid] = G

        cwj_list = []
        mcs_equal_list = []

        for idx, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Computing CWJ/MCS"):
            id1, id2 = int(row['id1']), int(row['id2'])
            G1 = graph_cache.get(id1)
            G2 = graph_cache.get(id2)

            if G1 is None or G2 is None:
                cwj_list.append(0.0)
                mcs_equal_list.append(0)
                continue

            cwj = compute_cwj(G1, G2, alpha=self.cwj_alpha, beta=self.cwj_beta, iterations=2)
            is_sub = check_mcs_equal(G1, G2, iterations=2)

            cwj_list.append(cwj)
            mcs_equal_list.append(is_sub)

        pairs['cwj'] = cwj_list
        pairs['mcs_equal'] = mcs_equal_list
        pairs.to_pickle(split_file)
        print(f"Saved LCGS features to {split_file}")

    def run(self):
        print('=' * 60)
        print(f'ASTNN LCGS Pipeline  [lang={self.lang}]')
        flags = []
        if self.no_r1: flags.append('no_r1')
        if self.no_r2: flags.append('no_r2')
        if self.no_r3: flags.append('no_r3')
        if flags:
            print(f'  Disabled rules: {", ".join(flags)}')
        print('=' * 60)

        pipeline_start = time.time()

        t0 = time.time()
        print('\n[1/7] Parsing source with R2 normalization...')
        self.parse_source(output_file=self.bench + '/ast.pkl', option='existing')
        print(f'  Step 1 time: {time.time() - t0:.2f}s')

        t0 = time.time()
        print('\n[2/7] Reading pairs and applying R1 augmentation...')
        self.read_data()
        print(f'  Step 2 time: {time.time() - t0:.2f}s')

        t0 = time.time()
        print('\n[3/7] Computing corpus IDF statistics...')
        idf_dict, avg_doc_len = compute_corpus_idf(self.sources, lang=self.lang)
        print(f"  Corpus: {len(self.sources)} docs, avg_len={avg_doc_len:.1f}, vocab={len(idf_dict)}")
        print(f'  Step 3 time: {time.time() - t0:.2f}s')

        t0 = time.time()
        print('\n[4/7] Training word embeddings...')
        self.dictionary_and_embedding(None, 128)
        print(f'  Step 4 time: {time.time() - t0:.2f}s')

        t0 = time.time()
        print('\n[5/7] Computing LCGS features (CWJ & MCS)...')
        self.calculate_lcgs_features(self.train_file_path, idf_dict, avg_doc_len)
        self.calculate_lcgs_features(self.dev_file_path, idf_dict, avg_doc_len)
        self.calculate_lcgs_features(self.test_file_path, idf_dict, avg_doc_len)
        print(f'  Step 5 time: {time.time() - t0:.2f}s')

        t0 = time.time()
        print('\n[6/7] Generating block sequences...')
        self.generate_block_seqs()
        print(f'  Step 6 time: {time.time() - t0:.2f}s')

        t0 = time.time()
        print('\n[7/7] Merging pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')
        print(f'  Step 7 time: {time.time() - t0:.2f}s')

        print(f'\nPipeline complete! Total time: {time.time() - pipeline_start:.2f}s')

DATASET_CONFIG = {
    'bcb': {'lang': 'java', 'bench_prefix': 'astnn_bcb'},
    'poj': {'lang': 'c', 'bench_prefix': 'astnn_poj'},
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASTNN LCGS Data Pipeline')
    parser.add_argument('--dataset', default='bcb', choices=['bcb', 'poj'],
                        help='Dataset: bcb (Java) or poj (C)')
    parser.add_argument('--data_dir', default=None,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for processed data')
    parser.add_argument('--no_r1', action='store_true',
                        help='Disable R1 transitivity augmentation')
    parser.add_argument('--no_r2', action='store_true',
                        help='Disable R2 semantic normalization')
    parser.add_argument('--no_r3', action='store_true',
                        help='Disable R3 type compatibility in CWJ')
    parser.add_argument('--cwj_alpha', type=float, default=0.5,
                        help='CWJ weight for structural Jaccard (J)')
    parser.add_argument('--cwj_beta', type=float, default=0.5,
                        help='CWJ weight for weighted Jaccard (WJ)')
    args = parser.parse_args()

    from pathlib import Path
    project_root = str(Path(__file__).resolve().parents[3])

    cfg = DATASET_CONFIG[args.dataset]
    lang = cfg['lang']
    bench = cfg['bench_prefix']

    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', args.dataset)
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), 'data_LCGS') + '/'

    sys.setrecursionlimit(10000)

    print(f"Dataset: {args.dataset} ({lang})")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")

    ppl = PipelineLCGS(
        ratio=None,
        root=args.output_dir,
        bench=bench,
        dataset_path=args.data_dir,
        lang=lang,
        no_r1=args.no_r1,
        no_r2=args.no_r2,
        no_r3=args.no_r3,
        cwj_alpha=args.cwj_alpha,
        cwj_beta=args.cwj_beta,
    )
    ppl.run()
