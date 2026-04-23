import hashlib
from collections import defaultdict

import networkx as nx

from .compatibility import get_canonical_label
from .tptf import compute_tptf_vector, get_ast_helpers

def ast_to_networkx(ast_root, weight_dict, lang='java', use_compatibility=True):
    get_token, get_children = get_ast_helpers(lang)
    G = nx.DiGraph()

    nodes = []
    if isinstance(ast_root, list):
        nodes.extend(ast_root)
    elif hasattr(ast_root, 'ext'):
        nodes.extend(ast_root.ext)
    else:
        nodes.append(ast_root)

    node_map = {}

    def get_nx_id(node):
        if id(node) not in node_map:
            node_map[id(node)] = len(node_map)
        return node_map[id(node)]

    queue = list(nodes)

    for root_node in nodes:
        u_id = get_nx_id(root_node)
        token = get_token(root_node)
        label = get_canonical_label(token, lang) if use_compatibility else token
        w = weight_dict.get(token, 1.0)
        G.add_node(u_id, label=label, weight=w)

    while queue:
        u = queue.pop(0)
        u_id = get_nx_id(u)

        children = get_children(u)
        for child in children:
            v_id = get_nx_id(child)
            token = get_token(child)
            label = get_canonical_label(token, lang) if use_compatibility else token
            w = weight_dict.get(token, 1.0)

            if v_id not in G:
                G.add_node(v_id, label=label, weight=w)
            G.add_edge(u_id, v_id)
            queue.append(child)

    return G

def get_wl_features(G, iterations=2):
    node_labels = {n: d['label'] for n, d in G.nodes(data=True)}

    for _ in range(iterations):
        new_labels = {}
        for n in G.nodes():
            label = node_labels[n]
            neighbors = G.neighbors(n)
            neighbor_labels = sorted([node_labels[nbr] for nbr in neighbors])
            aggregate_str = label + "_" + "_".join(neighbor_labels)
            new_labels[n] = hashlib.md5(aggregate_str.encode('utf-8')).hexdigest()
        node_labels = new_labels

    features = defaultdict(float)
    for n, d in G.nodes(data=True):
        wl_hash = node_labels[n]
        weight = d.get('weight', 1.0)
        features[wl_hash] += weight

    return features

def compute_cwj(G1, G2, alpha=0.5, beta=0.5, iterations=2):
    feats1 = get_wl_features(G1, iterations=iterations)
    feats2 = get_wl_features(G2, iterations=iterations)

    common_hashes = set(feats1.keys()) & set(feats2.keys())

    w_mcs_sum = 0.0
    for h in common_hashes:
        w_mcs_sum += min(feats1[h], feats2[h])

    w1_sum = sum(d['weight'] for n, d in G1.nodes(data=True))
    w2_sum = sum(d['weight'] for n, d in G2.nodes(data=True))

    denom = w1_sum + w2_sum - w_mcs_sum
    wj = w_mcs_sum / (denom + 1e-9)

    tokens1 = set(d['label'] for n, d in G1.nodes(data=True))
    tokens2 = set(d['label'] for n, d in G2.nodes(data=True))
    intersect_tokens = len(tokens1 & tokens2)
    union_tokens = len(tokens1 | tokens2)
    j_idx = intersect_tokens / (union_tokens + 1e-9)

    cwj = alpha * j_idx + beta * wj
    return cwj

def check_mcs_equal(G1, G2, iterations=2, threshold=0.9):
    feats1 = get_wl_features(G1, iterations=iterations)
    feats2 = get_wl_features(G2, iterations=iterations)

    def is_subset(f_sub, f_super):
        for h, w in f_sub.items():
            if h not in f_super:
                return False
            if f_super[h] < w * threshold:
                return False
        return True

    if is_subset(feats1, feats2):
        return 1
    if is_subset(feats2, feats1):
        return 1

    return 0

def compute_pair_features(ast1, ast2, idf_dict, avg_doc_len, lang='java',
                          use_compatibility=True, cwj_alpha=0.5, cwj_beta=0.5):
    if ast1 is None or ast2 is None:
        return 0.0, 0

    weights1 = compute_tptf_vector(ast1, idf_dict, avg_doc_len, lang=lang)
    weights2 = compute_tptf_vector(ast2, idf_dict, avg_doc_len, lang=lang)

    G1 = ast_to_networkx(ast1, weights1, lang=lang, use_compatibility=use_compatibility)
    G2 = ast_to_networkx(ast2, weights2, lang=lang, use_compatibility=use_compatibility)

    cwj = compute_cwj(G1, G2, alpha=cwj_alpha, beta=cwj_beta, iterations=2)
    mcs = check_mcs_equal(G1, G2, iterations=2)

    return cwj, mcs
