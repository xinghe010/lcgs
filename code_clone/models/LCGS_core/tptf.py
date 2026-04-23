import math
from collections import Counter, defaultdict

TYPE_TIERS_JAVA = {

    'MethodInvocation': 'high', 'ClassCreator': 'high',
    'MemberReference': 'high', 'Literal': 'high',
    'BinaryOperation': 'high', 'Cast': 'high',

    'ReferenceType': 'mid', 'BasicType': 'mid',
    'VariableDeclarator': 'mid', 'FormalParameter': 'mid',
    'Assignment': 'mid',

    'IfStatement': 'low', 'ForStatement': 'low',
    'WhileStatement': 'low', 'DoStatement': 'low',
    'SwitchStatement': 'low', 'Block': 'low',
    'ReturnStatement': 'low', 'MethodDeclaration': 'low',
    'TryStatement': 'low',
}

TYPE_TIERS_C = {

    'FuncCall': 'high', 'BinaryOp': 'high', 'UnaryOp': 'high',
    'Constant': 'high', 'ID': 'high', 'Cast': 'high',

    'Decl': 'mid', 'TypeDecl': 'mid', 'PtrDecl': 'mid',
    'ArrayDecl': 'mid', 'Typename': 'mid', 'IdentifierType': 'mid',
    'ParamList': 'mid', 'Assignment': 'mid',

    'If': 'low', 'For': 'low', 'While': 'low', 'DoWhile': 'low',
    'Switch': 'low', 'Case': 'low', 'Compound': 'low',
    'Return': 'low', 'FuncDef': 'low', 'Break': 'low', 'Continue': 'low',
}

TYPE_TIERS = TYPE_TIERS_JAVA

TYPE_PARAMS = {
    'high': {'k': 2.0, 'b': 0.75, 'bounds': (2.0, 6.0)},
    'mid': {'k': 1.2, 'b': 0.75, 'bounds': (1.0, 3.0)},
    'low': {'k': 0.5, 'b': 0.90, 'bounds': (0.5, 1.5)},
    'default': {'k': 1.2, 'b': 0.75, 'bounds': (1.0, 3.0)},
}

def get_type_tiers(lang='java'):
    if lang == 'c':
        return TYPE_TIERS_C
    return TYPE_TIERS_JAVA

def get_ast_token_java(node):
    token = getattr(node, 'token', None)
    if token is None:
        token = (getattr(node, 'name', None) or
                 getattr(node, 'value', None) or
                 getattr(node, 'declname', None) or
                 getattr(node, 'op', None) or
                 node.__class__.__name__)
    return str(token)

def get_ast_children_java(node):
    if not hasattr(node, 'children'):
        return []
    children_attr = node.children
    if callable(children_attr):
        return [c for _, c in children_attr() if c is not None]
    elif isinstance(children_attr, list):
        result = []
        for child in children_attr:
            if isinstance(child, list):
                result.extend([c for c in child if c is not None])
            elif child is not None:
                result.append(child)
        return result
    return []

def get_ast_token_c(node):
    for attr in ['name', 'value', 'declname', 'op']:
        if hasattr(node, attr):
            val = getattr(node, attr)
            if val is not None:
                return str(val)
    return str(type(node).__name__)

def get_ast_children_c(node):
    if not hasattr(node, 'children'):
        return []
    if callable(node.children):
        return [c for _, c in node.children() if c is not None]
    elif isinstance(node.children, list):
        return [c for c in node.children if c is not None]
    return []

def get_ast_helpers(lang='java'):
    if lang == 'c':
        return get_ast_token_c, get_ast_children_c
    return get_ast_token_java, get_ast_children_java

_get_ast_token = get_ast_token_java
_get_ast_children = get_ast_children_java

def collect_token_depths(node, current_depth=0, depths_dict=None, lang='java'):
    if depths_dict is None:
        depths_dict = defaultdict(list)

    get_token, get_children = get_ast_helpers(lang)

    token = get_token(node)
    depths_dict[token].append(current_depth)

    children = get_children(node)
    for child in children:
        collect_token_depths(child, current_depth + 1, depths_dict, lang=lang)

    return depths_dict

def compute_tptf_vector(ast_root, idf_dict, avg_doc_len, alpha=0.5, lang='java'):
    token_depths = collect_token_depths(ast_root, lang=lang)
    doc_len = sum(len(depths) for depths in token_depths.values())
    if doc_len == 0:
        return {}

    type_tiers = get_type_tiers(lang)
    final_weights = {}

    for token, depths in token_depths.items():
        raw_idf = idf_dict.get(token, 1.5)

        tier = type_tiers.get(token, 'default')
        params = TYPE_PARAMS.get(tier, TYPE_PARAMS['default'])
        low, high = params['bounds']
        idf_tau = max(low, min(high, raw_idf))

        tf_pos_val = 0.0
        for d in depths:
            pos_factor = 1.0 + alpha * math.log(d + 1)
            tf_pos_val += pos_factor

        k = params['k']
        b = params['b']
        numerator = tf_pos_val * (k + 1)
        denominator = tf_pos_val + k * (1 - b + b * (doc_len / avg_doc_len))

        tptf_score = idf_tau * (numerator / denominator)
        final_weights[token] = tptf_score

    return final_weights

def compute_corpus_idf(sources, lang='java'):
    get_token_fn, get_children_fn = get_ast_helpers(lang)

    doc_freq = Counter()
    total_docs = len(sources)
    total_tokens = 0

    for idx, row in sources.iterrows():
        current_tokens = set()
        count = 0

        nodes = []
        root = row['func']
        if isinstance(root, list):
            nodes.extend(root)
        elif root is not None:

            if hasattr(root, 'ext'):
                nodes.extend(root.ext)
            else:
                nodes.append(root)

        while nodes:
            node = nodes.pop()
            token_str = get_token_fn(node)
            current_tokens.add(token_str)
            count += 1
            children = get_children_fn(node)
            nodes.extend(children)

        total_tokens += count
        for t in current_tokens:
            doc_freq[t] += 1

    idf_dict = {}
    for t, df in doc_freq.items():
        idf_dict[t] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

    avg_doc_len = total_tokens / total_docs if total_docs > 0 else 0
    return idf_dict, avg_doc_len
