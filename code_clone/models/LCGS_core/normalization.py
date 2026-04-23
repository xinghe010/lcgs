import re

def normalize_code_semantics(code_str, lang='java'):
    if not isinstance(code_str, str):
        return code_str

    prev = None
    while prev != code_str:
        prev = code_str
        code_str = _apply_common_rules(code_str)
        if lang == 'java':
            code_str = _apply_java_rules(code_str)

    return code_str

def _apply_common_rules(code_str):

    code_str = re.sub(r'\b(\w+)\s*=\s*\1\s*\+\s*1\b', r'\1++', code_str)
    code_str = re.sub(r'\b(\w+)\s*\+=\s*1\b', r'\1++', code_str)

    code_str = re.sub(r'\b(\w+)\s*=\s*\1\s*-\s*1\b', r'\1--', code_str)
    code_str = re.sub(r'\b(\w+)\s*-=\s*1\b', r'\1--', code_str)

    code_str = re.sub(r'(^|[;\s\{])\+\+(\w+)', r'\1\2++', code_str)
    code_str = re.sub(r'(^|[;\s\{])--(\w+)', r'\1\2--', code_str)

    code_str = re.sub(r'\s*==\s*true\b', '', code_str)
    code_str = re.sub(r'\s*!=\s*false\b', '', code_str)

    code_str = re.sub(r'\b(\w+)\s*==\s*false\b', r'!\1', code_str)
    code_str = re.sub(r'\b(\w+)\s*!=\s*true\b', r'!\1', code_str)

    code_str = re.sub(r'\belse\s*\{\s*\}', '', code_str)

    return code_str

def _apply_java_rules(code_str):

    code_str = re.sub(
        r'\bfor\s*\(\s*;\s*([^;]+)\s*;\s*\)',
        r'while (\1)',
        code_str
    )

    code_str = re.sub(
        r'\b(\w+)\s*\?\s*true\s*:\s*false\b',
        r'\1',
        code_str
    )

    code_str = re.sub(
        r'\b(\w+)\s*\?\s*false\s*:\s*true\b',
        r'!\1',
        code_str
    )

    return code_str

def normalize_ast(ast_node):
    return ast_node
