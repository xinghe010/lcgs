TYPE_HIERARCHY_JAVA = {
    'NumericType': {'int', 'long', 'byte', 'short', 'float', 'double', 'char'},
    'Modifier': {'public', 'private', 'protected', 'static', 'final', 'abstract'},
    'LoopStatement': {'ForStatement', 'WhileStatement', 'DoStatement'},
    'ConditionalStatement': {'IfStatement', 'SwitchStatement'},
    'DeclarationType': {'LocalVariableDeclaration', 'FieldDeclaration'},
    'InvocationType': {'MethodInvocation', 'SuperMethodInvocation'},
    'LiteralType': {'Literal'},
}

TYPE_HIERARCHY_C = {
    'NumericType': {'int', 'long', 'short', 'float', 'double', 'char',
                    'unsigned', 'signed'},
    'LoopStatement': {'For', 'While', 'DoWhile'},
    'ConditionalStatement': {'If', 'Switch'},
    'DeclarationType': {'Decl', 'TypeDecl', 'PtrDecl', 'ArrayDecl'},
    'CallType': {'FuncCall'},
    'LiteralType': {'Constant'},
    'OperatorType': {'BinaryOp', 'UnaryOp'},
}

TYPE_HIERARCHY = TYPE_HIERARCHY_JAVA

def get_type_hierarchy(lang='java'):
    if lang == 'c':
        return TYPE_HIERARCHY_C
    return TYPE_HIERARCHY_JAVA

def _build_reverse_map(hierarchy):
    m = {}
    for group_name, members in hierarchy.items():
        for member in members:
            m[member] = group_name
    return m

_TYPE_TO_GROUP_JAVA = _build_reverse_map(TYPE_HIERARCHY_JAVA)
_TYPE_TO_GROUP_C = _build_reverse_map(TYPE_HIERARCHY_C)

def is_compatible(type_a, type_b, lang='java'):
    if type_a == type_b:
        return True
    hierarchy = get_type_hierarchy(lang)
    for group in hierarchy.values():
        if type_a in group and type_b in group:
            return True
    return False

def get_canonical_label(node_type, lang='java'):
    rev_map = _TYPE_TO_GROUP_C if lang == 'c' else _TYPE_TO_GROUP_JAVA
    return rev_map.get(node_type, node_type)
