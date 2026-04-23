import os
import re
import json
import random

import pandas as pd
from anytree import AnyNode

from LCGS_core.normalization import normalize_code_semantics

def get_token(node):
    if isinstance(node, str):
        return node
    for attr in ['name', 'value', 'declname', 'op']:
        if hasattr(node, attr):
            val = getattr(node, attr)
            if val is not None:
                return str(val)
    return str(type(node).__name__)

def get_child(root):
    if isinstance(root, str):
        return []
    if not hasattr(root, 'children'):
        return []
    children_attr = root.children
    if callable(children_attr):
        return [c for _, c in children_attr() if c is not None]
    elif isinstance(children_attr, (list, tuple)):
        result = []
        for child in children_attr:
            if isinstance(child, (list, tuple)):
                result.extend([c for c in child if c is not None])
            elif child is not None:
                result.append(child)
        return result
    return []

def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    for child in children:
        get_sequence(child, sequence)

def getnodes(node, nodelist):
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child, nodelist)

def createtree(root, node, nodelist, parent=None):
    id = len(nodelist)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token = token
        root.data = node
    else:
        newnode = AnyNode(id=id, token=token, data=node, parent=parent)
    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root, child, nodelist, parent=root)
        else:
            createtree(root, child, nodelist, parent=newnode)

def getnodeandedge_astonly(node, nodeindexlist, vocabdict, src, tgt):
    token = node.token
    nodeindexlist.append([vocabdict.get(token, 0)])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child, nodeindexlist, vocabdict, src, tgt)

def getnodeandedge(node, nodeindexlist, vocabdict, src, tgt, edgetype):
    token = node.token
    nodeindexlist.append([vocabdict.get(token, 0)])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child, nodeindexlist, vocabdict, src, tgt, edgetype)

_edges = {
    'Nexttoken': 2, 'Prevtoken': 3, 'Nextuse': 4, 'Prevuse': 5,
    'If': 6, 'Ifelse': 7, 'While': 8, 'For': 9,
    'Nextstmt': 10, 'Prevstmt': 11, 'Prevsib': 12,
}

def getedge_nextsib(node, vocabdict, src, tgt, edgetype):
    for i in range(len(node.children) - 1):
        src.append(node.children[i].id)
        tgt.append(node.children[i + 1].id)
        edgetype.append([1])
        src.append(node.children[i + 1].id)
        tgt.append(node.children[i].id)
        edgetype.append([_edges['Prevsib']])
    for child in node.children:
        getedge_nextsib(child, vocabdict, src, tgt, edgetype)

def getedge_flow(node, vocabdict, src, tgt, edgetype,
                 ifedge=False, whileedge=False, foredge=False):
    token = node.token
    if whileedge:
        if token == 'While' and len(node.children) >= 2:
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([_edges['While']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([_edges['While']])
        if token == 'DoWhile' and len(node.children) >= 2:
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([_edges['While']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([_edges['While']])
    if foredge:
        if token == 'For' and len(node.children) >= 2:
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([_edges['For']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([_edges['For']])
    if ifedge:
        if token == 'If' and len(node.children) >= 2:
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([_edges['If']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([_edges['If']])
            if len(node.children) == 3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edgetype.append([_edges['Ifelse']])
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
                edgetype.append([_edges['Ifelse']])
    for child in node.children:
        getedge_flow(child, vocabdict, src, tgt, edgetype, ifedge, whileedge, foredge)

def getedge_nextstmt(node, vocabdict, src, tgt, edgetype):
    token = node.token
    if token == 'Compound':
        for i in range(len(node.children) - 1):
            src.append(node.children[i].id)
            tgt.append(node.children[i + 1].id)
            edgetype.append([_edges['Nextstmt']])
            src.append(node.children[i + 1].id)
            tgt.append(node.children[i].id)
            edgetype.append([_edges['Prevstmt']])
    for child in node.children:
        getedge_nextstmt(child, vocabdict, src, tgt, edgetype)

def getedge_nexttoken(node, vocabdict, src, tgt, edgetype, tokenlist):
    def gettokenlist(node, vocabdict, edgetype, tokenlist):
        if len(node.children) == 0:
            tokenlist.append(node.id)
        for child in node.children:
            gettokenlist(child, vocabdict, edgetype, tokenlist)

    gettokenlist(node, vocabdict, edgetype, tokenlist)
    for i in range(len(tokenlist) - 1):
        src.append(tokenlist[i])
        tgt.append(tokenlist[i + 1])
        edgetype.append([_edges['Nexttoken']])
        src.append(tokenlist[i + 1])
        tgt.append(tokenlist[i])
        edgetype.append([_edges['Prevtoken']])

def getedge_nextuse(node, vocabdict, src, tgt, edgetype, variabledict):
    def getvariables(node, vocabdict, edgetype, variabledict):
        token = node.token

        if hasattr(node, 'data') and type(node.data).__name__ == 'ID':
            variable = token
            if variable not in variabledict:
                variabledict[variable] = [node.id]
            else:
                variabledict[variable].append(node.id)
        for child in node.children:
            getvariables(child, vocabdict, edgetype, variabledict)

    getvariables(node, vocabdict, edgetype, variabledict)
    for v in variabledict.keys():
        for i in range(len(variabledict[v]) - 1):
            src.append(variabledict[v][i])
            tgt.append(variabledict[v][i + 1])
            edgetype.append([_edges['Nextuse']])
            src.append(variabledict[v][i + 1])
            tgt.append(variabledict[v][i])
            edgetype.append([_edges['Prevuse']])

def clean_c_code(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.S)
    return code.strip()

def _make_fallback_ast():
    from pycparser import c_parser
    parser = c_parser.CParser()
    return parser.parse("void func() {}")

def jsonl_to_df(jf):
    with open(jf, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)
    x = []
    for json_str in json_list:
        result = json.loads(json_str)
        x.append([int(result['idx']), result['func']])
    return pd.DataFrame(x, columns=['idx', 'func'])

def createast(args):
    from pycparser import c_parser

    asts = []
    paths = []
    alltokens = []

    p = args.data + args.bench + '/data.jsonl'
    df = jsonl_to_df(p)
    df = df.sample(frac=1)

    fallback_ast = _make_fallback_ast()
    parser = c_parser.CParser()

    for row in range(len(df)):
        d = df.iloc[row]
        programtext = d['func']

        try:
            programtext = normalize_code_semantics(programtext, lang='c')
            programtext = clean_c_code(programtext)
            programast = parser.parse(programtext)
        except:
            try:
                wrapped = "void func() { " + programtext + " }"
                programast = parser.parse(wrapped)
            except:
                programast = fallback_ast

        if hasattr(programast, 'ext') and len(programast.ext) > 0:
            programast_flat = programast.ext[0]
        else:
            programast_flat = programast

        paths.append(d['idx'])
        asts.append(programast_flat)
        get_sequence(programast_flat, alltokens)

    astdict = dict(zip(paths, asts))

    ifcount = sum(1 for t in alltokens if t == 'If')
    whilecount = sum(1 for t in alltokens if t == 'While')
    forcount = sum(1 for t in alltokens if t == 'For')
    blockcount = sum(1 for t in alltokens if t == 'Compound')
    docount = sum(1 for t in alltokens if t == 'DoWhile')
    switchcount = sum(1 for t in alltokens if t == 'Switch')
    print(ifcount, whilecount, forcount, blockcount, docount, switchcount)
    print('allnodes ', len(alltokens))

    alltokens = list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    print(vocabsize)
    print(len(asts))
    return astdict, vocabsize, vocabdict

def createseparategraph(args, astdict, vocablen, vocabdict, device,
                        mode='astonly', nextsib=False, ifedge=False,
                        whileedge=False, foredge=False, blockedge=False,
                        nexttoken=False, nextuse=False):
    pathlist = []
    treelist = []
    print('nextsib ', nextsib)
    print('ifedge ', ifedge)
    print('whileedge ', whileedge)
    print('foredge ', foredge)
    print('blockedge ', blockedge)
    print('nexttoken', nexttoken)
    print('nextuse ', nextuse)
    print(len(astdict))

    for path, tree in astdict.items():
        nodelist = []
        newtree = AnyNode(id=0, token=None, data=None)
        createtree(newtree, tree, nodelist)

        x = []
        edgesrc = []
        edgetgt = []
        edge_attr = []

        if mode == 'astonly':
            getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)
        else:
            getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
            if nextsib:
                getedge_nextsib(newtree, vocabdict, edgesrc, edgetgt, edge_attr)
            getedge_flow(newtree, vocabdict, edgesrc, edgetgt, edge_attr,
                         ifedge, whileedge, foredge)
            if blockedge:
                getedge_nextstmt(newtree, vocabdict, edgesrc, edgetgt, edge_attr)
            tokenlist = []
            if nexttoken:
                getedge_nexttoken(newtree, vocabdict, edgesrc, edgetgt, edge_attr, tokenlist)
            variabledict = {}
            if nextuse:
                getedge_nextuse(newtree, vocabdict, edgesrc, edgetgt, edge_attr, variabledict)

        edge_index = [edgesrc, edgetgt]
        astlength = len(x)

        pathlist.append(path)
        treelist.append([[x, edge_index, edge_attr], astlength])
        astdict[path] = [[x, edge_index, edge_attr], astlength]

    return astdict

def creategmndata(args, id, treedict, vocablen, vocabdict, device):
    dataset_path = args.data + '/' + args.bench + '/'
    k = ['id1', 'id2', 'label']
    trainlist = pd.read_csv(dataset_path + 'train.csv', names=k)
    trainlist['label'] = trainlist['label'].replace(0, -1)
    validlist = pd.read_csv(dataset_path + 'valid.csv', names=k)
    validlist['label'] = validlist['label'].replace(0, -1)
    testlist = pd.read_csv(dataset_path + 'test.csv', names=k)
    testlist['label'] = testlist['label'].replace(0, -1)

    print(len(trainlist), len(validlist), len(testlist))
    print('train data')
    traindata = createpairdata(treedict, trainlist, device=device)
    print('valid data')
    validdata = createpairdata(treedict, validlist, device=device)
    print('test data')
    testdata = createpairdata(treedict, testlist, device=device)
    return traindata, validdata, testdata

def createpairdata(treedict, pathlist, device):
    f = pathlist
    datalist = []
    for row in range(len(f)):
        code1path = int(f.iloc[row]['id1'])
        code2path = int(f.iloc[row]['id2'])
        label = int(f.iloc[row]['label'])
        if code1path not in treedict or code2path not in treedict:
            continue
        data1 = treedict[code1path]
        data2 = treedict[code2path]
        x1, edge_index1, edge_attr1, ast1length = data1[0][0], data1[0][1], data1[0][2], data1[1]
        x2, edge_index2, edge_attr2, ast2length = data2[0][0], data2[0][1], data2[0][2], data2[1]
        if edge_attr1 == []:
            edge_attr1 = None
            edge_attr2 = None
        data = [[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2], label]
        datalist.append(data)
    return datalist
