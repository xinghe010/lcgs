import os
import random
import javalang
import javalang.tree
import javalang.ast
import javalang.util

from javalang.ast import Node
import torch
from anytree import AnyNode, RenderTree

from anytree import find
from createclone_java import getedge_nextsib,getedge_flow,getedge_nextstmt,getedge_nexttoken,getedge_nextuse
import pandas as pd

def get_token(node):
    token = ''

    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token
def get_child(root):

    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):

                    yield sub_item
            elif item:

                yield item
    return list(expand(children))
def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)

    for child in children:
        get_sequence(child, sequence)

def getnodes(node,nodelist):
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child,nodelist)

def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)

    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)
def getnodeandedge_astonly(node,nodeindexlist,vocabdict,src,tgt):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child,nodeindexlist,vocabdict,src,tgt)
def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)

def countnodes(node,ifcount,whilecount,forcount,blockcount):
    token=node.token
    if token=='IfStatement':
        ifcount+=1
    if token=='WhileStatement':
        whilecount+=1
    if token=='ForStatement':
        forcount+=1
    if token=='BlockStatement':
        blockcount+=1
    print(ifcount,whilecount,forcount,blockcount)
    for child in node.children:
        countnodes(child,ifcount,whilecount,forcount,blockcount)

def jsonl_to_df(jf):
    import json
    import pandas as pd
    with open(jf, 'r') as json_file:
        json_list = list(json_file)
    x = []
    for json_str in json_list:
        result = json.loads(json_str)
        x.append([int(result['idx']), result['func']])
    return pd.DataFrame(x, columns=['idx','func'])
def unparsable_replace():
    programtext=open('test.java').read()
    programtokens=javalang.tokenizer.tokenize(programtext)
    parser=javalang.parse.Parser(programtokens)
    programast=parser.parse_member_declaration()
    return programast

def createast(args):
    asts=[]
    paths=[]
    alltokens=[]
    import pandas as pd
    p = args.data+args.bench+'/data.jsonl'
    df = jsonl_to_df(p)
    df=df.sample(frac=1)
    for row in range(len(df)): 
        d = df.iloc[row]
        programtext=d['func']        

        try:
            programtokens=javalang.tokenizer.tokenize(programtext)
            parser=javalang.parse.Parser(programtokens)
            programast=parser.parse_member_declaration()
        except:
           programast=unparsable_replace()

        paths.append(d['idx'])
        asts.append(programast)
        get_sequence(programast,alltokens)

    astdict=dict(zip(paths,asts))
    ifcount=0
    whilecount=0
    forcount=0
    blockcount=0
    docount = 0
    switchcount = 0
    for token in alltokens:
        if token=='IfStatement':
            ifcount+=1
        if token=='WhileStatement':
            whilecount+=1
        if token=='ForStatement':
            forcount+=1
        if token=='BlockStatement':
            blockcount+=1
        if token=='DoStatement':
            docount+=1
        if token=='SwitchStatement':
            switchcount+=1
    print(ifcount,whilecount,forcount,blockcount,docount,switchcount)
    print('allnodes ',len(alltokens))
    alltokens=list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    print(vocabsize)
    print(len(asts))
    return astdict,vocabsize,vocabdict

def createseparategraph(args, astdict,vocablen,vocabdict,device,mode='astonly',nextsib=False,ifedge=False,whileedge=False,foredge=False,blockedge=False,nexttoken=False,nextuse=False):
    pathlist=[]
    treelist=[]
    print('nextsib ',nextsib)
    print('ifedge ',ifedge)
    print('whileedge ',whileedge)
    print('foredge ',foredge)
    print('blockedge ',blockedge)
    print('nexttoken', nexttoken)
    print('nextuse ',nextuse)
    print(len(astdict))
    for path,tree in astdict.items():

        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
        createtree(newtree, tree, nodelist)

        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]
        if mode=='astonly':
            getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)
        else:
            getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt,edge_attr)
            if nextsib==True:
                getedge_nextsib(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            getedge_flow(newtree,vocabdict,edgesrc,edgetgt,edge_attr,ifedge,whileedge,foredge)
            if blockedge==True:
                getedge_nextstmt(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            tokenlist=[]
            if nexttoken==True:
                getedge_nexttoken(newtree,vocabdict,edgesrc,edgetgt,edge_attr,tokenlist)
            variabledict={}
            if nextuse==True:
                getedge_nextuse(newtree,vocabdict,edgesrc,edgetgt,edge_attr,variabledict)

        edge_index=[edgesrc, edgetgt]

        astlength=len(x)

        pathlist.append(path)
        treelist.append([[x,edge_index,edge_attr],astlength])
        astdict[path]=[[x,edge_index,edge_attr],astlength]

    return astdict
def creategmndata(args, id,treedict,vocablen,vocabdict,device):

    dataset_path = args.data +'/'+ args.bench +'/'
    k = ['id1', 'id2', 'label']
    trainlist = pd.read_csv(dataset_path+'train.csv', names=k)
    trainlist['label'] = trainlist['label'].apply(lambda x: 1 if x > 0 else -1)
    validlist = pd.read_csv(dataset_path+'valid.csv', names=k)
    validlist['label'] = validlist['label'].apply(lambda x: 1 if x > 0 else -1)
    testlist = pd.read_csv(dataset_path+'test.csv', names=k)
    testlist['label'] = testlist['label'].apply(lambda x: 1 if x > 0 else -1)

    print(len(trainlist), len(validlist), len(testlist))
    print('train data')
    traindata=createpairdata(treedict,trainlist,device=device)

    print('valid data')
    validdata=createpairdata(treedict,validlist,device=device)
    print('test data')
    testdata=createpairdata(treedict,testlist,device=device)
    return traindata, validdata, testdata 
def createpairdata(treedict,pathlist,device):
    f = pathlist
    datalist=[]
    for row in range(len(f)):
        code1path = int(f.iloc[row]['id1'])
        code2path = int(f.iloc[row]['id2'])
        label = int(f.iloc[row]['label'])

        data1 = treedict[code1path]
        data2 = treedict[code2path]
        x1,edge_index1,edge_attr1,ast1length=data1[0][0],data1[0][1],data1[0][2],data1[1]
        x2,edge_index2,edge_attr2,ast2length=data2[0][0],data2[0][1],data2[0][2],data2[1]
        if edge_attr1==[]:
            edge_attr1 = None
            edge_attr2 = None
        data = [[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2], label]
        datalist.append(data)
    return datalist

if __name__ == '__main__':
    astdict, vocabsize, vocabdict=createast()
    treedict=createseparategraph(astdict, vocabsize, vocabdict,device='cpu',mode='else',nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True)
