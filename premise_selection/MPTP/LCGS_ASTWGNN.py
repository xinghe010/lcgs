import networkx as nx
import torch
import numpy as np
from networkx.algorithms import isomorphism
from torch_geometric.data import Data, InMemoryDataset
from torch.nn.functional import one_hot
import json
from torch_geometric.data import Batch, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import global_mean_pool
import pickle
from networkx.algorithms import isomorphism
import os
import torch
import argparse
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
import json
import pickle
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import networkx as nx
from lark import Lark, Transformer
import copy
from sklearn.metrics import recall_score, precision_score, f1_score
from torch_geometric.nn import GCNConv, SAGPooling, global_max_pool
from torch_scatter import scatter_mean  
import math 
from torch_geometric.utils import softmax as geo_softmax

fof_parser = Lark(r"""
    annotated_formula: "fof(" name "," formula_role ","  fof_formula ")."

    name: NAME
    NAME: UPPER_LOW_ALPHA_NUMERIC+

    formula_role: FORMULA_ROLE
    FORMULA_ROLE: "axiom"

    ?fof_formula: unitary_formula | binary_formula

    ?unitary_formula: literal | type_bool | quantified_formula | negative "(" fof_formula ")" | constant

    ?binary_formula: assoc_formula | non_assoc_formula

    non_assoc_formula: "(" fof_formula binary_connective fof_formula ")"
    binary_connective: BINARY_CONNECTIVE

    assoc_formula: "(" fof_formula assoc_connective fof_formula ")"
    assoc_connective: ASSOC_CONNECTIVE

    quantified_formula: "(" quantifier variable_list fof_formula ")"
    quantifier: QUANTIFIER
    variable_list: "[" variable ( "," variable )* "]" ":"

    ?literal: atom | negative "(" atom ")"
    negative: NEGATIVE

    atom: predicate "(" term_argument ")" | term equal term
    predicate: PREDICATE
    equal: EQUAL

    ?term: functional_term | variable | constant
    functional_term: functor "(" term_argument ")"
    term_argument: term ("," term)*
    functor: FUNCTOR
    variable: VARIABLE
    constant: CONSTANT

    PREDICATE: LOWER_ALPHA LOW_ALPHA_NUMERIC*
    FUNCTOR: LOWER_ALPHA LOW_ALPHA_NUMERIC*
    VARIABLE: UPPER_ALPHA UPPER_ALPHA_NUMERIC*
    CONSTANT: NUMERIC+ | LOWER_ALPHA LOW_ALPHA_NUMERIC*

    ?type_bool: type_true | type_false
    type_true: TYPE_TRUE
    type_false: TYPE_FALSE
    TYPE_TRUE: "$true"
    TYPE_FALSE: "$false"

    EQUAL: "="
    QUANTIFIER: "!" | "?"
    NEGATIVE: "~"
    BINARY_CONNECTIVE: "<=>" | "=>"
    ASSOC_CONNECTIVE : "&" | "|"

    LOW_ALPHA_NUMERIC : LOWER_ALPHA | NUMERIC | "_"
    UPPER_ALPHA_NUMERIC: UPPER_ALPHA | NUMERIC | "_"
    UPPER_LOW_ALPHA_NUMERIC : UPPER_ALPHA | LOWER_ALPHA | NUMERIC | "_"
    LOWER_ALPHA : "a" .. "z"
    UPPER_ALPHA : "A" .. "Z"
    NUMERIC : "0" .. "9" 
    %ignore " "
    """, start='annotated_formula')

class Transform(Transformer):
    annotated_formula = lambda self, a: a[2]
    name = lambda self, a: a[0][:]
    formula_role = lambda self, a: a[0][:]
    fof_formula = lambda self, a: a
    unitary_formula = lambda self, a: a
    binary_formula = lambda self, a: a
    assoc_formula = lambda self, a: a
    non_assoc_formula = lambda self, a: self._transform_non_assoc(a)
    quantified_formula = lambda self, a: a
    literal = lambda self, a: a
    atom = lambda self, a: a

    term = lambda self, a: a
    term_argument = lambda self, a: a
    functional_term = lambda self, a: a
    variable_list = lambda self, a: a
    type_bool = lambda self, a: a
    constant = lambda self, a: a[0][:]
    variable = lambda self, a: a[0][:]
    predicate = lambda self, a: a[0][:]
    functor = lambda self, a: a[0][:]
    quantifier = lambda self, a: a[0][:]
    negative = lambda self, a: a[0][:]
    binary_connective = lambda self, a: a[0][:]
    assoc_connective = lambda self, a: a[0][:]
    equal = lambda self, a: a[0][:]
    type_true = lambda self, a: a[0][:]
    type_false = lambda self, a: a[0][:]

    def _transform_non_assoc(self, args):
            left, connective, right = args[0], args[1], args[2]
            if connective == "=>":
                return [["~", left], "|", right]
            else:
                return [left, connective, right]

    def quantified_formula(self, args):
        quant, var_list, sub = args[0], args[1], args[2]
        var_set = set(var_list)

        def replace_vars(term):
            if isinstance(term, str):
                if term in var_set:
                    return quant + term
                return term
            elif isinstance(term, list):
                return [replace_vars(t) for t in term]
            return term

        replaced_sub = replace_vars(sub)
        return replaced_sub

def fof_formula_transformer(formula):
    fomula_pharse = fof_parser.parse(formula)
    formula_tree = Transform().transform(fomula_pharse)
    return formula_tree

NEGATIVE_CONNECTIVE = {"~"}
BINARY_CONNECTIVE = {"=>", "<=>"}
ASSOC_CONNECTIVE = {"|", "&"}
QUANTIFIER = {"!", "?"}
EQUAL = {"=", "!="}
BOOL = {"$true"}
VARIABLE_PATTERN = re.compile(r'[!?]?[A-Z][A-Z0-9_]*')
FUNCTOR_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")

class Node:

    def __init__(self, name):
        self.id = self.__class__.id
        self.name = name
        self.children = []
        self.parents = []
        self.scoped = []
        self.__class__.id += 1

    @classmethod
    def rest_id(cls):
        cls.id = 0

    def __str__(self):
        parents_info = ' '.join([str(x.id) for x in self.parents if x is not None])
        children_info = ' '.join([str(x.id) for x in self.children if x is not None])
        scoped_info = ' '.join([str(x.id) for x in self.scoped if x is not None])
        return '<{}>: token {} | parents {} | children {} | scoped {}'.format(
            self.id, self.name, parents_info, children_info, scoped_info)

    def __repr__(self):
        return self.__str__()

class Graph:

    def __init__(self, formula, rename):
        self.graph = []
        self.id2subterm = dict()
        self.convert(formula, rename)

    def __iter__(self):
        return self.graph.__iter__()

    def __getitem__(self, index):
        return self.graph[index]

    def __len__(self):
        return len(self.graph)

    def __str__(self):
        return '\n'.join([str(node) for node in self.graph if node is not None])

    def __repr__(self):
        return self.__str__()

    def create_functor_node(self, name, parent):
        functor_node = Node(name)
        self.graph.append(functor_node)
        if parent:
            functor_node.parents.append(parent)
            for node in parent.scoped:
                if node not in functor_node.scoped and node is not None:
                    functor_node.scoped.append(node)
        return functor_node

    def create_variable_node(self, name, parent):
        assert parent is not None, "a variable should have at least one parent"
        variable_node = Node(name)
        variable_node.parents.append(parent)
        if parent:
            variable_node.scoped.extend([n for n in parent.scoped if n is not None])
        self.graph.append(variable_node)
        return variable_node

    def create_constant_node(self, name, parent):
        exsiting_same_constant_nodes = [node for node in self.graph if node.name == name]
        if len(exsiting_same_constant_nodes) == 1 and exsiting_same_constant_nodes[0].children == []:
            constant_node = exsiting_same_constant_nodes[0]
        else:
            constant_node = Node(name)
            self.graph.append(constant_node)
        if parent:
            constant_node.parents.append(parent)
            for node in parent.scoped:
                if node not in constant_node.scoped and node is not None:
                    constant_node.scoped.append(node)
        return constant_node

    def create_connective_node(self, name, parent):
        connective_node = Node(name)
        if parent:
            connective_node.parents.append(parent)
            connective_node.scoped.extend([n for n in parent.scoped if n is not None])
        self.graph.append(connective_node)
        return connective_node

    def create_negative_node(self, name, parent):
        negative_node = Node(name)
        if parent:
            negative_node.parents.append(parent)
            negative_node.scoped.extend([n for n in parent.scoped if n is not None])
        self.graph.append(negative_node)
        return negative_node

    def merge_sub(self, formula, parent):
        for id in self.id2subterm:
            if self.id2subterm[id][0] == formula:
                pre_node = self.id2subterm[id][1]
                pre_node.parents.append(parent)
                return pre_node

    def check_merge(self, formula, parent):
        if formula in [self.id2subterm[id][0] for id in self.id2subterm]:
            for id in self.id2subterm:
                if self.id2subterm[id][0] == formula:
                    pre_node = self.id2subterm[id][1]
                    var_flag = self.check_variable(pre_node)
                    if not var_flag:
                        return True
                    else:
                        if set(pre_node.scoped).issubset(set(parent.scoped)):
                            return True
                        else:
                            return False

    def check_variable(self, node):
        if re.match(VARIABLE_PATTERN, node.name):
            return True
        else:
            if not node.children:
                return False
            else:
                for child in node.children:
                    if child is None:
                        continue
                    flag = self.check_variable(child)
                    if flag:
                        return True
                return False

    def formula_to_dense_graph(self, formula, parent=None):
        if isinstance(formula, str):
            if re.match(VARIABLE_PATTERN, formula):
                return self.create_variable_node(formula, parent)
            if re.match(FUNCTOR_PATTERN, formula) or (formula in BOOL):
                return self.create_constant_node(formula, parent)

        if isinstance(formula, list) and len(formula) == 1:
            return self.formula_to_dense_graph(formula[0], parent)

        if isinstance(formula[0], str) and \
                formula[0] in NEGATIVE_CONNECTIVE and len(formula) == 2:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                negative_node = self.create_negative_node(formula[0], parent)
                self.id2subterm[negative_node.id] = (formula, negative_node)
                node = self.formula_to_dense_graph(formula[1], negative_node)
                if node is not None:
                    negative_node.children.append(node)
                return negative_node

        if isinstance(formula[1], str) and \
                formula[1] in (BINARY_CONNECTIVE | ASSOC_CONNECTIVE | EQUAL) \
                and len(formula) == 3:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                connective_node = self.create_connective_node(
                    formula[1], parent)
                self.id2subterm[connective_node.id] = (
                    formula, connective_node)
                left_node = self.formula_to_dense_graph(
                    formula[0], connective_node)
                if left_node is not None:
                    connective_node.children.append(left_node)
                right_node = self.formula_to_dense_graph(
                    formula[2], connective_node)
                if right_node is not None:
                    connective_node.children.append(right_node)
                return connective_node

        if isinstance(formula[0], str) and \
                re.match(FUNCTOR_PATTERN, formula[0]) and len(formula) == 2:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                functor_node = self.create_functor_node(formula[0], parent)
                self.id2subterm[functor_node.id] = (formula, functor_node)
                args = formula[1]
                if not isinstance(args, list):
                    args = [args]
                for argument in args:
                    argument_node = self.formula_to_dense_graph(
                        argument, functor_node)
                    if argument_node is not None:
                        functor_node.children.append(argument_node)
                return functor_node

    def convert(self, formula, rename):
        Node.rest_id()
        self.formula_to_dense_graph(formula)
        if rename:
            variable_nodes = [node for node in self.graph if re.match(
                VARIABLE_PATTERN, node.name)]
            for node in variable_nodes:
                if node.name.startswith('!'):
                    node.name = '!VAR'
                elif node.name.startswith('?'):
                    node.name = '?VAR'
                else:
                    node.name = 'VAR'

            var_nodes = [node for node in self.graph if node and node.name == '!VAR']
            if len(var_nodes) > 1:
                main = var_nodes[0]
                for other in var_nodes[1:]:
                    for parent in other.parents:
                        for i in range(len(parent.children)):
                            if parent.children[i] == other:
                                parent.children[i] = main
                    for p in other.parents:
                        if p not in main.parents:
                            main.parents.append(p)
                    main.scoped = list(set(main.scoped + other.scoped))
                    self.graph[other.id] = None
        self.graph = [n for n in self.graph if n is not None]

        id_map = {node.id: new_id for new_id, node in enumerate(self.graph)}
        for node in self.graph:
            node.id = id_map[node.id]

            node.parents = [self.graph[id_map[p.id]] if p is not None else None for p in node.parents if p.id in id_map]
            node.children = [self.graph[id_map[c.id]] if c is not None else None for c in node.children if c.id in id_map]
            node.scoped = [self.graph[id_map[s.id]] if s is not None else None for s in node.scoped if s.id in id_map]
        return self.graph

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines

def dumps_list_to_json(obj, file_path):
    with open(file_path, "w+") as f:
        f.write("\n".join([json.dumps(element) for element in obj]))

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def dump_pickle_file(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def set_recorder(name, logfile):
    recorder = logging.getLogger(name)
    recorder.setLevel(logging.INFO)
    rf_handler = logging.StreamHandler()
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

    f_handler = logging.FileHandler(logfile)
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    recorder.addHandler(rf_handler)
    recorder.addHandler(f_handler)
    return recorder

class Statements:
    def __init__(self, statements_file):
        self.statements = self.build_statements(statements_file)

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, name):
        return self.statements[name]

    def __iter__(self):
        return self.statements.__iter__()

    def build_statements(self, statements_file):
        statements = dict()
        lines = read_file(statements_file)
        for line in lines:
            name = line.split(',')[0].replace("fof(", "")
            statements[name] = line.replace(" ", "")
        return statements

def py_plot(title, train_loss, valid_loss, train_acc, valid_acc, save_file):
    assert len(train_loss) == len(valid_loss)
    assert len(train_acc) == len(valid_acc)
    epochs = np.arange(1, len(train_loss)+1)
    plt.figure()
    plt.subplot(121)
    plt.plot(epochs, train_loss, "-", color="salmon", label="train loss")
    plt.plot(epochs, valid_loss, "-", color="saddlebrown", label="valid loss")
    plt.title("loss")
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(epochs, train_acc, "-", color="salmon", label="train acc")
    plt.plot(epochs, valid_acc, "-", color="saddlebrown", label="valid acc")
    plt.title("accuracy")
    plt.legend()
    plt.grid()
    plt.suptitle(title)
    plt.savefig(save_file, dpi=2000)
    plt.close()

class MLPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=F.relu,
                 bias=True,
                 batch=False,
                 drop=False):
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.drop = drop
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = nn.BatchNorm1d(out_channels)
        if self.drop:
            self.drop = nn.Dropout(0.8)
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.relu:
            nn.init.kaiming_normal_(self.lin.weight, nonlinearity="relu")
        elif self.activation == F.leaky_relu:
            nn.init.kaiming_normal_(self.lin.weight)
        else:
            nn.init.xavier_normal_(self.lin.weight)
        if self.bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        x = self.lin(x)
        if self.batch and x.size()[0] > 1:
            x = self.BN(x)
        if self.drop:
            x = self.drop(x)
        x = self.activation(x)
        return x

class Initialization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Initialization, self).__init__()
        self.embedding = nn.Embedding(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        indices = torch.argmax(x, dim=1)
        output = self.embedding(indices)
        return output

class DAGEmbedding(nn.Module):
    def __init__(self, node_out_channels, layers):
        super(DAGEmbedding, self).__init__()
        self.K = layers

        self.F_T = nn.ModuleList(
            [MLPBlock(3 * node_out_channels, node_out_channels, batch=True) for _ in range(layers)])
        self.F_M = nn.ModuleList(
            [MLPBlock(3 * node_out_channels, node_out_channels, batch=True) for _ in range(layers)])
        self.F_B = nn.ModuleList(
            [MLPBlock(3 * node_out_channels, node_out_channels, batch=True) for _ in range(layers)])
        self.F_1 = nn.ModuleList([MLPBlock(3 * node_out_channels, 1, batch=True) for _ in range(layers)])
        self.F_trans_TW = nn.ModuleList(
            [MLPBlock(node_out_channels, node_out_channels, batch=True) for _ in range(layers)])

    def forward(self, x, term_walk_index, node_weight=None):
        N = x.size()[0]

        if node_weight is not None:

            x = x * node_weight.unsqueeze(-1)

        for i in range(self.K):
            term_walk_feat = torch.cat([x[term_walk_index[0]],
                                        x[term_walk_index[1]],
                                        x[term_walk_index[2]]], dim=1)

            score = self.F_1[i](term_walk_feat)

            att = geo_softmax(score, term_walk_index[1], num_nodes=N)

            trans_T = self.F_T[i](term_walk_feat)

            m_T = scatter_add(att * trans_T,
                               index=term_walk_index[0],
                               dim=0, dim_size=N)

            trans_M = self.F_M[i](term_walk_feat)
            m_M = scatter_add(att * trans_M,
                               index=term_walk_index[1],
                               dim=0, dim_size=N)

            trans_B = self.F_B[i](term_walk_feat)
            m_B = scatter_add(att * trans_B,
                               index=term_walk_index[2],
                               dim=0, dim_size=N)

            m_TW = m_T + m_M + m_B
            m_TW = self.F_trans_TW[i](m_TW)
            x = x + m_TW
        return x

class DAGPooling(nn.Module):
    def __init__(self, node_out_channels):
        super(DAGPooling, self).__init__()
        self.lin1 = MLPBlock(node_out_channels, 1, bias=False)

        self.SAGPool = SAGPooling(node_out_channels, ratio=0.9, GNN=GCNConv)

    def forward(self, x, batch_index, child_index, parent_index):

        x1, edge_index1, _, batch1, perm, score = self.SAGPool(x, child_index, None, batch_index)
        x2, edge_index2, _, batch2, perm, score = self.SAGPool(x, parent_index, None, batch_index)

        output1 = torch.cat([global_mean_pool(x1, batch1), global_max_pool(x1, batch1)], dim=1)
        output2 = torch.cat([global_mean_pool(x2, batch2), global_max_pool(x2, batch2)], dim=1)

        output = output1 + output2
        return output

class Classifier(nn.Module):
    def __init__(self, node_out_channels):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            MLPBlock(4 * node_out_channels, node_out_channels,
                     batch=True),
            MLPBlock(node_out_channels, 2, activation=lambda x: x,
                     batch=True))

    def forward(self, conj_batch, prem_batch):
        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.classifier(x_concat)
        return pred_y

class MinNormSolver:
    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):

        if v1v2 >= v1v1:
            return 1.0, 0.0
        if v1v2 >= v2v2:
            return 0.0, 1.0
        d2 = v1v1 + v2v2 - 2.0 * v1v2
        if d2 < 1e-12:
            return 0.5, 0.5
        gamma = (v2v2 - v1v2) / d2
        return gamma, 1.0 - gamma

    @staticmethod
    def _min_norm_2d(vecs, dps):

        gamma, _ = MinNormSolver._min_norm_element_from2(dps[0, 0], dps[0, 1], dps[1, 1])
        return torch.tensor([gamma, 1 - gamma], device=vecs.device)

    @staticmethod
    def find_min_norm_element(vecs):

        n = vecs.shape[0]

        if n == 1:
            return torch.tensor([1.0], device=vecs.device)

        dps = torch.matmul(vecs, vecs.t())

        if n == 2:
            return MinNormSolver._min_norm_2d(vecs, dps)

        return torch.tensor([1.0/n]*n, device=vecs.device)

class MGDA(nn.Module):
    def __init__(self):
        super(MGDA, self).__init__()

    def forward(self, losses, shared_params):
        active_losses = [l for l in losses if l.requires_grad]

        if not active_losses:
            return sum(losses) 

        if len(active_losses) == 1:
            return active_losses[0]

        grads = []

        for loss in active_losses:

            g = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)

            g_flat = []
            for gi in g:
                if gi is not None:
                    g_flat.append(gi.view(-1))

            if g_flat:
                grads.append(torch.cat(g_flat))
            else:
                grads.append(None) 

        valid_grads = []
        valid_indices = [] 
        for i, g in enumerate(grads):
            if g is not None:
                valid_grads.append(g)
                valid_indices.append(i)

        if len(valid_grads) <= 1:

            return sum([active_losses[i] for i in valid_indices])

        grad_mat = torch.stack(valid_grads)

        with torch.no_grad():
            weights = MinNormSolver.find_min_norm_element(grad_mat)

        weighted_loss = 0.0
        for i, w in zip(valid_indices, weights):
            weighted_loss += w * active_losses[i]

        return weighted_loss

class PremiseSelectionModel(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, layers, lambda_param=1.0, mcs_margin=0.8):
        super(PremiseSelectionModel, self).__init__()
        self.initial = Initialization(node_in_channels, node_out_channels)

        self.dag_emb = DAGEmbedding(node_out_channels, layers)
        self.pooling = DAGPooling(node_out_channels) 

        self.classifier = Classifier(node_out_channels) 
        self.criterion = nn.CrossEntropyLoss()
        self.corrects = None

        self.mgda = MGDA() 
        self.lambda_param = lambda_param  
        self.mcs_margin = mcs_margin

    def forward(self, batch):
        device = next(self.parameters()).device 

        h_s = self.initial(batch.x_s)
        h_t = self.initial(batch.x_t)

        w_s = getattr(batch, 'node_w_s', None)
        w_t = getattr(batch, 'node_w_t', None)
        if w_s is not None: w_s = w_s.to(device)
        if w_t is not None: w_t = w_t.to(device)

        h_s = self.dag_emb(h_s, batch.term_walk_index_s, w_s)
        h_t = self.dag_emb(h_t, batch.term_walk_index_t, w_t)

        h_g_s = self.pooling(h_s, batch.x_s_batch, batch.child_index_s.to(device), batch.parent_index_s.to(device))
        h_g_t = self.pooling(h_t, batch.x_t_batch, batch.child_index_t.to(device), batch.parent_index_t.to(device))

        pred_y = self.classifier(h_g_s, h_g_t)

        pred_prob = F.softmax(pred_y, dim=1)[:, 1]
        y_float = batch.y.float().view(-1)
        w = getattr(batch, 'sample_weight', None)
        if w is None:
            w = torch.ones_like(pred_prob)
        else:
            w = w.view(-1)

        ce_vec = F.cross_entropy(pred_y, batch.y, reduction='none')
        ce_loss = (ce_vec * w).sum() / (w.sum() + 1e-8)

        jaccard_tensor = batch.mcs_jaccard.view(-1)
        epsilon = 0.1
        weights = jaccard_tensor + epsilon
        l1_vec = weights * torch.abs(pred_prob - (y_float * jaccard_tensor))
        l1_loss = (l1_vec * w).sum() / (w.sum() + 1e-8)

        kappa = 5.0
        mask = (batch.mcs_equal.view(-1) == 1)
        if mask.any():
            soft_penalty = torch.sigmoid(kappa * (self.mcs_margin - pred_prob[mask]))
            w_mask = w[mask]
            l2_loss = (soft_penalty * w_mask).sum() / (w_mask.sum() + 1e-8)
        else:
            l2_loss = pred_prob.new_tensor(0.)

        logic_losses = [l1_loss, l2_loss]
        weighted_logic_loss = self.mgda(logic_losses, list(self.parameters()))

        total_loss = ce_loss + self.lambda_param * weighted_logic_loss

        pred_label = torch.argmax(pred_y, dim=1)
        self.corrects = (pred_label == batch.y).sum().cpu().item()
        return total_loss, pred_label

_NODE_DICT_CACHE = None
def _load_node_dict():
    global _NODE_DICT_CACHE
    if _NODE_DICT_CACHE is not None:
        return _NODE_DICT_CACHE
    candidates = []
    env_path = os.environ.get("HP_NODE_DICT_PATH")
    if env_path:
        candidates.append(env_path)

    candidates += [
        os.path.join(os.getcwd(), "re_node_dict.pkl"),
        os.path.join(os.getcwd(), "node_dict.pkl"),
    ]

    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_ASTWGNN", "re_node_dict.pkl"))
    for p in candidates:
        try:
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    _NODE_DICT_CACHE = pickle.load(f)
                    return _NODE_DICT_CACHE
        except Exception:
            continue
    raise FileNotFoundError("Cannot locate re_node_dict.pkl. Set env HP_NODE_DICT_PATH to its location.")

def to_networkx(graph):
    node_dict = _load_node_dict()
    G = nx.DiGraph()
    for node in graph.graph:
        label = node.name
        if label in node_dict:
            type_id = node_dict[label]
        else:
            raise ValueError(f"Label '{label}' not found in re_node_dict.pkl")
        G.add_node(node.id, label=label, type_id=type_id)

    for node in graph.graph:
        for child in node.children:
            G.add_edge(node.id, child.id)

    for n in list(G.nodes()):
        preds = list(G.predecessors(n))
        parent_syms = tuple(sorted(G.nodes[p]['label'] for p in preds)) if preds else tuple()
        G.nodes[n]['parent_syms'] = parent_syms
    for n in list(G.nodes()):
        G.nodes[n]['is_leaf'] = (G.out_degree(n) == 0)

    type_id_edges = [(G.nodes[src]['type_id'], G.nodes[tgt]['type_id']) for src, tgt in G.edges()]
    return G, type_id_edges

def compute_mcs_and_jaccard(graph_s, graph_t, *, count_limit=20000, time_limit_sec=0.5,
                            enable_relax_forall_exists_const=True):

    import time as _time
    from networkx.algorithms import isomorphism as _iso

    G_s, _ = to_networkx(graph_s)
    G_t, _ = to_networkx(graph_t)

    def _make_node_match(left_is_premise: bool):
        def _match(a, b):

            if a.get('type_id') == b.get('type_id'):
                return True
            if not enable_relax_forall_exists_const:
                return False

            if left_is_premise and a.get('is_leaf') and b.get('is_leaf'):
                pa = set(a.get('parent_syms', ()))
                pb = set(b.get('parent_syms', ()))
                if pa and (pa & pb):
                    return True
            return False
        return _match

    def _max_submono(G1, G2, node_match, count_limit, time_limit_sec):
        matcher = _iso.DiGraphMatcher(G1, G2, node_match=node_match)
        best, t0 = 0, _time.time()
        for k, iso in enumerate(matcher.subgraph_monomorphisms_iter()):
            if len(iso) > best:
                best = len(iso)
            if (k + 1) >= count_limit or (_time.time() - t0) >= time_limit_sec:
                break
        return best

    m1 = _max_submono(G_t, G_s, _make_node_match(True), count_limit, time_limit_sec)

    m2 = _max_submono(G_s, G_t, _make_node_match(False), count_limit, time_limit_sec)

    max_mcs_size = max(m1, m2)
    v1_size = G_s.number_of_nodes()
    v2_size = G_t.number_of_nodes()
    denom = v1_size + v2_size - max_mcs_size
    jaccard = (max_mcs_size / (denom + 1e-6)) if denom != 0 else 0.0
    return torch.tensor([jaccard], dtype=torch.float32)

class PairData(Data):
    def __init__(self, x_s=None, term_walk_index_s=None,
                 x_t=None, term_walk_index_t=None, y=None,
                 graph_s=None, graph_t=None, mcs_jaccard=None,
                 mcs_equal=None, prem_key=None, conj_key=None,
                 sample_weight=None, is_augmented=None,
                 parent_index_s=None, child_index_s=None,
                 parent_index_t=None, child_index_t=None):
        super().__init__()
        self.x_s = x_s
        self.x_t = x_t
        self.term_walk_index_s = term_walk_index_s
        self.term_walk_index_t = term_walk_index_t
        self.y = y
        self.graph_s = graph_s
        self.graph_t = graph_t

        if mcs_jaccard is None:
            self.mcs_jaccard = torch.tensor([0.0], dtype=torch.float32)
        else:
            self.mcs_jaccard = mcs_jaccard.view(-1)
        if mcs_equal is None:
            self.mcs_equal = torch.tensor([0], dtype=torch.uint8)
        else:
            self.mcs_equal = mcs_equal.view(-1)

        self.prem_key = prem_key
        self.conj_key = conj_key
        self.sample_weight = torch.tensor([1.0], dtype=torch.float32) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32).view(-1)
        self.is_augmented = torch.tensor([0], dtype=torch.uint8) if is_augmented is None else torch.as_tensor(is_augmented, dtype=torch.uint8).view(-1)

        self.parent_index_s = parent_index_s
        self.child_index_s = child_index_s
        self.parent_index_t = parent_index_t
        self.child_index_t = child_index_t

    def __inc__(self, key, value, *args, **kwargs):

        if key == "term_walk_index_s":
            return self.x_s.size(0)
        if key == "term_walk_index_t":
            return self.x_t.size(0)

        if key in ["parent_index_s", "child_index_s"]:
            return self.x_s.size(0)
        if key in ["parent_index_t", "child_index_t"]:
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class FormulaGraphDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_class,
                 statements_file,
                 node_dict_file,
                 rename=True):
        self.root = root
        self.data_class = data_class
        self.statements = Statements(statements_file)
        self.rename = rename
        self.node_dict = load_pickle_file(node_dict_file)

        self.symbol_types = self._load_symbol_types(statements_file)
        self.type_bounds = {'predicate': (0.5, 6.0), 'function': (0.0, 3.0), 'other': (0.0, 6.0)}
        self.k1_tau     = {'predicate': 1.2, 'function': 1.2, 'other': 1.2}
        self.b_tau      = {'predicate': 0.75, 'function': 0.60, 'other': 0.75}
        self.tp_alpha   = 1.0

        self.cwj_alpha  = 0.5  
        self.cwj_beta   = 0.5  

        self._load_or_build_idf_meta(statements_file)  
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["{}.json".format(self.data_class)]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.data_class)]

    def graph_process(self, G):
        nodes = []
        term_walk_indices = []
        parent_indices = []
        child_indices = []

        for node in G:
            nodes.append(node.name)
            for parent in node.parents:
                parent_indices.append([node.id, parent.id])
            for child in node.children:
                child_indices.append([node.id, child.id])

            if node.parents and node.children:
                for parent in node.parents:
                    for child in node.children:
                        term_walk_indices.append([parent.id,
                                                  node.id,
                                                  child.id])

        term_walk_indices = np.array(
            term_walk_indices, dtype=np.int64).reshape(-1, 3).T
        parent_indices = np.array(parent_indices, dtype=np.int64).reshape(-1, 2).T
        child_indices = np.array(child_indices, dtype=np.int64).reshape(-1, 2).T

        return nodes, term_walk_indices, parent_indices, child_indices

    def vectorization(self, objects, object_dict):
        indices = [object_dict[obj] for obj in objects]
        onehot = one_hot(torch.LongTensor(indices), len(object_dict)).float()
        return onehot

    def _load_symbol_types(self, statements_file):

        import os, json
        _locals = locals()
        base_dir = None
        for _cand in ['statements_file', 'data_file', 'dataset_path', 'statements_path']:
            if _cand in _locals and isinstance(_locals[_cand], str):
                base_dir = os.path.dirname(_locals[_cand])
                break
        if base_dir is None:
            base_dir = os.getcwd()
        pred_path = os.path.join(base_dir, "predicates_scope_rule_formula_only.json")
        func_path = os.path.join(base_dir, "functions_scope_rule_formula_only.json")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing predicates file: {pred_path}")
        if not os.path.exists(func_path):
            raise FileNotFoundError(f"Missing functions file: {func_path}")
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        with open(func_path, "r", encoding="utf-8") as f:
            funcs = json.load(f)
        if isinstance(preds, dict):
            preds = list(preds.keys())
        if isinstance(funcs, dict):
            funcs = list(funcs.keys())
        predicates = set(preds)
        functions = set(funcs)
        symbol_types = {}
        for t in predicates:
            symbol_types[t] = "predicate"
        for t in functions:
            symbol_types[t] = "function"
        try:
            print(f"[symbol types] predicates={len(predicates)} functions={len(functions)} (dir={base_dir})")
        except Exception:
            pass
        return symbol_types
    def _compute_depths(self, graph):
        from collections import deque
        n = len(graph.graph)
        parents = [[] for _ in range(n)]
        children = [[] for _ in range(n)]
        for node in graph.graph:
            for p in node.parents:
                parents[node.id].append(p.id)
            for c in node.children:
                children[node.id].append(c.id)
        roots = [i for i in range(n) if len(parents[i]) == 0]
        depth = [None] * n
        dq = deque()
        for r in roots:
            depth[r] = 0
            dq.append(r)
        while dq:
            u = dq.popleft()
            for v in children[u]:
                if depth[v] is None or depth[v] > depth[u] + 1:
                    depth[v] = (depth[u] + 1)
                    dq.append(v)
        return [d if d is not None else 0 for d in depth]

    def _is_content_symbol(self, sym: str) -> bool:
        EXCLUDE_TOKENS = {"!VAR", "?VAR", "$true", "|", "&", "=>", "<=>", "~", "=", "0", "1", "2"}
        if sym in EXCLUDE_TOKENS:
            return False
        if sym.isdigit():
            return False
        return True

    def _idf_meta_path(self):
        return os.path.join(self.root, "processed", "idf_meta.pt")

    def _load_or_build_idf_meta(self, statements_file):
        meta_path = self._idf_meta_path()
        try:
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location="cpu")

                if "idf_vec" in meta and "avg_doc_len" in meta:
                    self.idf_vec = meta["idf_vec"].float()
                    self.avg_doc_len = float(meta["avg_doc_len"])
                    return
        except Exception:
            pass 

        self._build_idf_from_statements(statements_file)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        torch.save({
            "idf_vec": self.idf_vec.cpu(),
            "avg_doc_len": float(self.avg_doc_len),

            "node_dict_size": len(self.node_dict),
            "statements_mtime": os.path.getmtime(statements_file),
            "params": {
                "type_bounds": self.type_bounds,
                "k1_tau": self.k1_tau,
                "b_tau": self.b_tau,
                "tp_alpha": self.tp_alpha,
            },
        }, meta_path)

    def _build_idf_from_statements(self, statements_file: str):
        import math
        node_dict = self.node_dict
        V = len(node_dict)
        df = [0] * V
        doc_lens = []
        N = 0

        for name, line in self.statements.statements.items():
            try:
                g = Graph(fof_formula_transformer(line), rename=self.rename)
                names = [n.name for n in g.graph]
                content = [t for t in names if (t in node_dict) and self._is_content_symbol(t)]
                if not content:
                    continue
                N += 1
                doc_lens.append(len(content))
                for t in set(content):
                    df[node_dict[t]] += 1
            except Exception:
                continue

        if N == 0:
            self.idf_vec = torch.ones(V, dtype=torch.float32)
            self.avg_doc_len = 1.0
            return
        avgL = float(sum(doc_lens)) / max(1, len(doc_lens))
        self.avg_doc_len = avgL
        idf = [1.0] * V
        for tok, tid in node_dict.items():
            dfi = df[tid]
            base = math.log(max(((N - dfi + 0.5) / (dfi + 0.5)), 1e-9) + 1.0)
            tau = self.symbol_types.get(tok, 'other')
            low, high = self.type_bounds.get(tau, self.type_bounds['other'])
            idf[tid] = float(min(max(base, low), high))

        self.idf_vec = torch.tensor(idf, dtype=torch.float32)

    def _bm25_node_weights(self, node_names, node_depths):
        from collections import Counter
        node_dict = self.node_dict
        content_tokens = [t for t in node_names if (t in node_dict) and self._is_content_symbol(t)]
        L = len(content_tokens)
        if L == 0:
            return torch.ones(len(node_names), dtype=torch.float32)

        tf = Counter(content_tokens)
        avgL = float(getattr(self, "avg_doc_len", 1.0))
        idf_vec = self.idf_vec
        alpha = float(getattr(self, "tp_alpha", 1.0))

        denom_cache = {}
        for tau in ('predicate', 'function', 'other'):
            k1 = self.k1_tau.get(tau, self.k1_tau['other'])
            b_ = self.b_tau.get(tau, self.b_tau['other'])
            denom_cache[tau] = k1 * (1.0 - b_ + b_ * (L / max(1.0, avgL)))

        weights = []
        for i, t in enumerate(node_names):
            if (t in node_dict) and self._is_content_symbol(t):
                tau = self.symbol_types.get(t, 'other')
                k1 = self.k1_tau.get(tau, self.k1_tau['other'])
                denom_norm = denom_cache[tau]
                base_tf = float(tf.get(t, 0))
                if base_tf <= 0:
                    w = 1.0
                else:
                    d_i = int(node_depths[i]) if i < len(node_depths) else 0
                    tf_pos = base_tf * (1.0 + alpha / (d_i + 1.0))
                    num = tf_pos * (k1 + 1.0)
                    den = tf_pos + denom_norm + 1e-9
                    w = float(idf_vec[node_dict[t]].item()) * (num / den)
                weights.append(w)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)
    def graph_to_nx(self, graph):
        G = nx.DiGraph()
        for node in graph.graph:
            G.add_node(node.id, name=node.name)
            for child in node.children:
                G.add_edge(node.id, child.id)
        return G

    def process(self):
        raw_examples = [json.loads(line) for line in read_file(self.raw_paths[0])]
        dataList = []

        max_new_per_prem = 1000
        aug_weight = 0.8
        include_conflict = False  

        from collections import defaultdict
        pos_succ = defaultdict(set)
        all_pairs = set()
        pos_pairs = set()
        neg_pairs = set()
        for example in raw_examples:
            conj, prem, label = example
            all_pairs.add((conj, prem))
            if int(label) == 1:
                pos_succ[prem].add(conj)
                pos_pairs.add((conj, prem))
            else:
                neg_pairs.add((conj, prem))

        aug_pairs = []
        seen_aug = set()  

        for A, Bs in pos_succ.items():
            added = 0
            for B in Bs:
                for C in pos_succ.get(B, ()):
                    if A == C:
                        continue
                    pair2 = (C, A) 
                    if pair2 in all_pairs:
                        continue
                    if pair2 in seen_aug:
                        continue
                    aug_pairs.append((C, A, 1, aug_weight, True))
                    seen_aug.add(pair2)
                    added += 1
                    if added >= max_new_per_prem:
                        break
                if added >= max_new_per_prem:
                    break

        graph_cache = {}
        def get_graph_by_key(key):
            if key in graph_cache:
                return graph_cache[key]
            g = Graph(fof_formula_transformer(self.statements[key]), rename=self.rename)
            graph_cache[key] = g
            return g

        def _best_mcs_mapping(G_s, G_t, *, count_limit=20000, time_limit_sec=0.5, enable_relax=True):

            import time as _time
            from networkx.algorithms import isomorphism as _iso

            def _make_node_match(left_is_premise: bool):
                def _match(a, b):
                    if a.get('type_id') == b.get('type_id'):
                        return True
                    if not enable_relax or not left_is_premise:
                        return False
                    if a.get('is_leaf') and b.get('is_leaf'):
                        pa = set(a.get('parent_syms', ()))
                        pb = set(b.get('parent_syms', ()))
                        if pa and (pa & pb):
                            return True
                    return False
                return _match

            matcher1 = _iso.DiGraphMatcher(G_t, G_s, node_match=_make_node_match(True))
            best1, best_map1, t0 = 0, None, _time.time()
            for k, iso in enumerate(matcher1.subgraph_monomorphisms_iter()):
                if len(iso) > best1:
                    best1 = len(iso); best_map1 = iso.copy()
                if (k + 1) >= count_limit or (_time.time() - t0) >= time_limit_sec:
                    break

            matcher2 = _iso.DiGraphMatcher(G_s, G_t, node_match=_make_node_match(False))
            best2, best_map2, t0 = 0, None, _time.time()
            for k, iso in enumerate(matcher2.subgraph_monomorphisms_iter()):
                if len(iso) > best2:
                    best2 = len(iso); best_map2 = iso.copy()
                if (k + 1) >= count_limit or (_time.time() - t0) >= time_limit_sec:
                    break

            if best1 >= best2 and best_map1 is not None:

                pairs = [(sid, best_map1[tid]) for tid, sid in best_map1.items()]
                return pairs, 1
            elif best_map2 is not None:

                pairs = [(sid, tid) for sid, tid in best_map2.items()]
                return pairs, 2
            else:
                return [], 0

        def _compute_wj_and_cwj(conj_graph, prem_graph, w_c, w_p, j_scalar, alpha, beta):

            G_s, _ = to_networkx(conj_graph)
            G_t, _ = to_networkx(prem_graph)

            for G in (G_s, G_t):
                for n in list(G.nodes()):
                    if 'is_leaf' not in G.nodes[n]:
                        G.nodes[n]['is_leaf'] = (G.out_degree(n) == 0)
                    if 'parent_syms' not in G.nodes[n]:
                        preds = list(G.predecessors(n))
                        parent_syms = tuple(sorted(G.nodes[p].get('label', '') for p in preds)) if preds else tuple()
                        G.nodes[n]['parent_syms'] = parent_syms

            pairs, direction = _best_mcs_mapping(G_s, G_t)

            W1 = float(torch.as_tensor(w_c).sum().item())
            W2 = float(torch.as_tensor(w_p).sum().item())

            W_MCS = 0.0
            if direction in (1, 2):
                for sid, tid in pairs:

                    if 0 <= sid < len(w_c) and 0 <= tid < len(w_p):
                        W_MCS += 0.5 * (float(w_c[sid]) + float(w_p[tid]))

            denom = W1 + W2 - W_MCS
            WJ = (W_MCS / (denom + 1e-9)) if denom > 0 else 0.0

            CWJ = alpha * float(j_scalar) + beta * float(WJ)
            return torch.tensor([CWJ], dtype=torch.float32)

        def mcs_subiso_equal(conj_graph , prem_graph ):

            Gp, _ = to_networkx(prem_graph)
            Gc, _ = to_networkx(conj_graph)

            for G in (Gp, Gc):
                for n in list(G.nodes()):
                    if 'is_leaf' not in G.nodes[n]:
                        G.nodes[n]['is_leaf'] = (G.out_degree(n) == 0)
                    if 'parent_syms' not in G.nodes[n]:
                        preds = list(G.predecessors(n))
                        parent_syms = tuple(sorted(G.nodes[p].get('label', '') for p in preds)) if preds else tuple()
                        G.nodes[n]['parent_syms'] = parent_syms

            def _relaxed_node_match(a, b):
                if a.get('type_id') == b.get('type_id'):
                    return True
                if a.get('is_leaf') and b.get('is_leaf'):
                    pa = set(a.get('parent_syms', ()))
                    pb = set(b.get('parent_syms', ()))
                    if pa and (pa & pb):
                        return True
                return False

            matcher = isomorphism.DiGraphMatcher(Gp, Gc, node_match=_relaxed_node_match)
            ok = matcher.subgraph_is_isomorphic()
            return torch.tensor([1.0 if ok else 0.0], dtype=torch.float32)

        def make_pairdata(conj, prem, label, weight=1.0, is_aug=False):
            conj_graph = get_graph_by_key(conj)
            prem_graph = get_graph_by_key(prem)

            mcs_equal = mcs_subiso_equal(conj_graph , prem_graph ) 

            c_nodes, c_term_walk_indices, c_parent, c_child = self.graph_process(conj_graph)
            p_nodes, p_term_walk_indices, p_parent, p_child = self.graph_process(prem_graph)

            c_depths = self._compute_depths(conj_graph)
            p_depths = self._compute_depths(prem_graph)
            bm25_w_c = self._bm25_node_weights(c_nodes, c_depths)
            bm25_w_p = self._bm25_node_weights(p_nodes, p_depths)

            J = compute_mcs_and_jaccard(conj_graph, prem_graph) 
            mcs_jaccard = _compute_wj_and_cwj(conj_graph, prem_graph, bm25_w_c, bm25_w_p,
                                             J.item(), self.cwj_alpha, self.cwj_beta)

            x_c = self.vectorization(c_nodes, self.node_dict)
            x_p = self.vectorization(p_nodes, self.node_dict)

            c_term_walk_indices = torch.from_numpy(c_term_walk_indices).long()
            p_term_walk_indices = torch.from_numpy(p_term_walk_indices).long()

            c_parent = torch.from_numpy(c_parent).long()
            c_child = torch.from_numpy(c_child).long()
            p_parent = torch.from_numpy(p_parent).long()
            p_child = torch.from_numpy(p_child).long()

            dataList.append(PairData(
                x_s=x_c, term_walk_index_s=c_term_walk_indices,
                x_t=x_p, term_walk_index_t=p_term_walk_indices,

                parent_index_s=c_parent, child_index_s=c_child,
                parent_index_t=p_parent, child_index_t=p_child,

                y=torch.LongTensor([int(label)]),
                graph_s=conj_graph, graph_t=prem_graph,
                mcs_jaccard=mcs_jaccard,
                mcs_equal=mcs_equal,
                prem_key=prem, conj_key=conj,
                sample_weight=torch.tensor([float(weight)], dtype=torch.float32),
                is_augmented=torch.tensor([1 if is_aug else 0], dtype=torch.uint8)
            ))
            dataList[-1].node_w_s = bm25_w_c
            dataList[-1].node_w_t = bm25_w_p

        for (conj, prem, label) in raw_examples:
            make_pairdata(conj, prem, label, weight=1.0, is_aug=False)

        for (conj, prem, label, w, is_aug) in aug_pairs:
            try:
                make_pairdata(conj, prem, label, weight=w, is_aug=is_aug)
            except Exception:

                continue

        data, slices = self.collate(data_list=dataList)
        torch.save((data, slices), self.processed_paths[0])

def train(epoch, data_loader, model, optimizer, device, recorder):
    recorder.info('------starting {} epoch training------'.format(epoch))
    model.train()
    total = 0
    corrects = 0
    train_loss = 0.0

    for i, batch in enumerate(data_loader, 1):
        optimizer.zero_grad()
        batch.to(device=device)
        loss, pred_label = model(batch) 
        corrects += model.corrects
        loss.backward()
        optimizer.step()
        total += batch.y.size()[0]
        train_loss += loss.cpu().item()
    log = "train epoch[{}] end! train loss: {:.4f} train accuarcy: {:.2f}%".format(
        epoch, train_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return train_loss / i, corrects / total

def valid(epoch, data_loader, model, device, recorder):
    recorder.info('------starting {} epoch valid------'.format(epoch))
    model.eval()
    total = 0
    corrects = 0
    valid_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch.to(device=device)
            loss, pred_label = model(batch)  
            corrects += model.corrects
            total += batch.y.size()[0]
            valid_loss += loss.cpu().item()

    log = "valid epoch[{}] end! valid loss: {:.4f} valid accuarcy: {:.2f}%".format(
        epoch, valid_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return valid_loss / i, corrects / total

def test(data_loader, model, device, recorder):
    recorder.info('------starting test------')
    model.eval()
    total = 0
    corrects = 0
    test_loss = 0.0
    all_pred_labels = []  
    all_true_labels = []  

    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch.to(device=device)
            loss, pred_label = model(batch)
            corrects += model.corrects
            total += batch.y.size()[0]
            test_loss += loss.cpu().item()
            all_pred_labels.append(pred_label.cpu())  
            all_true_labels.append(batch.y.cpu()) 

    all_pred_labels = torch.cat(all_pred_labels)
    all_true_labels = torch.cat(all_true_labels)

    with open('test_predictions.txt', 'w') as f:
        for pred, true in zip(all_pred_labels, all_true_labels):
            f.write(f'{pred.item()} {true.item()}\n')

    recall = recall_score(all_true_labels, all_pred_labels, pos_label=1)
    precision = precision_score(all_true_labels, all_pred_labels, pos_label=1)
    f1 = f1_score(all_true_labels, all_pred_labels, pos_label=1)

    log = "test end! test loss: {:.4f} test accuracy: {:.2f}% recall: {:.4f} precision: {:.4f} f1: {:.4f}".format(
        test_loss / i, (corrects / total) * 100, recall, precision, f1)
    recorder.info(log)
    return test_loss / i, corrects / total

def hyper_parameters():
    params = argparse.ArgumentParser()
    params.add_argument(
        "--model_save",
        type=str,
        default="./model_save/model_save_128_1_32",
        help="the directory to save models")
    params.add_argument(
        "--root_dir",
        type=str,
        default="./dataset_ASTWGNN/",
        help="the directory to save data")
    params.add_argument("--node_out_channels",
                        type=int,
                        default=128,
                        help="the dimension of node")
    params.add_argument("--layers",
                        type=int,
                        default=1,
                        help="the number of message passing steps")
    params.add_argument("--device",
                        type=str,
                        default="cuda:1",
                        help="device name {cpu, cuda:0, cuda:1}")
    params.add_argument("--epochs",
                        type=int,
                        default=60,
                        help='Number of training episodes')
    params.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Initial learning rate for Adam")
    params.add_argument("--weight_decay",
                        type=float,
                        default=1e-4,
                        help="L2 normalization penality")
    params.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch Size")
    params.add_argument("--train_fraction", 
                        type=float, 
                        default=1, 
                        help="Fraction of training data to use (e.g., 0.1 for 10%)")
    args = params.parse_args()
    return args

def main():
    args = hyper_parameters()
    if torch.cuda.is_available() and "cuda" in args.device:
        torch.cuda.set_device(args.device)
    os.environ["HP_NODE_DICT_PATH"] = os.path.join(args.root_dir, "re_node_dict.pkl")

    if args.train_fraction < 1.0:
        base_name = os.path.basename(args.model_save.rstrip('/'))
        dir_name = os.path.dirname(args.model_save.rstrip('/'))
        new_folder_name = f"{base_name}_frac_{int(args.train_fraction * 100)}"
        args.model_save = os.path.join(dir_name, new_folder_name)
        print(f"--- Experiment: Using {args.train_fraction*100}% Training Data ---")
        print(f"--- New Model Save Path: {args.model_save} ---")
    if not os.path.exists(args.model_save):
        os.makedirs(args.model_save)
    recorder = set_recorder("TWNN_MCS_parser_graph",
                            os.path.join(args.model_save, "record.log"))

    if args.device == "cpu":
        torch.manual_seed(24)
    else:
        torch.cuda.manual_seed_all(24)

    params_info = ''
    for key, value in vars(args).items():
        params_info += '\n{}: {}'.format(key, value)
    recorder.info(params_info)

    model = PremiseSelectionModel(792,
                                  args.node_out_channels,
                                  args.layers,
                                  lambda_param=1.0).to(device=args.device)

    optimizer = Adam(params=[{'params': model.parameters()}], lr=args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer)

    recorder.info('------DATA LOADING------')

    train_dataset = FormulaGraphDataset(os.path.join(args.root_dir, "train"),
                                        "train",
                                        os.path.join(args.root_dir,
                                                     "statements"),
                                        os.path.join(args.root_dir,
                                                     "re_node_dict.pkl"),
                                        rename=True)
    if args.train_fraction < 1.0:

        total_len = len(train_dataset)
        subset_len = int(total_len * args.train_fraction)

        indices = torch.randperm(total_len)[:subset_len]
        train_dataset = train_dataset[indices]

        recorder.info(f"--- Training Set Reduced: {total_len} -> {len(train_dataset)} samples ({args.train_fraction*100}%) ---")

    valid_dataset = FormulaGraphDataset(os.path.join(args.root_dir, "valid"),
                                        "valid",
                                        os.path.join(args.root_dir,
                                                     "statements"),
                                        os.path.join(args.root_dir,
                                                     "re_node_dict.pkl"),
                                        rename=True)
    test_dataset = FormulaGraphDataset(os.path.join(args.root_dir, "test"),
                                       "test",
                                       os.path.join(args.root_dir,
                                                    "statements"),
                                       os.path.join(args.root_dir,
                                                    "re_node_dict.pkl"),
                                       rename=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              follow_batch=["x_s", "x_t"])
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              follow_batch=["x_s", "x_t"])
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             follow_batch=["x_s", "x_t"])

    recorder.info('------DATA LOADED------')

    recorder.info('------PROCESS START------')
    history = {"train_loss": [], "valid_loss": [], "train_acc": [],
               "valid_acc": [], "test_loss": None, "test acc": None}

    top_models = [] 

    for epoch in range(1, args.epochs + 1):
        recorder.info('------learning rate is {}------'.format(
            optimizer.param_groups[0]["lr"]))
        train_loss, train_acc = train(epoch, train_loader, model,
                                      optimizer, args.device, recorder)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        valid_loss, valid_acc = valid(epoch, valid_loader, model,
                                      args.device, recorder)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        current_state_dict = copy.deepcopy(model.state_dict())
        top_models.append((valid_loss, current_state_dict))

        top_models = sorted(top_models, key=lambda x: x[0])[:5]

        lr_scheduler.step(valid_loss)

    recorder.info('------ Saving and Testing Individual Top-5 Models ------')
    if top_models:
        for rank, (v_loss, state_dict) in enumerate(top_models):

            filename = f"{rank + 1}.pt"
            save_path = os.path.join(args.model_save, filename)
            torch.save({"model": state_dict}, save_path)
            recorder.info(f'Saved Rank {rank+1} model (Valid Loss: {v_loss:.4f}) to {filename}')

            model.load_state_dict(state_dict) 
            recorder.info(f'--- Testing Rank {rank+1} model ---')
            t_loss, t_acc = test(test_loader, model, args.device, recorder)
            recorder.info(f'Rank {rank+1} Result -> Test Loss: {t_loss:.4f}, Test Acc: {t_acc:.4f}')

    if top_models:
        def average_state_dicts(state_dicts):
            avg_state_dict = {}
            num_models = len(state_dicts)
            for key in state_dicts[0].keys():
                avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / num_models
            return avg_state_dict

        top_state_dicts = [sd for _, sd in top_models]
        avg_state_dict = average_state_dicts(top_state_dicts)
        model.load_state_dict(avg_state_dict)
        recorder.info('------Loaded averaged parameters from top-5 models------')
    else:
        recorder.info('------No models to average; using last model------')

    torch.save({"model": model.state_dict()}, os.path.join(args.model_save, "averaged_top5.pt"))

    recorder.info('--- Testing Averaged Model ---') 
    test_loss, test_acc = test(test_loader, model, args.device, recorder)
    test_loss, test_acc = test(test_loader, model, args.device, recorder)
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    dump_pickle_file(history, os.path.join(args.model_save, "history.pkl"))
    py_plot("evaluation", history["train_loss"], history["valid_loss"],
            history["train_acc"], history["valid_acc"],
            os.path.join(args.model_save, "figure"))

    recorder.info('------PROCESS FINISH------')

if __name__ == "__main__":
    main()
