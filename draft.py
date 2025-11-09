from multiprocessing import Pool
from utils.data_utils import get_graph_features_from_smi
from collections.abc import Iterable
from typing import List
import numpy as np

src_file="/home/sonh/MAINCE_project/Graph2SMILES/data/BioChem_USPTO_NPL/src-train.txt"

with open(src_file, "r") as f:
    src_lines = f.readlines()[0:1000]

# Seperate the reactions with 2 or more reactants to several lines, each contains one reactant
# Save the mapping e.g. reactants at line 2,3,4 map to reaction at line 2
splitted_src_lines = [i.split(' . ') for i in src_lines]


def flatten_and_map(data: List):
    flattened = []
    mapping = {}
    offset = 0

    for i, item in enumerate(data):

        flattened.extend(item)
        mapping[i] = list(range(offset, offset + len(item)))
        offset += len(item)

    return flattened, mapping

flattened_src_lines, mapping = flatten_and_map(splitted_src_lines)

p = Pool(8)
graph_features_and_lengths = p.imap(
    get_graph_features_from_smi,
    ((i, "".join(line.split()), False) for i, line in enumerate(flattened_src_lines))
)

p.close()
p.join()

graph_features_and_lengths = list(graph_features_and_lengths)

flattened_a_scopes, flattened_a_scopes_lens, flattened_b_scopes, flattened_b_scopes_lens, flattened_a_features, flattened_a_features_lens, \
    flattened_b_features, flattened_b_features_lens, flattened_a_graphs, flattened_b_graphs = zip(*graph_features_and_lengths)


a_scopes = [tuple([flattened_a_scopes[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
a_scopes = tuple(a_scopes)

a_scopes_lens = [tuple([flattened_a_scopes_lens[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
a_scopes_lens = tuple(a_scopes_lens)

b_scopes = [tuple([flattened_b_scopes[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
b_scopes = tuple(b_scopes)

b_scopes_lens = [tuple([flattened_b_scopes_lens[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
b_scopes_lens = tuple(b_scopes_lens)

a_features = [tuple([flattened_a_features[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
a_features = tuple(a_features)

a_features_lens = [tuple([flattened_a_features_lens[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
a_features_lens = tuple(a_features_lens)

b_features = [tuple([flattened_b_features[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
b_features = tuple(b_features)

b_features_lens = [tuple([flattened_b_features_lens[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
b_features_lens = tuple(b_features_lens)

a_graphs = [tuple([flattened_a_graphs[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
a_graphs = tuple(a_graphs)

b_graphs = [tuple([flattened_b_graphs[i2] for i2 in mapping[i1]]) for i1 in range(len(src_lines))]
b_graphs = tuple(b_graphs)

print(len(a_scopes))
print(len(a_scopes_lens))
print(len(b_scopes))
print(len(b_scopes_lens))
print(len(a_features))
print(len(a_features_lens))
print(len(b_features))
print(len(b_features_lens))
print(len(a_graphs))
print(len(b_graphs))

a_scopes = np.concatenate(a_scopes, axis=0)
b_scopes = np.concatenate(b_scopes, axis=0)
a_features = np.concatenate(a_features, axis=0)
b_features = np.concatenate(b_features, axis=0)
a_graphs = np.concatenate(a_graphs, axis=0)
b_graphs = np.concatenate(b_graphs, axis=0)
print(b_scopes.shape)