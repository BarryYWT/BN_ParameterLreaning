
from __future__ import division
from collections import OrderedDict
import numpy as np
import pandas as pd

###############################################################################
# Bayesian Network core classes + utilities
# This version strips the quotes found in the Alarm .bif file so that internal
# node names match those used in records.dat. It’s been modernised for pandas ≥ 2.
###############################################################################

# ----------------------------  Graph Node  ---------------------------------- #
class Graph_Node:
    def __init__(self, name: str, n: int, vals):
        self.Node_Name  = name            # variable name
        self.nvalues    = n               # number of categorical levels
        self.values     = vals            # list of level labels
        self.Children   = []              # indices (ints) of child nodes
        self.Parents    = []              # names (str) of parent nodes

        # CPT is stored twice:
        #   * raw list in self.CPT (legacy)
        #   * tidy pandas.DataFrame in self.cpt_data (preferred)
        self.CPT        = []
        self.cpt_data   = pd.DataFrame()

    # Convenience:
    def add_child(self, child_idx: int):
        if child_idx not in self.Children:
            self.Children.append(child_idx)

    def set_Parents(self, parents):
        self.Parents = parents

    def set_CPT(self, values):
        self.CPT = list(values)

    def set_cpt_data(self, df: pd.DataFrame):
        self.cpt_data = df.copy()


# ----------------------------  Network  ------------------------------------- #
class network:
    """A Bayesian network made of Graph_Node objects"""

    def __init__(self):
        self.Pres_Graph = OrderedDict()   # name → Graph_Node
        self.MB         = OrderedDict()   # name → list of blanket node names

    # -- Node look‑ups ------------------------------------------------------- #
    def addNode(self, node: Graph_Node):
        self.Pres_Graph[node.Node_Name] = node

    def search_node(self, name: str):
        return self.Pres_Graph.get(name)

    def get_index(self, name: str):
        try:
            return list(self.Pres_Graph.keys()).index(name)
        except ValueError:
            print(f'No node of the name: {name}')
            return None

    # -- Helpers ------------------------------------------------------------- #
    def get_parent_nodes(self, node: Graph_Node):
        return [self.search_node(p) for p in node.Parents]

    def normalise_cpt(self, X: str):
        """Normalise the counts column of node X into probabilities p."""
        node = self.Pres_Graph[X]
        cpt  = node.cpt_data

        if cpt.empty:
            # Build DataFrame from legacy CPT list if necessary
            nvals = node.nvalues
            nums  = np.array(node.CPT, dtype=float)
            chunks = [nums[i:i+nvals] for i in range(0, len(nums), nvals)]
            rows = []
            parents = node.Parents
            grid = np.indices([self.search_node(p).nvalues for p in parents] + [nvals])
            grid = grid.reshape(len(parents)+1, -1).T
            for idx, prob in zip(grid, nums):
                row = {p: idx[i] for i, p in enumerate(parents)}
                row[X] = idx[-1]
                row['counts'] = prob
                rows.append(row)
            cpt = pd.DataFrame(rows)

        # normalise per parent configuration:
        keys = node.Parents + [X]
        def _norm(group):
            counts = group['counts'].to_numpy(dtype=float)
            counts[counts == 0] = 5e-6
            group['p'] = counts / counts.sum()
            return group

        node.cpt_data = cpt.groupby(node.Parents, group_keys=False).apply(_norm)

    # -- Markov blankets ----------------------------------------------------- #
    def set_mb(self):
        for name in self.Pres_Graph:
            self.MB[name] = markov_blanket(self, name)

# -----------------------------  Parsers  ------------------------------------ #
def read_network(bif_filepath: str) -> network:
    """Very simple .bif reader that supports the ALARM network."""
    net = network()

    with open(bif_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            head = parts[0]
            if head == 'variable':
                # e.g. variable  "Hypovolemia" { ...
                name = parts[1].strip('"')
                # read next line to grab value labels
                vals_line = f.readline().strip().split()
                labels = [tok.strip('"') for tok in vals_line[3:-1]]
                net.addNode(Graph_Node(name, len(labels), labels))

            elif head == 'probability':
                # e.g. probability (  "StrokeVolume"  "LVFailure"  "Hypovolemia" ) {
                names = [tok.strip('(")') for tok in parts[1:]]
                names = [n for n in names if n]   # remove empty strings / )
                child = names[0]
                parents = names[1:]

                node = net.search_node(child)
                if node is None:
                    continue
                node.set_Parents(parents)

                # read until ';' to capture CPT numbers
                nums = []
                for segment in iter(lambda: f.readline(), ''):
                    seg = segment.strip()
                    if not seg:
                        continue
                    if seg.startswith('table'):
                        nums.extend([float(x) for x in seg.split()[1:]])
                    elif seg.endswith(';'):
                        nums.extend([float(x) for x in seg[:-1].split()])
                        break
                node.set_CPT(nums)

    # Build DataFrames & normalise
    for var in net.Pres_Graph:
        net.normalise_cpt(var)

    return net

# -----------------------------  Utilities  ---------------------------------- #
def markov_blanket(net: network, var: str):
    node = net.search_node(var)
    mb   = set()

    # parents
    mb.update(node.Parents)
    # children
    for idx in node.Children:
        child_name = list(net.Pres_Graph.keys())[idx]
        mb.add(child_name)
        # spouses
        mb.update(net.search_node(child_name).Parents)

    mb.discard(var)
    return list(mb)

# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('Run main.py instead of bayesnet.py')
