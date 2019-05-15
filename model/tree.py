import torch
import torch.nn as nn
import time
from enum import Enum
import numpy as np


class Node(object):
    """Each node includes a index in {0,...,126}, a node_class in {0,1,2},
    and node data (only similar node and shape node have data)"""
    def __init__(self, index=None, node_class=None, sim=None, shape=None):
        self.index = int(index)
        self.parent_index = self.get_parent(self.index)
        self.children_index = self.get_children(self.index)
        # node_class: 0--split_node; 1--similar_node; 2--shape_node
        self.node_class = node_class
        self.sim = sim      # similar parameters data
        self.shape = shape  # shape data

    @staticmethod
    def index2level(index):
        """Given an index in {0,...,126}, return its level and index in current level, from *ZERO*"""
        level = int(np.floor(np.log2(index+1)))
        index_in_level = index - pow(2, level) + 1
        return level, index_in_level

    @staticmethod
    def level2index(level, index_in_level):
        """Given a level and index in current level, from *ZERO*, return an index in {0,...,126}"""
        index = pow(2, level) - 1 + index_in_level
        return index

    def get_parent(self, index):
        """Given an index in {0,...,126}, return its parent in {0,...,126}"""
        level, index_in_level = self.index2level(index)
        parent_level = level - 1
        parent_index_in_level = int(np.floor(index_in_level / 2))
        parent_index = self.level2index(parent_level, parent_index_in_level)
        return parent_index

    def get_children(self, index):
        """Given an index in {0,...,126}, return its left and right children in {0,...,126}"""
        level, index_in_level = self.index2level(index)
        child_level = level + 1
        child_index_in_level = index_in_level*2
        child_index = self.level2index(child_level, child_index_in_level)
        return child_index, child_index+1


if __name__ == '__main__':
    """test the tree"""
    tree = []
    tree.append(Node(index=0, node_class=0, sim=None, shape=None))
    tree.append(Node(index=1, node_class=0, sim=None, shape=None))
    tree.append(Node(index=2, node_class=1, sim=np.random.randn(8), shape=None))
    tree.append(Node(index=3, node_class=1, sim=np.random.randn(8), shape=None))
    tree.append(Node(index=4, node_class=1, sim=np.random.randn(8), shape=None))
    tree.append(Node(index=5, node_class=2, sim=None, shape=np.random.randn(8)))
    tree.append(Node(index=6, node_class=None, sim=None, shape=None))
    tree.append(Node(index=7, node_class=2, sim=np.random.randn(8), shape=None))
    tree.append(Node(index=8, node_class=None, sim=None, shape=None))
    tree.append(Node(index=9, node_class=2, sim=np.random.randn(8), shape=None))
    tree.append(Node(index=10, node_class=None, sim=None, shape=None))
    tree.append(Node(index=11, node_class=None, sim=None, shape=None))
    tree.append(Node(index=12, node_class=None, sim=None, shape=None))
    tree.append(Node(index=13, node_class=None, sim=None, shape=None))
    tree.append(Node(index=14, node_class=None, sim=None, shape=None))

    tree = np.array(tree)
    print(type(tree[9].sim))

