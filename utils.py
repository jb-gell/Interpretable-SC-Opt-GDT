import numpy as np
from GDTR import GeneticDecisionTreeRegressor
from or_gym.envs.supply_chain.inventory_management import InvManagementMasterEnv
import random
import json
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from graphviz import Digraph, Source


def get_all_split_features(node, feature_list=None):    
    if node is None:
        return
        
    if feature_list is None:
        feature_list = []
    
    if hasattr(node, 'feature'):  # Internal node
        feature_list.append(node.feature)

        if hasattr(node, 'left'):
            get_all_split_features(node.left, feature_list)

        if hasattr(node, 'right'):
            get_all_split_features(node.right, feature_list)
   
    return feature_list


def get_all_split_values(node, value_list=[]):    
    if node is None:
        return
        
    if hasattr(node, 'feature'):  # Internal node
        value_list.append(node.value)

        if hasattr(node, 'left'):
            get_all_split_features(node.left, value_list)

        if hasattr(node, 'right'):
            get_all_split_features(node.right, value_list)
   
    return value_list


def get_max_depth(node):
    if node is None:
        return 0

    if ((not hasattr(node, 'left')) and (not hasattr(node, 'right'))):
        return 1
    
    left_depth = get_max_depth(node.left)    
    right_depth = get_max_depth(node.right)

    return max(left_depth, right_depth) + 1


def visualize_tree_graphviz(node, feature_names=None, dot=None, parent_name=None, graph=None):
    '''
    Example usage:
    dot_tree = visualize_tree_graphviz(vis_tree)
    display(Source(dot_tree.source))
    '''
    if dot is None:
        dot = Digraph(comment='Decision Tree')
        graph = Digraph(comment='Decision Tree')

    if parent_name is None:
        parent_name = "Root"

    current_name = str(hash(node))  # Unique name for the current node

    if hasattr(node, 'feature'):  # Internal node
        dot.node(current_name, label=f"x[{node.feature}] < {round(node.value, 2)} ?", color="bisque3", style="filled", fillcolor="bisque", penwidth="2")
        if parent_name != "Root":
            dot.edge(parent_name, current_name)
        
        if hasattr(node, 'left'):
            visualize_tree_graphviz(node.left, feature_names, dot, current_name, graph)

        if hasattr(node, 'right'):
            visualize_tree_graphviz(node.right, feature_names, dot, current_name, graph)
    else:
        dot.node(current_name, label=f"y: {np.around(node.value)}", color="#ADD8E6", style="filled", fillcolor="#E6F7FF", penwidth="2")
        dot.edge(parent_name, current_name)

    return dot



