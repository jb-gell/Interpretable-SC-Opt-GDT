import numpy as np
import random
import json
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from graphviz import Digraph, Source



# * quadruple barplot: best reward (for each of the four depths) VS generations

def plot_quad_plot(fix_what, what_val, source, var_list, letter):
    means = {}
    std_devs = {}
    
    if fix_what == 'GA':
        for depth in var_list:
            filename = source + f"list{depth}_depth_{what_val}_gen.json"
            
            with open(filename) as json_file:
                reward = json.load(json_file)
                
            second_elements = [[t[1] for t in sublist] for sublist in reward]

            # Convert the nested lists to a NumPy array for efficient mean calculation
            second_elements_array = np.array(second_elements)
                            
            means[depth] = np.mean(second_elements_array, axis=1)   #[np.mean(l[1]) for l in reward]
            std_devs[depth] = np.std(second_elements_array, axis=1) #[np.std(l[1]) for l in reward]

        
        with plt.style.context(['no-latex']):
            plt.style.use(['no-latex'])
            fig, ax1 = plt.subplots()
            
            ax1.plot([2*i for i in range(0, 11)], means[3], color='#ffd700') 
            ax1.plot([2*i for i in range(0, 11)], means[5], color='#f45140') 
            ax1.plot([2*i for i in range(0, 11)], means[10], color='#10d209') 
            ax1.plot([2*i for i in range(0, 11)], means[13], color='#9d02d7') # #1aede9 #10d209
            
            ax1.fill_between([2*i for i in range(0, 11)], np.subtract(means[3], std_devs[3]), np.add(means[3], std_devs[3]), alpha=0.15, color='#ffd700')
            ax1.fill_between([2*i for i in range(0, 11)], np.subtract(means[5], std_devs[5]), np.add(means[5], std_devs[5]), alpha=0.15, color='#f45140')
            ax1.fill_between([2*i for i in range(0, 11)], np.subtract(means[10], std_devs[10]), np.add(means[10], std_devs[10]), alpha=0.15, color='#10d209')
            ax1.fill_between([2*i for i in range(0, 11)], np.subtract(means[13], std_devs[13]), np.add(means[13], std_devs[13]), alpha=0.15, color='#9d02d7')

            ax1.legend(['3', '5', '10', '13'], loc='lower right') #, bbox_to_anchor=(1, 1))
            
            #ticks = [2*i for i in range(1, 11)] + [10]
            ticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            plt.xticks(ticks=ticks, labels=[i for i in range(0, 11)], fontname= 'Times New Roman', fontsize=14)
            plt.yticks(fontsize=14, fontname= 'Times New Roman')
            
            plt.xlabel('Generation Number', fontsize=16, fontname= 'Times New Roman')
            plt.ylabel('Total reward', fontsize=16, fontname= 'Times New Roman')
            #plt.title(f'Best rewards for different max depth values and GA strategy {what_val}')
            plt.ylim(bottom=-2, top=1.5)
            
            fig.suptitle(f'({letter}) {what_val}', y=-0.01, fontname= 'Times New Roman', weight ='bold', fontsize=16)
            
            plt.show()    


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



