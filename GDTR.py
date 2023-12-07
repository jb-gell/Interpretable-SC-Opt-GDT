import numpy as np
import copy
import random

class GeneticDecisionTreeRegressor:
    def __init__(self, max_depth=None, parent=None, n_outputs=1, mutation_percentage=30):
        self.max_depth = max_depth
        self.parent = parent
        self.n_outputs = n_outputs
        self.mutation_percentage = mutation_percentage


    def fit(self, X, y, depth=0):
        if depth == self.max_depth or all(len(set(y[:, i])) == 1 for i in range(self.n_outputs)) == 1:
            # If we reach the maximum depth or if all target values are the same, stop splitting (it's a leaf)
            self.value = np.mean(y, axis=0)
            return

        # Find the best split based on a simple mean squared error criterion
        best_split_feature = None
        best_split_value = None
        best_sse = float('inf')
        
        for feature in range(X.shape[1]):
            for value in X[:, feature]:
                
                # select the indices where the feature values are less than or equal to the current 'value'
                left_indices = X[:, feature] <= value
                # select the indices where the feature values are bigger than the current 'value'
                right_indices = X[:, feature] > value
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                left_y = y[left_indices]
                right_y = y[right_indices]
                
                sse = np.sum(np.mean((left_y - np.mean(left_y, axis=0))**2, axis=0)) + np.sum(np.mean((right_y - np.mean(right_y, axis=0))**2, axis=0))

                if sse < best_sse:
                    best_sse = sse
                    best_split_feature = feature
                    best_split_value = value

        if best_split_feature is None:
            print('this is a problem (1)')
            self.value = np.mean(y, axis=0)
            return
        
        self.feature = best_split_feature
        self.value = best_split_value
        
        # create boolean masks again
        left_indices = X[:, best_split_feature] <= best_split_value
        right_indices = X[:, best_split_feature] > best_split_value
        
        # create left and right children of the current node
        self.left = GeneticDecisionTreeRegressor(max_depth=self.max_depth, parent=self, n_outputs=self.n_outputs)
        self.right = GeneticDecisionTreeRegressor(max_depth=self.max_depth, parent=self, n_outputs=self.n_outputs)
        
        self.left.fit(X[left_indices], y[left_indices], depth + 1)
        self.right.fit(X[right_indices], y[right_indices], depth + 1)


    def predict(self, X):
        # check that the current node is not a leaf node and has a feature and value for splitting
        if hasattr(self, 'feature'):
            # create boolean masks
            left_indices = X[:, self.feature] <= self.value
            right_indices = X[:, self.feature] > self.value
            
            y = np.empty((X.shape[0], self.n_outputs))
            
            if hasattr(self.left, 'feature'):
                y[left_indices] = self.left.predict(X[left_indices]) # use recursion for the predictions
                
            else:
                y[left_indices] = self.left.value
            
            if hasattr(self.right, 'feature'):
                y[right_indices] = self.right.predict(X[right_indices])
            else:
                y[right_indices] = self.right.value
                
            return y
        
        else:
            return np.full((X.shape[0], self.n_outputs), self.value)
        
    
    def predict_action(self, X):
        x = np.array(X).reshape(1, -1)
        
        if not hasattr(self, 'feature'):
            y = self.value
        else:
            if x[0, self.feature] <= self.value:
                # Recursive call for left subtree
                y = self.left.predict_action(x)
            else:
                # Recursive call for right subtree
                y = self.right.predict_action(x)

        return y
    
    
    def create_copy(self):
        # Create a deep copy of the instance using copy.deepcopy
        return copy.deepcopy(self)
    
    
    def get_all_child_nodes(self, node):
        child_nodes = []

        # Check if the provided node is not a leaf node and has child nodes
        if hasattr(node, 'feature'):
            # If it's a split node, add it to the list
            child_nodes.append(node)

            # Recursively traverse the left and right child nodes
            child_nodes.extend(node.get_all_child_nodes(self.left))
            child_nodes.extend(node.get_all_child_nodes(self.right))
        else:
            child_nodes.append(node)

        return list(set(child_nodes))


    def get_all_ancestor_nodes(self):
        ancestors = []
        current_node = self
        
        while current_node is not None:
            ancestors.append(current_node)
            current_node = getattr(current_node, 'parent', None)
        
        return list(set(ancestors)) # NB: this includes the first node itself as well
    
    
    def select_node_to_cross(self, rnd_seed=0):
        ''' 
        For now, just randomly choose one child node (where the crossover should happen)
        '''
        child_nodes = self.get_all_child_nodes(self)
        random.seed(rnd_seed)
        node_to_cross = random.choice(child_nodes)
        
        return node_to_cross
    
    
    def generate_crossover(self, other_tree, random_seed=0):
        crossed_tree = self.create_copy()
        other_tree_copy = other_tree.create_copy()

        node_1_to_cross = crossed_tree.select_internal_node()
        node_2_to_cross = other_tree_copy.select_internal_node()
        
        # make *shallow* copies of the deep copy of the other tree (so that we retain references to the deep copy)
        node_1_to_cross.feature = copy.copy(node_2_to_cross.feature)
        node_1_to_cross.value = copy.copy(node_2_to_cross.value)
        node_1_to_cross.left = copy.copy(node_2_to_cross.left)
        node_1_to_cross.right = copy.copy(node_2_to_cross.right)

        return crossed_tree
    
    
    def select_internal_node(self):
        child_nodes = self.get_all_child_nodes(self)
        eligible_child_nodes = [i for i in child_nodes if hasattr(i, 'feature')]
        node_to_mutate = random.choice(eligible_child_nodes)
        
        return node_to_mutate
    
    
    def generate_mutation(self, feature_num):
        mutated_tree = self.create_copy()
        
        # select a non-leaf node
        node_to_mutate = mutated_tree.select_internal_node()
                    
        # access the features to which we can mutate the current feature (and change the feature)
        feature_to_mutate = random.choice([i for i in range(feature_num)])
        node_to_mutate.feature = feature_to_mutate
        
        coeff_list = [round(x, 2) for x in range(-self.mutation_percentage, self.mutation_percentage+1)]
        coeff_list = [x / 100.0 for x in coeff_list]

        value_coeff = random.choice(coeff_list)
        node_to_mutate.value = node_to_mutate.value * value_coeff
        
        return mutated_tree

        