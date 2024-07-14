import numpy as np
class Node:
    def __init__(self,feature_index,feature_threshold,left_child,right_child,entropy, num_samples,values,is_leaf=False):
        self.feature_index= feature_index
        self.feature_threshold= feature_threshold
        self.left_child= left_child
        self.right_child= right_child
        self.entropy= entropy
        self.num_samples= num_samples
        self.values= values
        self.is_leaf= is_leaf

class DecisionTreeClassifier:
    def __init__(self,min_sample_split = 2,max_depth=2):
        self.min_sample_split= min_sample_split
        self.max_depth= max_depth
        self.root= None
        # min_sample_split is the minimum number of samples a node must have before it can be split
        # max_depth is the maximum depth of the tree
        # root is the root node of the decision tree
    
    def entropy(self,y):
        hist= np.bincount(y)
        ps= hist/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])
    
    
    def information_gain(self,left_y, right_y, current_entropy):
        p= float(len(left_y))/(len(left_y)+len(right_y))
        return current_entropy- p*self.entropy(left_y)-(1-p)*self.entropy(right_y)
    
    def split(self,X, y, feature_index, threshold):
        left_indices= np.where(X[:,feature_index]<=threshold)
        right_indices= np.where(X[:,feature_index]>threshold)
        left_X= X[left_indices]
        right_X= X[right_indices]
        left_y= y[left_indices]
        right_y= y[right_indices]
        return left_X, right_X, left_y, right_y
    
    def split_y(self,X, y, feature_index, threshold):
        left_indices= np.where(X[:,feature_index]<=threshold)
        right_indices= np.where(X[:,feature_index]>threshold)
        left_y= y[left_indices]
        right_y= y[right_indices]
        return left_y, right_y
    def best_split(self,X,y,no_of_features):
        max_information_gain= -float('inf')
        best_feature_index= -1
        best_threshold= -1
        best_entropy= -1
        for m in range(no_of_features):
            unique_classes= np.unique(X[:,m])
            x=len(unique_classes)-1
            averages = [(unique_classes[i] + unique_classes[i + 1]) / 2 for i in range(x)]
            for j in averages:
                y_left, y_right= self.split_y(X, y, m, j)
                entropy_parent= self.entropy(y)
                M=self.information_gain(y_left, y_right, entropy_parent)
                if M>max_information_gain:
                    max_information_gain=M
                    best_feature_index= m
                    best_threshold= j
                    best_entropy= entropy_parent
        x_left, x_right, y_left, y_right= self.split(X, y, best_feature_index, best_threshold)
        return x_left, x_right, y_left, y_right, best_feature_index, best_threshold, best_entropy,max_information_gain
    


    
    def buildTreeRecursive(self,dataset_x,dataset_y,depth=0):
        num_samples= len(dataset_y)
        num_features= dataset_x.shape[1]
        #print(num_samples,num_features)
        if (num_samples>=self.min_sample_split or depth<=self.max_depth):
            #print('here')
            x_left, x_right, y_left, y_right, best_feature_index, best_threshold, best_entropy,information_gain= self.best_split(dataset_x,dataset_y,num_features)
            #print(x_left, x_right, y_left, y_right, best_feature_index, best_threshold, best_entropy)
            #we can get our desired best feature index =2 and best threshold=2.45 for only 1st iteration
            if(information_gain>0):
                left_child= self.buildTreeRecursive(x_left, y_left, depth+1)
                right_child= self.buildTreeRecursive(x_right, y_right, depth+1)
                return Node(feature_index= best_feature_index, feature_threshold= best_threshold, left_child= left_child, right_child= right_child, entropy= best_entropy, num_samples= num_samples, values= np.bincount(dataset_y), is_leaf=False)
        return Node(is_leaf=True,values= np.bincount(dataset_y),num_samples=num_samples,entropy=0.0,feature_index=None,feature_threshold=None,left_child=None,right_child=None)
    
    def build_tree(self,X,y):
        self.root= self.buildTreeRecursive(X,y)

    def print_tree(self):
        if(self.root is None):
            print('Tree is empty')
        else:
            self.print_tree_recursive(self.root)
    
    def print_tree_recursive(self,node):
        if node.is_leaf:
            print('leaf node')
            print('num_samples:',node.num_samples)
            print('values:',node.values)
            print('entropy:',node.entropy)
            print('----------------')
        else:
            print('splitting feature:',node.feature_index)
            print('splitting threshold:',node.feature_threshold)
            print('num_samples:',node.num_samples)
            print('values:',node.values)
            print('entropy:',node.entropy)
            print('-----------------------')
            print('left child')
            self.print_tree_recursive(node.left_child)
            print('------------------------------')
            print('right child')
            self.print_tree_recursive(node.right_child)

   



