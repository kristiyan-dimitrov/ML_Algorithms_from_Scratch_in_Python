import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.node with a built decision tree.
        """
        # Need to handle this special case where I have just one feature left
        # And it's getting cast as a 1-dim vector
        # I need it to stay as a 2-dim vector with 1 column
        if features.ndim == 1:
            features = features.reshape(-1,1)
        
        inf_gains = [information_gain(features, index, targets) for index in range(features.shape[1])]

        max_gain = max(inf_gains) if len(inf_gains) != 0 else 0
        
        # Check that the number of columns in features and number of attribute names matches
        self._check_input(features)
        
        if (len(self.attribute_names) == 0) or (len(list(set(targets))) == 1): 
            
            zeros = sum(targets == 0)
            ones = sum(targets == 1)
            pred_value = 1
            if zeros >= ones:
                pred_value = 0
            self.tree = Node(value = pred_value, attribute_name='leaf')
        
        elif max_gain != 0:
            # Finding the first feature with max gain 
            # There might be more, but we just take the first one since it doesn't really matter
            for index in range(len(inf_gains)):
                if inf_gains[index] == max_gain:
                # the value of index is now the index where the max_gain is
                    break
            self.tree = Node(value=0, attribute_name=self.attribute_names[index], attribute_index=index)
            
            remaining_attribute_names = [self.attribute_names[i] for i in range(len(self.attribute_names)) if i != index ]
            remaining_attribute_indices = [ i for i in range(len(self.attribute_names)) if i != index]
        
            left_features = features[features[:,index] <= self.tree.value, :][:,remaining_attribute_indices]
            left_targets = targets[features[:,index] <= self.tree.value]
        
            right_features = features[features[:,index] > self.tree.value, :][:,remaining_attribute_indices]
            right_targets = targets[features[:,index] > self.tree.value]
            
            left_tree = DecisionTree(remaining_attribute_names)
            left_tree.fit(left_features, left_targets)
            self.tree.branches.append(left_tree)
        
            right_tree = DecisionTree(remaining_attribute_names)
            right_tree.fit(right_features, right_targets)
            self.tree.branches.append(right_tree)
        
        else:
            zeros = sum(targets == 0)
            ones = sum(targets == 1)
            pred_value = 1
            if zeros >= ones:
                pred_value = 0
            self.tree = Node(value = pred_value, attribute_name='leaf')
            
            
    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        if features.ndim == 1:
            features = features.reshape(1,-1)
        
        if features.shape[0] != 1:
            self._check_input(features)
        
        predictions = np.full(features.shape[0], np.nan, dtype=int)
        
        for i in range(features.shape[0]):
            if self.tree.attribute_index is None:
                if features.shape[0] != 1: # Special case if the tree is just a stump i.e. single leaf node
                    xor_values = [sum(features[i]) if sum(features[i]) < 2 else 0 for i in range(features.shape[0])]
                    return np.array(xor_values, dtype=int)
                return self.tree.value
            else:
                if features[i][self.tree.attribute_index] == 0:
                    # Need to modify the row I'm passing on so the index is properly interpreted
                    mask = [k for k in range(len(features[i])) if k != self.tree.attribute_index]
                    predictions[i] = self.tree.branches[0].predict(features[i][mask])
                else:
                    mask = [k for k in range(len(features[i])) if k != self.tree.attribute_index]
                    predictions[i] = self.tree.branches[1].predict(features[i][mask])
                    
        return predictions


    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        # val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name)) #, val

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)


def prior_proba(targets:np.array, class_:float):

    return sum(targets==class_)/len(targets)

def entropy(targets):

    # entropy = 0 for entirely homogeneous nodes i.e. nodes with just 1 class
    if len(list(set(targets))) == 1:
        return 0

    return sum([ -prior_proba(targets, class_) * np.log2(prior_proba(targets, class_))  for class_ in list(set(targets))])

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    parent_entropy = entropy(targets)

    feature_classes = list(set(features[:,attribute_index]))

    # if the selected feature has only 1 class in it, then it will not separate anything
    # i.e. it will not give any information
    if len(feature_classes) == 1:
        return 0

    child_entropies = [(sum(features[:,attribute_index]==feature_class) / len(targets)) * entropy(targets[features[:,attribute_index]==feature_class]) for feature_class in feature_classes]

    split_entropy = sum(child_entropies)

    return parent_entropy - split_entropy

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
