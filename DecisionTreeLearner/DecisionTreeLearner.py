"""
This snippet contains an implementation of a decision tree regressor, following the approach of Quinlan's 1986 paper.
In this approach, a table is built that reflects the structure of the decision tree. After the model is trained (i.e.
the table is built), each query involves a look-up in the table until reaching a leaf node that will represent the
predicted value.

The build_tree() method reflects the training of the decision tree (building the table).
The query_tree() method reflects the look-up performed at inference time.
"""

import numpy as np  		  	   		   	 			  		 			 	 	 		 		 	


class DecisionTree(object):

    def __init__(self, leaf_size=10):
        self.leaf_size = leaf_size

    def build_tree(self, data_x, data_y):
        """
        Recursively builds the Decision Tree Table.

        The 2 base cases are when:
            * a leaf node is reached
            * all y values are the same and there is no point in further splitting the data

        The recursive part of the method:
            * calculates which is the feature on which to perform a split. In this implementation, the feature with the
              highest correlation to y is chosen.
            * calculates the split-value as the median of the values of the split-feature.
            * splits the data in 2, based on which points are lower or greater than the split-value
            * appends to the table the root, the left tree, and the right tree

        Every row in the table represent a tree node. For a decision node, the row contains:
        node ID | split-value | the relative row reference for the left child | the relative row refernce for the right child.
        For a leaf node, the row contains:
        NaN | Predicted value | NaN | NaN
        """

        if data_x.shape[0] <= self.leaf_size:
            return np.array([[np.nan, np.mean(data_y), np.nan, np.nan]])
        elif np.all(data_y == data_y[0]):
            return np.array([[np.nan, data_y[0], np.nan, np.nan]])
        else:
            corrs = np.corrcoef(data_x, data_y, rowvar=False)[0:-1, -1]
            i = np.argmax(np.abs(corrs))
            splitVal = np.median(data_x[:, i])
            maskLeft = data_x[:, i] <= splitVal

            # This changes the feature when we cannot split on the best correlated one
            nextFeature = 2
            while (np.all(maskLeft == maskLeft[0])) and (nextFeature <= len(corrs)):
                i = np.argsort(np.abs(corrs))[-nextFeature]
                nextFeature += 1
                splitVal = np.median(data_x[:, i])
                maskLeft = data_x[:, i] <= splitVal

            if np.all(maskLeft == maskLeft[0]):
                leftTree = self.build_tree(np.mean(data_x, axis=0), np.array([np.mean(data_y, axis=0)]))
            else:
                leftTree = self.build_tree(data_x[maskLeft], data_y[maskLeft])

            maskRight = data_x[:, i] > splitVal
            if np.all(maskRight == maskRight[0]):
                rightTree = self.build_tree(np.mean(data_x, axis=0), np.array([np.mean(data_y, axis=0)]))
            else:
                rightTree = self.build_tree(data_x[maskRight], data_y[maskRight])

            root = np.array([[i, splitVal, 1, leftTree.shape[0] + 1]])

            return np.append(np.append(root, leftTree, axis=0), rightTree, axis=0)

    def add_training_data(self, data_x, data_y):
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Adds training data to the learner.

        Input:
        data_x: A set of feature values used to train the learner.
        data_y: The value corresponding to the X data.
        """
        self.table = self.build_tree(data_x, data_y)

    def query_tree(self, point, queryIndex):
        """
        Recursively search the table for the regression value for the given data point.

        The base case is when a leaf-node is reached.

        In the recursive part of the method the tree (table) is searched:
            * compare the query point value to the split-value
            * if the query point value is lower, jump to the left child
            * if the query point value is higher, jump to the right child
        """
        if ~np.isfinite(self.table[queryIndex, 0]):
            return np.array([self.table[queryIndex, 1]])
        else:
            if point[int(self.table[queryIndex, 0])] <= self.table[queryIndex, 1]:
                return self.query_tree(point, queryIndex + int(self.table[queryIndex, 2]))
            else:
                return self.query_tree(point, queryIndex + int(self.table[queryIndex, 3]))

    def query(self, points):
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Estimate a set of test points based on the model built.

        Input:
        points: A numpy array with each row corresponding to a specific query.

        Output:
        y_pred: The predicted result of the input data according to the trained model.
        """
        y_pred = np.empty((len(points)))
        for i in range(0, len(points[:, 0])):
            y_pred[i] = self.query_tree(points[i, :], 0)

        return y_pred
