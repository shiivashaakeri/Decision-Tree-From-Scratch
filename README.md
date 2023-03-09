# Decision Tree and Random Forest

This project has three parts:
1. Decision Tree from Scratch 
2. Random Forest from Scratch 
3. Random Forest using Sklearn Library


## Dataset
The dataset used in this project is `prison_dataset.csv`, which includes information about prisoners in a particular country. The dataset has 6 columns and 524 rows, containing the following information:
1. `age`: Age of the prisoner.
2. `gender`: Gender of the prisoner (Male or Female).
3. `education`: Education level of the prisoner (Primary, High School, University).
4. `criminal_past`: Whether the prisoner has a criminal past (Yes or No).
5. `time_served`: Time served in prison.
6. `parole`: Whether the prisoner is granted parole (Yes or No).
The aim of this project is to build a decision tree model to predict whether a prisoner is granted parole or not based on the information given in the dataset.

## Algorithms

### 1. Decision Tree from Scratch : `DecisionTree.py`
How it works

The algorithm works by recursively building the decision tree top-down. At each step, it selects the most informative feature by calculating the information gain of each feature. The information gain is calculated by comparing the entropy of the dataset before and after splitting by the feature. The algorithm stops when all the instances in a branch belong to the same class or the maximum depth of the tree is reached.
This is a Python code for building decision tree based on ID3 algorithm. The code is written in DecisionTree.py and the input dataset is `prison_dataset.csv`.
The `DecisionTree.py` includes the following functions:

- `trainTestSplit(data)`: Splits the input dataset into training and testing sets.
- `totalEntropy(data, classes)`: Calculates the total entropy of the dataset.
- `entropy(data, classes)`: Calculates the entropy of a given feature.
- `informationGain(colName, data, classes)`: Calculates the information gain of a given feature.
- `bestColForSplit(data, class_list)`: Determines the feature with the highest information gain and returns its name.
- `splitData(colName, data, classes, depth)`: Splits the dataset based on the given feature and returns a tree node and the updated dataset.
- `genTree(root, prevFeatureVal, data, classes, depth)`: Generates a decision tree recursively.
- `decisionTree(data)`: Builds the decision tree model and returns the root node.

