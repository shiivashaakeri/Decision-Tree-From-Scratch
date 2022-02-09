from numpy.lib.shape_base import split
import pandas as pd
import numpy as np


def trainTestSplit(data):
    rowNum = data.shape[0]
    splitIndex = int(80 / 100 * rowNum) #we use 80% of datas for training
    train = data.iloc[:splitIndex].reset_index(drop=True) #reset indexes to start from index 0
    test = data.iloc[splitIndex:].reset_index(drop=True)
    return train, test

def totalEntropy(data, classes):
    rowNum = data.shape[0] #total size of the dataset
    entropy = 0
    
    for c in classes: #for each class in the label
        classNum = data[data[labelCol] == c].shape[0] #number of the class
        classEntropy = - (classNum / rowNum) * np.log2(classNum / rowNum) #entropy of the class
        entropy += classEntropy #adding the class entropy to the total entropy of the dataset
    
    return entropy

def entropy(data, classes):
    rowNum = data.shape[0]
    entropy = 0
    
    for c in classes:
        classNum = data[data[labelCol] == c].shape[0] #row count of class c 
        classEntropy = 0
        if classNum != 0:
            classProb = classNum / rowNum #probability of the class
            classEntropy = - classProb * np.log2(classProb)  #entropy
        entropy += classEntropy
    return entropy

def informationGain(colName, data, classes):
    colVals = np.unique(data[colName]) #unqiue values of the feature
    rowNum = data.shape[0]
    colInformation = 0.0
    
    for val in colVals:
        valData = data[data[colName] == val] #filtering rows with the value of that feature
        valNum = valData.shape[0]
        valEntropy = entropy(valData, classes) #calculcating entropy for the feature value
        valProbabilty = valNum / rowNum
        colInformation += valProbabilty * valEntropy #calculating information of the feature value
        
    return totalEntropy(data, classes) - colInformation #calculating information gain by subtracting

def bestColForSplit(data, class_list):
    features = data.columns.drop(labelCol) #finding the feature names in the dataset #N.B. label is not a feature, so dropping it
    maxInfoGain = -1
    maxInfofeature = None
    
    for feature in features:  #for each feature in the dataset
        featureInfoGain = informationGain(feature, data, class_list)
        # print(feature , ":" , featureInfoGain)
        if maxInfoGain < featureInfoGain: #selecting feature name with highest information gain
            maxInfoGain = featureInfoGain
            maxInfofeature = feature
            
    return maxInfofeature

def splitData(colName, data, classes, depth):
    colValsAndCounts = data[colName].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tsree or node
    
    for val, count in colValsAndCounts.iteritems():
        valData = data[data[colName] == val] #dataset with only feature_name = feature_value
        maxClassesName = ''
        maxClassesNum = 0

        assigned = False #flag for tracking value of feature is pure class or not
        for c in classes: #for each class
            classCount = valData[valData[labelCol] == c].shape[0] #count of class c

            if classCount == count: #count of feature_value = count of class (pure class)
                tree[val] = c #adding node to the tree
                data = data[data[colName] != val] #removing rows with feature_value
                assigned = True
            
            if classCount > maxClassesNum:
                maxClassesNum = classCount
                maxClassesName = c

        if not assigned: #not pure class
            if depth != 2:
                tree[val] = "?" #should extend the node, so the branch is marked with ?
            else:
                tree[val] = maxClassesName
            
    return tree, data

def genTree(root, prevFeatureVal, data, classes, depth):
    if depth == 3:
        return

    if data.shape[0] != 0: #if dataset becomes enpty after updating
        maxInfoFeature = bestColForSplit(data, classes) #most informative feature
        # print(maxInfoFeature)
        tree, data = splitData(maxInfoFeature, data, classes, depth) #getting tree node and updated dataset
        # print(tree)
        nextRoot = None
        
        if prevFeatureVal != None: #add to intermediate node of the tree
            root[prevFeatureVal] = dict()
            root[prevFeatureVal][maxInfoFeature] = tree
            nextRoot = root[prevFeatureVal][maxInfoFeature]
        else: #add to root of the tree
            root[maxInfoFeature] = tree
            nextRoot = root[maxInfoFeature]
        
        for node, branch in list(nextRoot.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                featureValData = data[data[maxInfoFeature] == node] #using the updated dataset
                genTree(nextRoot, node, featureValData, classes, depth+1) #recursive call with updated dataset

def decisionTree(data):
    tree = {} #tree which will be updated
    classes = data[labelCol].unique() #getting unqiue classes of the label
    genTree(tree, None, data, classes, 0) #start calling recursion
    return tree


df = pd.read_csv('prison_dataset.csv') #load dataset

colNum = df.shape[1] #number of columns(features)
colNames = list(df.columns) #names of columns(features)

df = df.sample(frac = 1).reset_index(drop = True) #shuffle rows

trainData, testData = trainTestSplit(df) #spliting train-data and test-data

labelCol = 'Recidivism - Return to Prison numeric' #target feature
tree = decisionTree(trainData.copy())

# print(tree)

def predict(features, tree): #
        for key in list(features.keys()):
            if key in list(tree.keys()):
                result = tree[key][features[key]]
                if isinstance(result, dict):
                    return predict(features, result)
                else:
                    return result

def confusionMatrix(predicted, labels): #confusion matrix 
    mat = np.zeros((2, 2), dtype=np.int32)

    predictedNP = np.array(predicted)
    labelsNP = np.array(labels)

    for i in range(len(predictedNP)):
        if predictedNP[i] == 1: 
            if labelsNP[i] == 1:#TP
                mat[0][0] += 1
            elif labelsNP[i] == 0:#FP
                mat[0][1] += 1
        elif predictedNP[i] == 0:
            if labelsNP[i] == 1:#TN
                mat[1][0] += 1
            elif labelsNP[i] == 0:#FN
                mat[1][1] += 1
    
    print(mat)

def test(data, tree):
    #removing the target feature column from the original dataset and convert it to a dictionary
    Xtest = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create an empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(Xtest[i], tree)

    
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data[labelCol])/len(data))*100,'%') #calculating accuracy
    confusionMatrix(predicted, data[labelCol])

test(testData, tree)
