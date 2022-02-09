from numpy.lib.shape_base import split
import pandas as pd
import numpy as np
import scipy.stats as sps

def trainTestSplit(data):
    rowNum = data.shape[0]
    splitIndex = int(80 / 100 * rowNum)
    train = data.iloc[:splitIndex].reset_index(drop=True)#We drop the index respectively relabel the index
    #starting form 0, because we do not want to run into errors regarding the row labels / indexes
    test = data.iloc[splitIndex:].reset_index(drop=True)
    return train, test

def totalEntropy(data, label, classes):
    rowNum = data.shape[0] #the total size of the dataset
    entropy = 0
    
    for c in classes: #for each class in the label
        classNum = data[data[label] == c].shape[0] #number of the class
        classEntropy = - (classNum / rowNum) * np.log2(classNum / rowNum) #entropy of the class
        entropy += classEntropy #adding the class entropy to the total entropy of the dataset
    
    return entropy

def entropy(data, label, classes):
    rowNum = data.shape[0]
    entropy = 0
    
    for c in classes:
        classNum = data[data[label] == c].shape[0] #row count of class c 
        classEntropy = 0
        if classNum != 0:
            classProb = classNum / rowNum #probability of the class
            classEntropy = - classProb * np.log2(classProb)  #entropy
        entropy += classEntropy
    return entropy

def informationGain(colName, data, label, classes):
    colVals = np.unique(data[colName]) #unqiue values of the feature
    rowNum = data.shape[0]
    colInformation = 0.0
    
    for val in colVals:
        valData = data[data[colName] == val] #filtering rows with that feature_value
        valNum = valData.shape[0]
        valEntropy = entropy(valData, label, classes) #calculcating entropy for the feature value
        valProbabilty = valNum / rowNum
        colInformation += valProbabilty * valEntropy #calculating information of the feature value
        
    return totalEntropy(data, label, classes) - colInformation #calculating information gain by subtracting

def bestColForSplit(data, label, class_list):
    features = data.columns.drop(label) #finding the feature names in the dataset #N.B. label is not a feature, so dropping it
    maxInfoGain = -1
    maxInfofeature = None
    
    for feature in features:  #for each feature in the dataset
        featureInfoGain = informationGain(feature, data, label, class_list)
        if maxInfoGain < featureInfoGain: #selecting feature name with highest information gain
            maxInfoGain = featureInfoGain
            maxInfofeature = feature
            
    return maxInfofeature

def splitData(colName, data, label, classes, depth):
    colValsAndCounts = data[colName].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tree or node
    
    for val, count in colValsAndCounts.iteritems():
        valData = data[data[colName] == val] #dataset with only feature_name = feature_value
        maxClassesName = ''
        maxClassesNum = 0

        assigned = False #flag for tracking feature_value is pure class or not
        for c in classes: #for each class
            classCount = valData[valData[label] == c].shape[0] #count of class c

            if classCount == count: #count of feature_value = count of class (pure class)
                tree[val] = c #adding node to the tree
                data = data[data[colName] != val] #removing rows with feature_value
                assigned = True
            
            if classCount > maxClassesNum:
                maxClassesNum = classCount
                maxClassesName = c

        if not assigned: #not pure class
            if depth != 3:
                tree[val] = "?" #should extend the node, so the branch is marked with ?
            else:
                tree[val] = maxClassesName
            
    return tree, data

def genTree(root, prevFeatureVal, data, label, classes, depth):
    if depth == 4:
        return

    if data.shape[0] != 0: #if dataset becomes enpty after updating
        maxInfoFeature = bestColForSplit(data, label, classes) #most informative feature
        tree, data = splitData(maxInfoFeature, data, label, classes, depth) #getting tree node and updated dataset
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
                genTree(nextRoot, node, featureValData, label, classes, depth+1) #recursive call with updated dataset

def ID3(data, labelCol):
    tree = {} #tree which will be updated
    classes = data[labelCol].unique() #getting unqiue classes of the label
    genTree(tree, None, data, labelCol, classes, 0) #start calling recursion
    return tree


df = pd.read_csv('prison_dataset.csv')

colNum = df.shape[1]
colNames = list(df.columns)

df = df.sample(frac = 1).reset_index(drop=True)

trainData, testData = trainTestSplit(df)
labelCol = 'Recidivism - Return to Prison numeric'
tree = ID3(trainData.copy(), 'Recidivism - Return to Prison numeric')

# print(tree)

def predict(query, tree, default = 1):
        for key in list(query.keys()):
            if key in list(tree.keys()):
                result = tree[key][query[key]]
                if isinstance(result, dict):
                    return predict(query, result)
                else:
                    return result

def test(data, tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    Xtest = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(Xtest[i], tree, 1.0) 
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data['Recidivism - Return to Prison numeric'])/len(data))*100,'%')

# test(testData, tree)

def randomForest(dataset, label):
    #Create a list in which the single forests are stored
    subTree = []
    
    #Create a number of n models
    for i in range(7):
        #Create a number of bootstrap sampled datasets from the original dataset 
        sample = dataset.sample(frac=0.8)

        labels = sample[[label]]
        features = sample.drop(label, 1)

        features = features.sample(frac=0.8, axis=1)
        
        sample = features.join(labels).reset_index(drop=True)
        #Create a training and a testing datset by calling the train_test_split function
        sampleTrain = trainTestSplit(sample)[0]
        sampleTest = trainTestSplit(sample)[1] 
        
        
        #Grow a tree model for each of the training data
        #We implement the subspace sampling in the ID3 algorithm itself. Hence take a look at the ID3 algorithm above!
        subTree.append(ID3(sampleTrain, 'Recidivism - Return to Prison numeric'))
        
    return subTree

def confusionMatrix(predicted, labels):
    mat = np.zeros((2, 2), dtype=np.int32)

    predictedNP = np.array(predicted)
    labelsNP = np.array(labels)

    for i in range(len(predictedNP)):
        if predictedNP[i] == 1:
            if labelsNP[i] == 1:
                mat[0][0] += 1
            elif labelsNP[i] == 0:
                mat[0][1] += 1
        elif predictedNP[i] == 0:
            if labelsNP[i] == 1:
                mat[1][0] += 1
            elif labelsNP[i] == 0:
                mat[1][1] += 1
    
    print(mat)

rand_forest = randomForest(trainData.copy(), 'Recidivism - Return to Prison numeric')
print(rand_forest)

def RandomForest_Predict(query,random_forest,default='p'):
    predictions = []
    for tree in random_forest:
        predictions.append(predict(query, tree, default))
    return max(predictions, key=predictions.count) # majority voting


query = testData.iloc[0,:].drop('Recidivism - Return to Prison numeric').to_dict()
query_target = testData.iloc[0,0]
prediction = RandomForest_Predict(query,rand_forest)
print('prediction: ',prediction)

def RandomForest_Test(data,random_forest):
    data['predictions'] = None
    for i in range(len(data)):
        query = data.iloc[i,:].drop('Recidivism - Return to Prison numeric').to_dict()
        data.loc[i,'predictions'] = RandomForest_Predict(query,random_forest,default='p')
    accuracy = sum(data['predictions'] == data['Recidivism - Return to Prison numeric'])/len(data)*100
    print('The prediction accuracy is: ',sum(data['predictions'] == data['Recidivism - Return to Prison numeric'])/len(data)*100,'%')
    confusionMatrix(data['predictions'], data[labelCol])
    return accuracy

RandomForest_Test(testData, rand_forest)