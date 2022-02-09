from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('prison_dataset.csv')
rowNum = df.shape[0]
splitIndex = int(80 / 100 * rowNum)



"""
Train the model
"""
#Encode the feature values which are strings to integers
for label in df.columns:
    df[label] = LabelEncoder().fit(df[label]).transform(df[label])


target = df.drop(['Recidivism - Return to Prison numeric'],axis=1)
fatures = df['Recidivism - Return to Prison numeric']
print(fatures.value_counts())

train_targets, test_targets, train_features, test_features = train_test_split(target, fatures, random_state = 1, stratify=fatures)

forest = RandomForestClassifier(max_depth=3, random_state=1, criterion='entropy')
forest.fit(train_targets, train_features)

y_pred_test = forest.predict(test_targets)
print('accuracy: ', accuracy_score(test_features, y_pred_test))

print(confusion_matrix(test_features, y_pred_test))




