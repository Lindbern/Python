import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

#1.preprocessing state
#data processing
data = pd.read_csv("SkyServer.csv")

#1.1. Test and train set (one is using for modification and one that will be saved to use and test in the end)
#Data shall NOT be sorted -> could make porblems
#Shuffle and cut (!) -> random_state
train_data, test_data = train_test_split(data, test_size=0.2, random_state=52)

#1.2. fill empty or non exisitng values
# train_data.info() -> no nan values

#1.3. Encoding - Models in General only work with numerical values
#encoding is not needed since there are no non-numerical values except of "class"

#Before Scaler: Cut "class" from data -- create target and features
#features and target/labels from train data (for tree)
traintarget = train_data['class']
trainfeatures = train_data.drop(['class'], axis=1)

#features and targets/labels from test data (for prediction)
testtarget= test_data['class']
testfeatures = test_data.drop(['class'], axis=1)

#1.4.Normalize data to prevent dominance of certain features (features with different scaling can sometimes be dominant and manipulate the data)
#with StandardScaler ( mean value = 0, dev = 1)
scaler = StandardScaler().fit(trainfeatures)
trainfeatures = scaler.transform(trainfeatures)
testfeatures = scaler.transform(testfeatures)

#2. Decision Tree
#2.1 tree classifier
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(trainfeatures, traintarget)

#2.3 plotting the tree
tree.plot_tree(decision_tree, feature_names=list(train_data.drop('class', axis=1)), class_names=traintarget.unique(), filled=True, fontsize=4)
plt.show()


#3.Evaluation & Prediciton on test-data
Test_Prediction = decision_tree.predict(testfeatures)

print("classification \n:", classification_report(testtarget, Test_Prediction))
cf = confusion_matrix(testtarget, Test_Prediction)
print("confusion matrix:", cf)
print("accuracy score:", accuracy_score(testtarget, Test_Prediction) * 100)

#Prediction:
for i in range(len(Test_Prediction)):
    print("features", testfeatures[i])
    print("Prediction", Test_Prediction[i],"\n")
