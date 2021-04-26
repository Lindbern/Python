import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

#data processing
data = pd.read_csv("SkyServer.csv")

#split data to test and train data set
train_data, test_data = train_test_split(data, test_size=0.2, random_state=52)


#features and targets from train data (for tree)
traintarget = train_data['class']
trainfeatures = train_data.drop(['class'], axis=1)


#features and targets from test data (for prediction)
testtarget= test_data['class']
testfeatures = test_data.drop(['class'], axis=1)


#Scaler
scaler = StandardScaler().fit(trainfeatures)
trainfeatures = scaler.transform(trainfeatures)
testfeatures = scaler.transform(testfeatures)

#tree classifier
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(trainfeatures, traintarget)

#plotting the tree
tree.plot_tree(decision_tree, feature_names=list(trainfeatures), class_names=traintarget.unique(), filled=True, fontsize=3)
plt.show()


#Evaluation & Prediciton
Test_Prediction = decision_tree.predict(testfeatures)

print("classification \n:", classification_report(testtarget, Test_Prediction))
cf = confusion_matrix(testtarget, Test_Prediction)
print("confusion matrix:", cf)
print("accuracy score:", accuracy_score(testtarget, Test_Prediction) * 100)

for i in range(len(Test_Prediction)):
    print("features", testfeatures[i])
    print("Prediction", Test_Prediction[i],"\n")
