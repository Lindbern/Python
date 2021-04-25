import pandas as pd
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

#data processing

data = pd.read_csv("SkyServer.csv")
#blalba
# No nan values
data.fillna(0, inplace=True) # @alec can i use this aswell?

# Dropping columns bc data is not sequential
data.drop(['objid', 'specobjid'], axis=1, inplace=True)

# Dataset is unbalenced, We have very little QSO the model could struggle classifying this @alec - what is this
sns.countplot(x='class', data=data)

#split data to test and train
train_data, test_data = train_test_split(data, test_size=0.2, random_state=52)
data = train_data
print(data.head())


train = train_data['class']
print(train)

traindrop = train_data.drop(['class'], axis=1)
print(traindrop)

test = test_data['class']
testdrop = test_data.drop(['class'], axis=1)


#Scaler
scaler = StandardScaler().fit(traindrop)
traindrop = scaler.transform(traindrop)
testdrop = scaler.transform(testdrop)

#tree classifier
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(traindrop,train)

#plotting the tree
tree.plot_tree(decision_tree, feature_names=list(traindrop), class_names=data["class"].unique(), filled=True, fontsize=4)
plt.show()


#Evaluation & Prediciton
Test_Prediction = DecisionTreeClassifier.predict(testdrop)

print(classification_report(test, Test_Prediction))
cf = confusion_matrix(test, Test_Prediction)
print(cf)
print(accuracy_score(test, Test_Prediction) * 100)

for i in range(len(Test_Prediction)):
    print(f"X= {testdrop[i]} \n PREDICTED= {Test_Prediction[i]} \n")

target = data['class'].values
target = pd.get_dummies(target, drop_first=True)