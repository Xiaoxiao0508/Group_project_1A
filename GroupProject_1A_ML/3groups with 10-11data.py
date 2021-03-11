import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
#step1: loading data
player=pd.read_csv('2010-11.csv',sep=',')
print(player.info())
print(player.head(20))
# step2:processing data
bins=(-0.1,0.5,0.7,1.0)

group_names=['low','medium','high']
player['W_PCT']=pd.cut(player['W_PCT'], bins=bins,labels=group_names)
print(player['W_PCT'].unique())
label_ratio=LabelEncoder()
player['W_PCT']=label_ratio.fit_transform(player['W_PCT'].astype(str))
print(player.head(20))
print(player['W_PCT'].count())
print(sns.countplot(x=player['W_PCT']))
# step 3, seperate the dataset as res
# ponse variable and feature variables, get ready for model
X=player.drop('W_PCT',axis=1)
y=player['W_PCT']
# step 4,Use train_test_split(package) to split the data to  Train data and Test data, default test_size 25%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#step 5 ,Scale up the data Applying Standart scaling to get optimized result eg, big number overweight the impact of small number
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# X_test[:10]
# step 6,choose classifier
# RANDOM FOREST -Least amount of parts to fine-tune,
# used for a medium sized data set
# 1,create randomForest variable,2,fit the training data to it,3 predict
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)
# X_test[:20]
# the test data print out have been sacled
pred_rfc[:20]
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test,pred_rfc))
#  step 7, use the choosen model and given new feature data to predict 
Xnew=[[20,18,3.6,9.9,0.36,1.2,2,0.28,2,2,0.7,0.2,0,1,2,1,0.5,0.1,0.5,1.3,2,9,-2,15,0,0]]
Xnew=sc.transform(Xnew)
ynew=rfc.predict(Xnew)
print(ynew)