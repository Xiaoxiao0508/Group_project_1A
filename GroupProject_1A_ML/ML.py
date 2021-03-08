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
# loading data
player=pd.read_csv('2010-11.csv',sep=',')
print(player.info())
# processing data
bins=(0.1,0.5,1.0)

group_names=['low_ratio','high_ratio']
player['W_PCT']=pd.cut(player['W_PCT'], bins=bins,labels=group_names)
# print(player['W_PCT'].unique())
label_ratio=LabelEncoder()
player['W_PCT']=label_ratio.fit_transform(player['W_PCT'].astype(str))
# print(player.head(20))
print(player['W_PCT'].count())
print(sns.countplot(x=player['W_PCT']))