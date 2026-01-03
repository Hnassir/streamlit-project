import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib

#============================

data=pd.read_csv('../data/processed/expresso_encoded.csv')

#===================================

x_features=data.drop(columns='CHURN')
y_target=data['CHURN']

#=======================

normalization_scaler=StandardScaler()

x_norm=normalization_scaler.fit_transform(x_features)


#==================

x_train,x_test,y_train,y_test=train_test_split(x_norm,y_target,test_size=0.2,random_state=42)

#====================

tree_alg=DecisionTreeClassifier(max_depth=35,max_leaf_nodes=35)

tree_alg.fit(x_train,y_train)

#==============================

y_pred=tree_alg.predict(x_test)

accuracy_s=accuracy_score(y_test,y_pred)

#==============

joblib.dump(tree_alg,"decisiontree.pkl")

