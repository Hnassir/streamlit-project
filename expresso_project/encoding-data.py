import pandas as pd

#from scipy.stats import zscore
#import numpy as np

from sklearn.preprocessing import LabelEncoder



#====================

data=pd.read_csv('../data/interim/expresso_feature_engineering.csv')

#=====================

#new_data=data[(np.abs(zscore(data.select_dtypes(exclude='object')))<3).all(axis=1)]

#================

encoder=LabelEncoder()

catg_cols=data.select_dtypes(include='object').columns

for col in catg_cols:
    data.loc[:,col]=encoder.fit_transform(data[col])

#================================

data.to_csv('../data/processed/expresso_encoded.csv',index_label=False)
