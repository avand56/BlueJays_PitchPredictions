import pandas as pd
import numpy as np
from datetime import timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit

# import csv files and drop NaNs
train= pd.read_csv('/Users/alexvanderhoeff/Downloads/training.csv').dropna()
test= pd.read_csv('/Users/alexvanderhoeff/Downloads/deploy.csv').dropna()

# split dataset
xtrain = train.drop(columns=['InPlay'])
ytrain = train.InPlay

# Create a based model
rf = RandomForestClassifier(
    n_estimators=100, 
    criterion='gini', 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features='sqrt', 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    bootstrap=True, 
    oob_score=False, 
    n_jobs=None, 
    random_state=20000233, 
    verbose=0, 
    warm_start=False, 
    class_weight=None, 
    ccp_alpha=0.0, 
    max_samples=None
    )

# fit the training data
rf.fit(xtrain, ytrain)

# Predict the model
predictions = rf.predict(test)
np.savetxt("predictions.csv", predictions, delimiter=",")
