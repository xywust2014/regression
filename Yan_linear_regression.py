# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:40:21 2019

@author: peter
"""
# Useful Techniques 
# (1) list(pd) can get the columns of that dataframe 
# (2) pd.values can turn a dataframe into a numpy array 
# (3) ppl =  Pipeline([('a', method1()), ('b', method2()) ,..., ('c', method3())])
# (4) ppl.fit_transform()
# (5) fit transform and fit_transform 

### Yan linear regression 

import numpy as np 
import pandas as pd 
#from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVR 
import time 


data = pd.read_csv("housing.csv")
#y = np.array(data["median_house_value"])
#data.drop("median_house_value", axis = 1, inplace = True)

data["income_cat"] = np.ceil(data["median_income"]/1.5)
data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace = True)
#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, random_state = 42)
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(data, data["income_cat"]):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]
y_train =  train_set["median_house_value"]   
X_train = train_set.drop("median_house_value", axis = 1)
y_test = test_set["median_house_value"]
X_test = test_set.drop("median_house_value", axis = 1)
# Select Certain Features 
class FeatureSelector(object):
    def __init__(self, feature):
        # feature_columns: a list of column names 
        self.feature = feature
        
    def fit(self, X, y = None):
        return self  
    
    def transform(self, X):
        return X[self.feature].values
    
# Feature Engineering to get artificial features 
total_rooms_ix, total_bedrooms_ix, popupation_ix, total_households_ix = 3, 4, 5, 6
class Feature_Engineer(object):
    def __init__(self):
        pass 
    
    def fit(self, X, y = None):
        return self 
    
    def transform(self, X, y = None):
        Rooms_per_house = X[:,total_rooms_ix]/X[:,total_households_ix]
#        Bedrooms_per_house = X[:,total_bedrooms_ix]/X[:,total_households_ix]
        Pop_per_house = X[:,popupation_ix]/X[:,total_households_ix]
        return np.c_[X, Rooms_per_house,  Pop_per_house]

feature_import_rank = ['median_income','INLAND','Pop_per_house','income_cat',
                       'longitude','latitude','housing_median_age','Rooms_per_house',
                       'total_rooms','total_bedrooms','population','households',
                       '<1H OCEAN','NEAR OCEAN','NEAR BAY','ISLAND']
feature_import_ix = [7, 12, 8, 10, 0, 1, 2, 9, 4, 6, 3, 5, 11, 15, 14, 13]

from sklearn.base import BaseEstimator, TransformerMixin 
class Select_import_features(BaseEstimator, TransformerMixin):
    # import_features is a list that contains the most important attributes 
    def __init__(self, K):
        self.K = K  
    
    def fit(self, X, y = None):
        return self 
    
    def transform(self, X, y = None):
        ix = feature_import_ix[0:self.K]
        return X[:,ix] 
    

# Create a number pipeline to get the number data 
housing_num = X_train.drop("ocean_proximity", axis = 1)
num_feas = list(housing_num)
steps = [('Add_features', FeatureSelector(num_feas)),
         ('imputer', SimpleImputer( strategy = 'median')),
         ('Feature_eng', Feature_Engineer()), 
         ('std_scaler', StandardScaler()), 
         ]
num_ppl = Pipeline(steps)

# Create a catogorical pipeline to do one-hot encoding to get categorical data 
# K represents the number of important features that we would like to select. 
K = 8
steps_c = [('select_f', FeatureSelector(['ocean_proximity'])), 
        ('One_hot', OneHotEncoder(sparse = False)), 
        ]
cat_ppl = Pipeline(steps_c)

# Feature Union which combines two pipeline together 
final_ppl = FeatureUnion(transformer_list = [
        ("number_pipeline", num_ppl), 
        ("category_pipeline", cat_ppl), 
        ])

fea_import_final_ppl = Pipeline([
                ("final_ppl", final_ppl), 
                ("sel_f", Select_import_features(K)), 
                ])


X_train_prepared = fea_import_final_ppl.fit_transform(X_train)
X_test_prepared = fea_import_final_ppl.transform(X_test)
X_train_prepared = final_ppl.fit_transform(X_train)
X_test_prepared = final_ppl.transform(X_test)
# =============================================================================
# param_grid = [
#         {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 5]}, 
#         {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2, 4, 5]}, 
#         ]
# 
# start_time = time.time()
# grid_search = GridSearchCV(reg, param_grid, cv = 5, scoring = "neg_mean_squared_error")
# grid_search.fit(X_train_prepared, y_train)
# best_rf_model = grid_search.best_estimator_
# grid_search.best_params_
# feature_import = grid_search.best_estimator_.feature_importances_
# end_time = time.time()
# print("Grid Search time is :", end_time - start_time)
# 
# extra_attrib = ["Rooms_per_house",  "Pop_per_house"]
# cat_encoder = list(cat_ppl.named_steps["One_hot"].categories_[0])
# attributes = num_feas + extra_attrib + cat_encoder 
# list_a = [i for i in range(0, len(attributes))]
# feature_rank = sorted(zip(feature_import, attributes, list_a), reverse = True)
# fea_import = [x[1] for x in feature_rank]
# 
# best_rf_model = grid_search.best_estimator_
# prediction = best_rf_model.predict(X_test_prepared)
# print("RMSE is: ", np.sqrt(mean_squared_error(prediction, y_test)))
# =============================================================================
# Best random Forest prediction score is 48066, training time grid search is 33 seconds 

#### Try SVM Regressor with linear and RBF 
start_time = time.time()
svm = SVR(kernel = "linear") 
param_grid = {
        'C':[100, 1000, 10000, 30000],
        'gamma':[0.000001, 0.00001, 0.0001,],
        }
grid_search = GridSearchCV(svm, param_grid, scoring = 'neg_mean_squared_error', cv = 5)
grid_search.fit(X_train_prepared, y_train)
end_time = time.time()
print("Training time for SVR is: ", end_time - start_time)


# Best parameters "linear", gamma = 10^-6, C = 30000 
# Linear is better than RBF 
# Best Support Vector Regressor is 68617, and training time is 1656.4 seconds .
#### RandomizedSearchCV with SVR 

start_time = time.time() 
svm = SVR(kernel = "linear")
param_grid = {
        'C':[100, 1000, 10000, 30000],
        'gamma':[0.000001, 0.00001, 0.0001,],
        }
rand_search = RandomizedSearchCV(svm, param_grid, scoring = 'neg_mean_squared_error', cv = 5)
rand_search.fit(X_train_prepared, y_train)
end_time = time.time()
print("Training time for Randomized Search is:", end_time - start_time)
prediction = rand_search.predict(X_test_prepared)
print("RMSR is: ", np.sqrt(mean_squared_error(y_test, prediction)))

# Best parameters "linear", gamma = 10^-6, C = 30000 
# Best score is , and training time is 508 seconds. 
# Randomized Search is faster compared with Grid Search 

#### Q3: Add a transfomer that select only the most important attributes 
# If we select only the top five most import features, 
# For Random Forest model: The RMSE is 57550, which is worse than not selecting the most important features 
# For SVM: The RMSE is worse than selecting the top K important features 

#### Q4: Try creating a single pipeline that does full data preparation and prediction 
# Try search the best parameter 
start_time = time.time()
param_grid = {
         'sel_f__K': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
         'pre__n_estimators': [3, 10, 30], 
         'pre__max_features': [2, 4, 5], 
         'pre__bootstrap': [True, False],
          }

regressor = RandomForestRegressor(random_state = 42)
ppl_predict = Pipeline([
         ("final_ppl", final_ppl), 
         ("sel_f", Select_import_features(K)), 
        ("pre", regressor), 
        ])
rand_search = GridSearchCV(ppl_predict, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
rand_search.fit(X_train, y_train)
prediction = rand_search.best_estimator_.predict(X_test)
end_time = time.time()
print("RMSE is: ", np.sqrt(mean_squared_error(prediction, y_test)))
print("Random Search Time is: ", end_time - start_time)
# Random Search time is 38.71, RMSE = 48001.43, best_params: True, max_feature:5, n_estimators: 30 
# Grid Search time is 90.87, RMSE = 48228, best_params: False, max_features:2, n_estimators: 30

#### Q5: explore some preparation options using GridSearchCV 



