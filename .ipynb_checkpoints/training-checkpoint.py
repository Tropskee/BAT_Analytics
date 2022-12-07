import pandas as pd
import numpy as np
import pickle
from numpy.random import RandomState
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import lightgbm as ltb
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBRegressor

## Read CSV
data_source = r"./src/vehicle_data/bat_training_data.csv"
df = pd.read_csv(data_source)
target_feature = "Final Bid Price"

print(f'Data ingested -- Found {len(df)} Rows')

y = df[target_feature]
X = df.drop(target_feature, axis=1)

## Find Numerical & Categorical Columns
# Get numerical and categorical feature columns
print('\nCalculating Numerical and Categorical Features...')
print(f'There are {len(X.columns)} total columns.')

numerical_features = X.select_dtypes(include='number').columns.tolist()
print(f'There are {len(numerical_features)} numerical features.')

categorical_features = X.select_dtypes(exclude='number').columns.tolist()
print(f'There are {len(categorical_features)} categorical features.', '\n')

print(X.columns)

## Pre-Process Data

print('Fetching Preprocessing Pipeline...\n')

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean', missing_values=np.nan)),
    ('scale', MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_features),
    ('category', categorical_pipeline, categorical_features)
])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

model = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.05, verbosity=3)
# model = ltb.LGBMClassifier(random_state=42, n_estimators=1000, max_depth=3, learning_rate=0.05)

model_pipeline = Pipeline(steps=[
        ('preprocessor', full_processor),
        ('model', model)
    ])

full_processor.fit_transform(X_train)
X_valid_transformed = full_processor.transform(X_valid)
fit_params = {"model__eval_set": [(X_valid_transformed, y_valid)], 
              "model__early_stopping_rounds": 5,
              "model__verbose": True}

# print(len(X_train), len(y_train))
# Preprocessing of training data, fit model 
model_pipeline.fit(X_train, y_train, **fit_params)

# Preprocessing of validation data, get predictions
preds = model_pipeline.predict(X_valid)

# Evaluate the model
print('MAE:', mean_absolute_error(y_valid, preds))

# X_processed = full_processor.fit_transform(X)
# print(X_processed)

## Train/Test Split

# X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.30, random_state=42)

## LightGBM Model Pipeline

# print('Creating LightGBM Model...\n')

# # Run model without params to perform grid search
# if do_grid_search:
#     model = ltb.LGBMClassifier(seed=42, verbose=1)

# # Run model with specified params
# else: 
#     # model = ltb.LGBMClassifier(seed=42, verbose=1, n_estimators = n_estimators, learning_rate = lr, max_depth = max_depth)
#     model = ltb.LGBMClassifier(random_state=42, n_estimators=n_estimators, num_leaves=64, max_depth=5, learning_rate=lr, n_jobs=-1)

# # Create model pipeline including preprocessor
# ltb_pipeline = Pipeline(steps=[
#     # ('preprocess', preprocessor),
#     ('model', model)
# ])

# print('Fitting XGBoost Model...\n')
# return ltb_pipeline

## Grid Search

# model = ltb.LGBMClassifier(seed=42, verbosity=2)

# param_dict = {
#     'model__learning_rate': [0.01, 1, 0.1],
#     'model__max_depth': range (1, 5, 1),
#     'model__n_estimators': range(60, 200, 40),
#     'model__early_stopping_rounds' : [30]
# }

# print('Starting Grid Search...\n')
# search = GridSearchCV(model, param_dict, 
#                   cv=3, 
#                   scoring='neg_mean_absolute_error',
#                   verbose=3)

# search.fit(X_train, y_train)

# print('Best score:', abs(search.best_score_))

# print('Best params:', search.best_params_)

# print('Best estimator:', search.best_estimator_)

# return search.best_params_.values()

## Training

# # Create and fit model
# lgb_model = ltb.LGBMClassifier(random_state=42, n_estimators=100, num_leaves=64, max_depth=5, learning_rate=0.1, n_jobs=-1, verbosity=3)
# improved_model = lgb_model.fit(X_train, y_train) 

# # Get predictions and output results
# preds = improved_model.predict(X_valid)
# MAE = mean_absolute_error(y_valid, preds)
# r2 = improved_model.score(X_valid, y_valid)
# print('MAE:', MAE)
# print('R2:', r2)

# # Create and fit model
# xgbr = XGBRegressor(seed=42, max_depth=3, learning_rate=0.1, verbosity=3)
# improved_model = xgbr.fit(X_train, y_train) 

# # Get predictions and output results
# preds = improved_model.predict(X_valid)
# MAE = mean_absolute_error(y_valid, preds)
# r2 = improved_model.score(X_valid, y_valid)
# print('MAE:', MAE)
# print('R2:', r2)

