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
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


def get_data_and_split(data_source, target_feature):
    '''
    Read in data from a .csv and return a train/validate split

    :input data_source str: Filepath to .csv file
    :input target_feature str: Column name of feature you want to predict
    :return X_train, X_valid, y_train, y_valid: Train/Validation parameters
    '''
    df = pd.read_csv(data_source)
    print(f'Data ingested -- Found {len(df)} Rows')

    y = df[target_feature]
    X = df.drop(target_feature, axis=1)

    return train_test_split(X, y, test_size=0.30, random_state=42)

def explore_data(X_train):
    '''
    Data exploration and read outs
    
    :input X_train dataframe: Training data
    '''

    print('Numerical Cols:\n', X_train.describe().T.iloc[:10]) # All numerical cols
    # X_train.describe(include=np.object).T.iloc[:10] # All object cols
    print('\nObject Cols:\n', X_train.describe(include=object).T.iloc[:10]) # All object cols
    above_0_missing = X_train.isnull().sum() > 0
    print('\nNull Values:\n', X_train.isnull().sum()[above_0_missing])
   

def get_num_and_cat_features(X_train):
    '''
    Identify numerical and categorical features

    :input X_train dataframe: Training Data
    :return numerical_features list: List of numerical features
    :return categorical_features list: List of categorical features
    '''
    print('\nCalculating Numerical and Categorical Features...')
    print(f'There are {len(X_train.columns)} total columns.')

    numerical_features = X_train.select_dtypes(include='number').columns.tolist()
    print(f'There are {len(numerical_features)} numerical features.')

    categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
    print(f'There are {len(categorical_features)} categorical features.', '\n')

    return numerical_features, categorical_features

def preprocessing(X_train, y_train):
    '''
    Preprocess Data - numerical and categorical features
    
    :input X_train dataframe: Training Data
    :return full_processor pipeline: Returns pipeline which includes numerical, categorical, and column transformer
    '''

    # Get numerical and categorical feature columns
    numerical_features, categorical_features = get_num_and_cat_features(X_train)

    print('Fetching Preprocessing Pipeline...\n')

    numeric_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    full_processor = ColumnTransformer(transformers=[
        ('number', numeric_pipeline, numerical_features),
        ('category', categorical_pipeline, categorical_features)
    ])

    return full_processor

def lasso_pipeline(X_train, y_train):
    '''
    Create a lasso regression model
    '''
    preprocessor = preprocessing(X_train, y_train)
    lasso = Lasso(alpha=0.1)

    lasso_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', lasso)
    ])

    return lasso_pipeline.fit(X_train, y_train) 

def sk_gb_pipeline(X_train, y_train, n_estimators, lr, max_depth):
    '''
    Create an sklearn gradient boosted regression model
    '''
    preprocessor = preprocessing(X_train, y_train)
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=42, verbose=1)

    sk_gb_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', clf)
    ])

    print('Created sklearn Gradient Boosting Model...\n')
    return sk_gb_pipeline

def xgboost_pipeline(X_train, y_train, do_grid_search, n_estimators, lr, max_depth):
    '''
    Create an XGBoost regression model
    '''
    # Create preprocessor
    preprocessor = preprocessing(X_train, y_train)

    print('Creating XGBoost Model...\n')

    # Run model without params to perform grid search
    if do_grid_search:
        clf = XGBRegressor(seed=42, verbosity=1)

    # Run model with specified params
    else: 
        clf = XGBRegressor(seed=42, verbosity=1, n_estimators = n_estimators, learning_rate = lr, max_depth = max_depth)

    # Create model pipeline including preprocessor
    xgb_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', clf)
    ])

    print('Fitting XGBoost Model...\n')
    return xgb_pipeline

def grid_search(X_train, y_train, model_pipeline):
    '''
    Perform a grid search using the specified model pipeline
    '''
    # Note that 'model__' has to prefix each param
    param_dict = {
        'model__learning_rate': [0.01, 1, 0.1],
        'model__max_depth': range (2, 10, 1),
        'model__n_estimators': range(60, 240, 40)
    }

    print('Starting Grid Search...\n')
    search = GridSearchCV(model_pipeline, param_dict, 
                      cv=10, 
                      scoring='neg_mean_absolute_error',
                      verbose=1)

    search.fit(X_train, y_train)

    print('Best score:', abs(search.best_score_))

    print('Best params:', search.best_params_)

    print('Best estimator:', search.best_estimator_)

    return search.best_params_.values() # returns lr, max_depth, n_estimators


def main(data_source, target_feature):

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = get_data_and_split(data_source, target_feature)

    # Get data insights
    explore_data(X_train)

    # # Run Grid Search and get best params
    lr, max_depth, n_estimators = grid_search(X_train, y_train, xgboost_pipeline(X_train, y_train, True, 0, 0, 0))

    # Create and fit model
    improved_xgboost = xgboost_pipeline(X_train, y_train, False, n_estimators, lr, max_depth).fit(X_train, y_train) 

    # Get predictions and output results
    preds = improved_xgboost.predict(X_valid)
    MAE = mean_absolute_error(y_valid, preds)
    r2 = improved_xgboost.score(X_valid, y_valid)
    print('MAE:', MAE)
    print('R2:', r2)

    # Save the model to disk
    print('Saving Model File\n')
    filename = r'./models/BAT_XGBoost_Model.sav'
    pickle.dump(improved_xgboost, open(filename, 'wb'))
    print(f'Saved Model as {filename}.')
    
if __name__ == '__main__':
    # data_source = input('Enter the filepath to a saved .csv file: ')
    data_source = r"C:\Users\czaozo\OneDrive - BP\Desktop\house-prices-advanced-regression-techniques\train.csv"
    df = pd.read_csv(data_source)
    # raw_data_source = r'{}'.format(data_source)
    # target_feature = input('Enter the target feature name - must match data column: ')
    target_feature = "SalePrice"
    main(data_source, target_feature)

    
    

