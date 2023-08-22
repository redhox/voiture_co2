import pandas as pd
import numpy as np
#-------------------------------------------------
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import *
#-------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor, make_column_transformer

#--------------------------------------------------------
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor

#--------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
#------------------------------------------------
from xgboost import XGBRegressor

#-----------------------------------------------------------
from sklearn.model_selection import GridSearchCV
data_model = pd.read_csv("data_model.csv")
df = data_model[['Carrosserie', 'masse_ordma_min', 'masse_ordma_max', 'co2']]
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
objet_columns = df.select_dtypes(include='object').columns
for element in objet_columns:
    df[element]=le.fit_transform(df[element])
X = df.drop(['co2'] ,axis =1)
y = df['co2']

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
# Create the label encoder
le = LabelEncoder()

# Fit the label encoder to the target variable
le.fit(y)

# Transform the target variable
y= le.transform(y)



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold

# Create the classifiers and their respective hyperparameter grids
classifiers = [DummyRegressor(),LinearRegression(),SGDRegressor(),RandomForestRegressor(),GradientBoostingRegressor(),XGBRegressor()]
param_grids = [
    {'strategy': ['mean', 'median'], 'constant': [0, 1]},
    {'fit_intercept': [True, False]},
    {'alpha': [0.0001, 0.001], 'penalty': ['l1', 'l2']},
    {'n_estimators': [50, 100, 200], 'max_depth': [None, 2, 4], 'min_samples_split': [3, 5, 7]},
    {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]},
    {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5], 'n_estimators': [50, 80]}]

# Create the KFold object
#kf = KFold(n_splits=5)

# Create a list to store the results
results = []

# Loop through the classifiers and their hyperparameter grids
for i, (clf, param_grid) in enumerate(zip(classifiers, param_grids)):
    # Create the GridSearchCV object
    grid_search = GridSearchCV(clf, param_grid, scoring='r2', cv=5)
    # Fit the GridSearchCV object to the data
    grid_search.fit(X, y)
    # Store the results
    results.append((type(clf).__name__, grid_search.best_params_, grid_search.best_score_))
    # Print the best parameters and scores
    
# Find the winner
winner = max(results, key=lambda x: x[2])




from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Create the hyperparameter space
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 2, 4],
    'min_samples_split': [3, 5, 7],
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(RandomForestRegressor(), param_grid, scoring='r2', cv=5)

# Fit the RandomizedSearchCV object to the data
random_search.fit(X, y)

df_data=[[10,1505.0,1505.0]]
model=random_search.best_estimator_
predictions = model.predict(df_data)

#donne toujour le meme resulta quand relancé
print('prediction 1: ',predictions)


from sklearn.linear_model import LinearRegression
df_test=[[]]
# Get the best regressor
regressor = winner[0]

# Get the best parameters
parameters = winner[1]

# Convert the parameters to a dictionary
parameters_dict = dict(parameters)

# Create a new regressor with the best parameters
if regressor == 'GradientBoostingRegressor':
    new_regressor = GradientBoostingRegressor(**parameters_dict)
else:
    # Handle other regressors accordingly
    pass

# Fit the new regressor to the data
new_regressor.fit(X, y)

# Get the predictions for new data
predictions = new_regressor.predict(df_data)

#donne des resulta peu diferant quand relancé
print('prediction 2',predictions)