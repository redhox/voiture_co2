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


from sklearn.linear_model import LinearRegression

import joblib
import streamlit as st

#recuperation du model
model = joblib.load('./pkl_data/model.pkl')
maping = joblib.load('./pkl_data/maping.pkl')
scaler = joblib.load('./pkl_data/scaler.pkl')


masse_ordma_min_range = [825.0, 2760.0]
masse_ordma_max_range = [825.0, 3094.0]
carrosserie_choices = list(range(11))

# formulaire
masse_ordma_min = st.slider('masse_ordma_min', masse_ordma_min_range[0], masse_ordma_min_range[1], value=masse_ordma_min_range[0])
masse_ordma_max = st.slider('masse_ordma_max', masse_ordma_max_range[0], masse_ordma_max_range[1], value=masse_ordma_max_range[0])

carrosserie = st.selectbox('Choisissez un type de carrosserie:', list(maping.values()))
carrosserie = [k for k, v in maping.items() if v == carrosserie][0]


#IA stuff
df_data=[[carrosserie,masse_ordma_min,masse_ordma_max]]
df_data = scaler.transform(df_data)
predictions = model.predict(df_data)
predictions = model.predict(df_data)

#affichage
st.write('Selected values:')
st.write('masse_ordma_min:', masse_ordma_min)
st.write('masse_ordma_max:', masse_ordma_max)
st.write('Carrosserie:', carrosserie)
st.write('Co2:', predictions)

