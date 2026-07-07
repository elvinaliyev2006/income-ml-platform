import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline import ML_Pipeline
from database.database import DataBase_Manager
import pandas as pd
from optuna_config import best_params
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler , OneHotEncoder , OrdinalEncoder

education_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th','HS-grad', 'Some-college', 
                   'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']

encoding_stage = ColumnTransformer(transformers=[('cat', OneHotEncoder(sparse_output=False), ['workclass', 'marital.status', 'race', 'occupation', 
                                                                                              'relationship', 'sex', 'native.country']),
                                                ('car_ord', OrdinalEncoder(categories=[education_order]), ['education'])],remainder='passthrough' )

preprocessing_pipeline = Pipeline(steps=[('encoding', encoding_stage),('scaling', RobustScaler())])


dbm=DataBase_Manager()
df=dbm.load_table('census_income')

ml_pipeline=ML_Pipeline(df,'income',best_params=best_params,prep_pipeline=preprocessing_pipeline)
ml_pipeline.train_model()
ml_pipeline.evaluate()
ml_pipeline.save_model()