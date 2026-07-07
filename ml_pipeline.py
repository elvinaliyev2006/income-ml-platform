import warnings
warnings.filterwarnings('ignore')
import joblib
from data_preprocessor import data_preprocessor
import pandas as pd
import ml_utils
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import classification_report


class ML_Pipeline:

    def __init__(self,df,target_column,best_params, prep_pipeline):
        self.target_column=target_column
        self.df=df
        self.best_params=best_params
        self.prep_pipeline=prep_pipeline
        self.clean_df=self.prepare_data()
        self.X=self.clean_df.drop(columns=self.target_column)
        self.y=self.clean_df[self.target_column]
        self.model=self.build_model()

    def prepare_data(self):
        df=data_preprocessor(self.df)
        return df
        
    def build_model(self):
        estimators=[
                ('KNN',KNeighborsClassifier().set_params(**self.best_params['knn'])),
                ('Logistic Regression',LogisticRegression().set_params(**self.best_params['logreg'])),
                ('LightGBM', LGBMClassifier().set_params(**self.best_params['lgb'])),
                ('XGBoost', XGBClassifier().set_params(**self.best_params['xgb']))]
        
        model=ml_utils.voting_classifier(estimators=estimators,voting='soft',X=self.X,y=self.y,preprocessor_pipeline=self.prep_pipeline)
        return model

    def train_model(self):
        model=self.model
        X_train, _, y_train, _ = train_test_split(self.X,self.y,test_size=0.2,random_state=31,stratify=self.y)
        model.fit(X_train,y_train)
        return model

    def evaluate(self):
        model=self.model
        _, X_test,_ , y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=31,stratify=self.y)
        y_pred=model.predict(X_test)
        print(classification_report(y_test,y_pred))

    def save_model(self,model_path='model.pkl'):
        model=self.model
        model.fit(self.X,self.y)
        joblib.dump(model, model_path)