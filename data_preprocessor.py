from edatoolkit import Inspector, NormalityAnalyzer, OutlierAnalyzer
import pandas as pd
from lightgbm import LGBMClassifier
import numpy as np


def imputer(df, col_name, cat_cols, num_cols):
    all_features = col_name + cat_cols + num_cols
    
    df_2 = df[all_features].copy()
    for col in cat_cols:
        df_2[col] = df_2[col].astype('category')
        
    train_df = df_2[~df_2.isnull().any(axis=1)]

    test_df = df_2[df_2.isnull().any(axis=1)]
    indexes=test_df.index
    

    X_train = train_df.drop(columns=col_name[0])
    X_test = test_df.drop(columns=col_name[0])
    y_train = train_df[col_name[0]] 
    model = LGBMClassifier(verbosity=-1, force_row_wise=True)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    ser=pd.Series(pred)
    ser.name=col_name[0]
    pred=pd.DataFrame(ser).set_index(indexes)
    df[col_name[0]] = df[col_name[0]].fillna(pred[col_name[0]])
    
    return df


def data_preprocessor(dataframe):
    cat_cols,num_cols,_,cat_but_car=Inspector().get_columns_types(dataframe=dataframe,car_th=20,cat_th=10)
    dataframe=dataframe[~((dataframe['relationship']=='Wife') & (dataframe['sex']=='Male')) & ~((dataframe['relationship'] == 'Husband') &
                                                                                                 (dataframe['sex']=='Female'))]
    dataframe = dataframe[~((dataframe['marital.status'] == 'Married-civ-spouse') & (dataframe['relationship'] == 'Other-relative')) 
        & ~((dataframe['marital.status'] == 'Married-civ-spouse') & (dataframe['relationship'] == 'Not-in-family'))]
    dataframe['workclass'] = dataframe['workclass'].replace( ['Never-worked', 'Without-pay'], 'Other')
    dataframe['occupation'] = dataframe['occupation'].replace('Armed-Forces',  'Protective-serv')
    dataframe['marital.status'] = dataframe['marital.status'].replace('Married-AF-spouse', 'Married-civ-spouse')
    dataframe = dataframe.replace("?", np.nan)
    dataframe=imputer(dataframe,['native.country'],['education','marital.status','relationship','race','sex','income']
                      ,num_cols)
    dataframe=imputer(dataframe,['workclass'],['native.country','education','marital.status','relationship','race','sex','income']
                      ,num_cols)
    dataframe=imputer(dataframe,['occupation'],['workclass','native.country','education','marital.status','relationship','race','sex'
                                                ,'income']
                      ,num_cols)
    dict_={}
    for i in range(len(num_cols)):
        dict_[num_cols[i]]='Non-normal'
    nsd=NormalityAnalyzer().num_summary(num_cols=num_cols,result_dict=dict_)
    dataframe=dataframe.drop(columns=['education.num','fnlwgt'])
    num_cols = [col for col in num_cols if col not in ['education.num', 'fnlwgt']]
    _,dataframe=OutlierAnalyzer().check_outlier(dataframe=dataframe,num_cols=num_cols,
                                                   iqr_th=1.5,q1_th=0.04,q3_th=0.96, num_summary_df=nsd, cap=True)
    dataframe['income']=dataframe['income'].replace('<=50K', 0)
    dataframe['income'] =dataframe['income'].replace('>50K', 1)
    dataframe['income']=dataframe['income'].astype(int)
    natco_count=(dataframe['native.country'].value_counts().reset_index())
    smaller_than_30=natco_count[natco_count['count']<=30]['native.country'].tolist()
    dataframe['native.country']=dataframe['native.country'].replace(smaller_than_30,'other')
    dataframe[cat_cols+cat_but_car]=dataframe[cat_cols+cat_but_car].astype('category')
    dataframe=dataframe.dropna()
    return dataframe


