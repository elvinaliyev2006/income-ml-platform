from edatoolkit import Inspector , OutlierAnalyzer , NormalityAnalyzer
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier 
from sklearn.metrics import confusion_matrix,  accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate , learning_curve , validation_curve ,StratifiedKFold
from sklearn.pipeline import Pipeline
import seaborn as sns
import optuna
import shap
import matplotlib.pyplot as plt
import numpy as np
from optuna_config import get_hyperparameters



def optuna_function(X_train,X_test,y_train,y_test ,model_dict,preprocessor,n_trial=20):
    best_results = {}

    for model_name, model_obj in model_dict.items():

        def objective(trial):

            params = get_hyperparameters(model_name, trial)
            copied_model=clone(model_obj)

            current_model = copied_model.set_params(**params)

            model_pipeline = Pipeline(
                steps=[
                    ('preprocessor', preprocessor), 
                    ('model', current_model)
                ]
            )
            model_pipeline.fit(X_train, y_train)
            preds = model_pipeline.predict(X_test)
            
            return f1_score(y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trial) 
        
        best_results[model_name] = {
            "best_params": study.best_params,
            "best_f1_score": study.best_value
            }

    for model_name, res in best_results.items():
        print(f"\n{model_name} The best  F1: {res['best_f1_score']}")
        print(f"Parameters: {res['best_params']}")
    
    return best_results


def confusion_matrix_plot(y,y_pred,height_for_plot=10,width_for_plot=10):
    acc_score=round(accuracy_score(y,y_pred),2)
    cm=confusion_matrix(y,y_pred)
    plt.figure(figsize=(width_for_plot,height_for_plot))
    sns.heatmap(cm,annot=True,fmt='.0f')
    plt.title(f'Accuracy Score: {acc_score}',size=10)
    plt.ylabel('y')
    plt.xlabel('y_pred')
    plt.show()



def voting_classifier(estimators,voting,X,y,preprocessor_pipeline,cv_n=5):
    cv = StratifiedKFold(
    n_splits=cv_n,
    shuffle=True,
    random_state=187
    )
    voting_clsf=VotingClassifier(estimators=estimators, voting=voting)
    model_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor_pipeline), 
            ('model',voting_clsf )  ]
    )
    cv_results=cross_validate(model_pipeline,X,y,cv=cv,scoring=['accuracy','f1','roc_auc','recall','precision'])
    
    return model_pipeline


def roc_curve_plot(model, X_test, y_test, width_for_plot=8, height_for_plot=6):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(width_for_plot,height_for_plot))
    plt.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'Model (AUC = {auc_score:.4f})')
    plt.fill_between(fpr, tpr, alpha=0.10, color='#2196F3')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.50)')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate (Recall)', fontsize=13)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def validation_curve_params(model, X,y,param_name,param_range,scoring='f1',cv_n=5,width_for_graph=11,height_for_graph=6):
    cv = StratifiedKFold(
    n_splits=cv_n,
    shuffle=True,
    random_state=26
    )
    train_scores,validation_scores =validation_curve(model,X=X,y=y,param_name=param_name,
                                                   param_range=param_range,scoring=scoring,cv=cv)
    mean_tr_score=np.mean(train_scores,axis=1)
    train_std  = train_scores.std(axis=1)
    mean_val_score=np.mean(validation_scores,axis=1)
    val_std    = validation_scores.std(axis=1)
    plt.figure(figsize=(width_for_graph,height_for_graph))
    plt.plot(param_range,mean_tr_score,'o-',label='Trainin Score', color='#2196F3')
    plt.fill_between(param_range, mean_tr_score - train_std, mean_tr_score + train_std, alpha=0.15, color='#2196F3')
    plt.plot(param_range,mean_val_score,'s-',label='Validation Score',color='#FF5722')
    plt.fill_between(param_range, mean_val_score- val_std, mean_val_score + val_std, alpha=0.15, color='#FF5722')
    plt.title(f'Validation Curve - {param_name} Effect on Model')
    plt.xlabel(f'Number of {param_name}')
    plt.ylabel(f'{scoring}')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(blok=True)

def learning_curve_plot(model,X,y,scoring='f1',cv_n=5,width_for_graph=11,height_for_graph=6):
    y=y.astype(int)
    cv = StratifiedKFold(
        n_splits=cv_n,
        shuffle=True,
        random_state=42
        )
    train_sizes,train_scores,test_scores = learning_curve(model,X,y,cv=cv,train_sizes=np.linspace(0.1,1.0,10),
                                                          scoring=scoring,n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std  = train_scores.std(axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_std    = test_scores.std(axis=1)
    plt.figure(figsize=(width_for_graph,height_for_graph))
    plt.plot(train_sizes, train_mean,'o-', label="Training Score",color='#2196F3')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2196F3')
    plt.plot(train_sizes, test_mean,label="Cross-Validation Score", color='#FF5722')
    plt.fill_between(train_sizes, test_mean- val_std, test_mean + val_std, alpha=0.15, color='#FF5722')
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel(f"{scoring}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def shap_bar_plot(model, X,max_display=None, width_for_plot=8, height_for_plot=5):
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[..., 1]
    plt.figure(figsize=(width_for_plot, height_for_plot))
    shap.plots.bar(shap_values,  max_display=max_display,show=False)
    plt.title("Average SHAP Values ", fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


def shap_beeswarm_plot(model, X, max_display=None, width_for_plot=8, height_for_plot=5):
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[..., 1]
    plt.figure(figsize=(width_for_plot, height_for_plot))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm ", fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


def shap_waterfall_plot(model, X, index=0, width_for_plot=8, height_for_plot=5):
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[..., 1]
    plt.figure(figsize=(width_for_plot, height_for_plot))
    shap.plots.waterfall(shap_values[index],max_display=100, show=False)
    plt.title(f"SHAP Waterfall", fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()