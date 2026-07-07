def get_hyperparameters(name, trial):
    if name == 'knn':
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
            "leaf_size": trial.suggest_int("leaf_size", 10, 60),     
            "algorithm": trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"]),
            "p": trial.suggest_int("p", 1, 4)
        }
        
    elif name == 'logreg':
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        
        if solver == "liblinear":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        else:
            penalty = "l2"  
            
        return {
            "solver": solver,
            "penalty": penalty,
            "C": trial.suggest_float("C", 0.001, 100, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "random_state": 195
        }

    elif name == 'rf':
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=65),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 19,
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.1),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "bootstrap": True
        }
    
    elif name == 'xgb':
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=65),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "eval_metric": "logloss",
            "random_state": 19,
            "verbosity": 0,
             "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True), 
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),         
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0)
        }

    elif name == 'cat':
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=65),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10), 
            "verbose": 0,
            "random_seed": 20,
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0)
        }
    
    elif name == 'lgb':
        max_depth=trial.suggest_int('max_depth',3,20)
        max_leaves=min(2**max_depth-1,170)
        return {
            'boosting_type':'gbdt',
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=65),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", min(15, max_leaves),max(15, max_leaves)),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "verbosity": -1,
            "random_state": 37,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),  
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "max_bin": trial.suggest_int("max_bin", 64, 512),           
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    

best_params={
    'knn':{'n_neighbors': 23, 'weights': 'uniform', 'metric': 'manhattan', 'leaf_size': 53, 'algorithm': 'brute', 'p': 2},
             

'logreg':{'solver': 'liblinear', 'penalty': 'l2', 'C': 18.723180530898663, 'max_iter': 989},


'xgb':{'n_estimators': 880, 'learning_rate': 0.09878575027040652, 'max_depth': 19, 'subsample': 0.8705446114501136,
        'colsample_bytree': 0.47599822092149624, 'reg_alpha': 0.00100172701389466, 'reg_lambda': 0.0003585072513039218, 'min_child_weight': 3,
          'gamma': 1.6741977594753572, 'colsample_bylevel': 0.4101694058374038, 'scale_pos_weight': 2.038549031110637},


'lgb':{'max_depth': 5, 'n_estimators': 165, 'learning_rate': 0.10256462973804831, 'num_leaves': 21, 'min_child_samples': 20, 'subsample': 0.7368675112578026,
        'colsample_bytree': 0.6214719757409347, 'reg_alpha': 0.0002579570120840671, 'reg_lambda': 0.03789330254153911, 
        'min_split_gain': 0.21082767704788877, 'max_bin': 228, 'subsample_freq': 3, 'class_weight': 'balanced'}
        }