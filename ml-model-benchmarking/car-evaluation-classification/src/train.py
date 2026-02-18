from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


from src.preprocess import split_feature_target,encode_and_split

def train_logistics(x_train,y_train):
    model=LogisticRegression()
    model.fit(x_train,y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model 

def tune_logistic(X_train, y_train):

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=2000),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_score_

def tune_svm(X_train, y_train):

    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_score_

def tune_decision_tree(X_train, y_train):

    param_grid = {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_score_

def tune_random_forest(X_train, y_train):

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_score_

    
