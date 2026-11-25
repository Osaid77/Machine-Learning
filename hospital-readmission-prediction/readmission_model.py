import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer       # preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import brier_score_loss

from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from sklearn.pipeline import Pipeline              # Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold, cross_validate         # CV

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV        # HT

from sklearn.inspection import permutation_importance

import joblib





patients = pd.read_csv("Scikit-Learn/diabetic_data.csv")

print(patients)
print(patients.shape)
print(patients.columns)
print(patients.isna().sum())
print(patients["readmitted"].value_counts())

patients = patients.drop_duplicates(subset=["patient_nbr"], keep="first")

numerical_features = ["time_in_hospital", "num_lab_procedures", "num_medications", "number_inpatient"]
categorical_features = ["gender", "A1Cresult", "diabetesMed"]

X = patients[numerical_features + categorical_features]
y = np.where(patients["readmitted"] == "<30", 1, 0)

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.15/0.85, random_state=42, stratify=y_dev)

print(y_train.mean())     # 0.0879 = Class imbalance exists 8.79% for positive class
print(y_val.mean())       # 0.0879 = Class imbalance exists 8.79% for positive class


numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])



models = {
    "dummy": DummyClassifier(strategy="most_frequent"),

    "logr": LogisticRegression(max_iter=1000, random_state=42), 
    "logr_b": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),

    "dt": DecisionTreeClassifier(random_state=42), 
    "dt_b": DecisionTreeClassifier(random_state=42, class_weight="balanced"),

    "rf": RandomForestClassifier(random_state=42, n_jobs=-1), 
    "rf_b": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),

    "hgb": HistGradientBoostingClassifier(random_state=42), 
    "hgb_b": HistGradientBoostingClassifier(random_state=42, class_weight="balanced"), 
}


param_grids = {
    "logr": {
        "model__C": [0.1, 0.5, 1.0]
    },
    "logr_b": {
        "model__C": [0.1, 0.5, 1.0]
    },
    "dt": {
        "model__max_depth": [5, 10, 15],
        "model__min_samples_leaf": [10, 15, 20]
    },
    "dt_b": {
        "model__max_depth": [5, 10, 15],
        "model__min_samples_leaf": [10, 15, 20]
    },
    "rf": {
        "model__max_depth": [5, 10, 15],
        "model__min_samples_leaf": [10, 15, 20, 30],
        "model__n_estimators": [100, 150, 200]
    },
    "rf_b": {
        "model__max_depth": [5, 10, 15],
        "model__min_samples_leaf": [10, 15, 20, 30],
        "model__n_estimators": [100, 150, 200]
    },
    "hgb": {
        "model__max_leaf_nodes": [5, 10, 15, 20],
        "model__max_iter": [100, 150, 200],
        "model__learning_rate": [0.03, 0.05, 0.1]
    },
    "hgb_b": {
        "model__max_leaf_nodes": [10, 15, 20, 30],
        "model__max_iter": [100, 150, 200],
        "model__learning_rate": [0.03, 0.05, 0.1]
    },
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "ap": "average_precision"
}

cv = StratifiedKFold(shuffle=True, random_state=42, n_splits=5)

results = []

calibrated_pipes = {}

fitted_pipes = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    if name == "dummy":
        pipe.fit(X_train, y_train)
        best_score = None
        best_param = {}

    else:
        if name in ["logr", "logr_b", "dt", "dt_b"]:
            search = GridSearchCV(estimator=pipe, scoring=scoring, param_grid=param_grids[name], n_jobs=-1, refit="f1", cv=cv)

        elif name in ["rf", "rf_b", "hgb", "hgb_b"]:
            search = RandomizedSearchCV(estimator=pipe, param_distributions=param_grids[name], n_iter=10, scoring=scoring, n_jobs=1, refit="f1", cv=cv, random_state=42)

        search.fit(X_train, y_train)
        best_score = search.best_score_
        best_param = search.best_params_
        pipe = search.best_estimator_       


    if name in ["logr", "logr_b", "dt", "dt_b", "rf", "rf_b", "hgb", "hgb_b"]:
        fitted_pipes[name] = pipe

    
    pipe_proba = pipe.predict_proba(X_val)[:, 1]


    # Calibration:
    pipe_c = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=3, n_jobs=-1)
    pipe_c.fit(X_train, y_train)
    pipe_c_proba = pipe_c.predict_proba(X_val)[:, 1]

    calibrated_pipes[name] = pipe_c



    thresholds = [0.10, 0.30, 0.50]

    for threshold in thresholds:

        # Original
        pipe_pred = np.where(pipe_proba >= threshold, 1, 0)

        cm = confusion_matrix(y_val, pipe_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_val, pipe_pred)

        precision = precision_score(y_val, pipe_pred, zero_division=0)
        recall = recall_score(y_val, pipe_pred, zero_division=0)
        f1 = f1_score(y_val, pipe_pred, zero_division=0)

        specificity = tn / (tn + fp)

        roc_auc = roc_auc_score(y_val, pipe_proba)
        ap = average_precision_score(y_val, pipe_proba)

        y_true, y_pred = calibration_curve(y_val, pipe_proba, strategy="quantile", n_bins=5)
        brier = brier_score_loss(y_val, pipe_proba)


        # Calibrated
        pipe_c_pred = np.where(pipe_c_proba >= threshold, 1, 0)

        cm_c = confusion_matrix(y_val, pipe_c_pred, labels=[0, 1])
        tn_c, fp_c, fn_c, tp_c = cm_c.ravel()

        accuracy_c = accuracy_score(y_val, pipe_c_pred)

        precision_c = precision_score(y_val, pipe_c_pred, zero_division=0)
        recall_c = recall_score(y_val, pipe_c_pred, zero_division=0)
        f1_c = f1_score(y_val, pipe_c_pred, zero_division=0)

        specificity_c = tn_c / (tn_c + fp_c)

        roc_auc_c = roc_auc_score(y_val, pipe_c_proba)
        ap_c = average_precision_score(y_val, pipe_c_proba)

        y_true_c, y_pred_c = calibration_curve(y_val, pipe_c_proba, strategy="quantile", n_bins=5)
        brier_c = brier_score_loss(y_val, pipe_c_proba)

        results.append({
            "model": name,
            "thre": threshold,

            # Original
            "acc": accuracy,
            "prc": precision,
            "rec": recall,
            "f1": f1,
            "spc": specificity,
            "roc": roc_auc,
            "ap": ap,
            "y_true": y_true.round(2),
            "y_pred": y_pred.round(2),
            "brier": brier,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,


            # Calibrated
            "acc_c": accuracy_c,
            "prc_c": precision_c,
            "rec_c": recall_c,
            "f1_c": f1_c,
            "spc_c": specificity_c,
            "roc_c": roc_auc_c,
            "ap_c": ap_c,
            "y_true_c": y_true_c.round(2),
            "y_pred_c": y_pred_c.round(2),
            "brier_c": brier_c,
            "TP_c": tp_c,
            "FP_c": fp_c,
            "TN_c": tn_c,
            "FN_c": fn_c,

            "best_score": best_score,
            "best_param": best_param
        })

results_df = pd.DataFrame(results)


joblib.dump(fitted_pipes, "fitted_pipes.joblib")
joblib.dump(calibrated_pipes, "calibrated_pipes.joblib")

results_df.to_csv("model_results.csv", index=False)

pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", None)


print(results_df.round(3))





# Features interpretation:


logr_pipe = fitted_pipes["logr"]
logr_b_pipe = fitted_pipes["logr_b"]

features = logr_pipe.named_steps["preprocessor"].get_feature_names_out()

logr_coef = logr_pipe.named_steps["model"].coef_[0]
logr_b_coef = logr_b_pipe.named_steps["model"].coef_[0]

logr_coef_df = pd.DataFrame({
    "features": features,
    "logr_coef": logr_coef,
    "logr_b_coef": logr_b_coef

}).sort_values("logr_coef", ascending=False)

print(logr_coef_df.round(3))


#####


dt_pipe = fitted_pipes["dt"]
dt_b_pipe = fitted_pipes["dt_b"]

rf_pipe = fitted_pipes["rf"]
rf_b_pipe = fitted_pipes["rf_b"]

features = dt_pipe.named_steps["preprocessor"].get_feature_names_out()

dt_imp = dt_pipe.named_steps["model"].feature_importances_
dt_b_imp = dt_b_pipe.named_steps["model"].feature_importances_

rf_imp = rf_pipe.named_steps["model"].feature_importances_
rf_b_imp = rf_b_pipe.named_steps["model"].feature_importances_

tree_feature_importances = pd.DataFrame({
    "features": features,

    "dt_imp": dt_imp,
    "dt_b_imp": dt_b_imp,

    "rf_imp": rf_imp,
    "rf_b_imp": rf_b_imp

}).sort_values("dt_imp", ascending=False)

print(tree_feature_importances.round(3))


######


logr_pipe = fitted_pipes["logr"]
logr_b_pipe = fitted_pipes["logr_b"]

dt_pipe = fitted_pipes["dt"]
dt_b_pipe = fitted_pipes["dt_b"]

rf_pipe = fitted_pipes["rf"]
rf_b_pipe = fitted_pipes["rf_b"]

hgb_pipe = fitted_pipes["hgb"]
hgb_b_pipe = fitted_pipes["hgb_b"]

features = X_val.columns

logr_imp = permutation_importance(logr_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=1, random_state=42)
logr_b_imp = permutation_importance(logr_b_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=1, random_state=42)

dt_imp = permutation_importance(dt_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=1, random_state=42)
dt_b_imp = permutation_importance(dt_b_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=1, random_state=42)

rf_imp = permutation_importance(rf_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=1, random_state=42)
rf_b_imp = permutation_importance(rf_b_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=1, random_state=42)

hgb_imp = permutation_importance(hgb_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=1, random_state=42)
hgb_b_imp = permutation_importance(hgb_b_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=1, random_state=42)

perm_imp_df = pd.DataFrame({
    "features": features,

    "logr_imp_m": logr_imp.importances_mean,
    "logr_imp_std": logr_imp.importances_std,

    "logr_b_imp_m": logr_b_imp.importances_mean,
    "logr_b_imp_std": logr_b_imp.importances_std,

    "dt_imp_m": dt_imp.importances_mean,
    "dt_imp_std": dt_imp.importances_std,

    "dt_b_imp_m": dt_b_imp.importances_mean,
    "dt_b_imp_std": dt_b_imp.importances_std,

    "rf_imp_m": rf_imp.importances_mean,
    "rf_imp_std": rf_imp.importances_std,

    "rf_b_imp_m": rf_b_imp.importances_mean,
    "rf_b_imp_std": rf_b_imp.importances_std,

    "hgb_imp_m": hgb_imp.importances_mean,
    "hgb_imp_std": hgb_imp.importances_std,

    "hgb_b_imp_m": hgb_b_imp.importances_mean,
    "hgb_b_imp_std": hgb_b_imp.importances_std

}).sort_values("logr_imp_m", ascending=False)
    
print(perm_imp_df.round(3))



#### Testing 


test_pipe = fitted_pipes["hgb_b"]

# pipe_proba = pipe.predict_proba(X_val)[:, 1]

test_proba = test_pipe.predict_proba(X_test)[:, 1]

threshold = 0.50

test_pred = np.where(test_proba >= threshold, 1, 0)

cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, test_pred)

precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)

specificity = tn / (tn + fp)

roc_auc = roc_auc_score(y_test, test_proba)
ap = average_precision_score(y_test, test_proba)

y_true, y_pred = calibration_curve(y_test, test_proba, n_bins=5, strategy="quantile")
brier = brier_score_loss(y_test, test_proba)

results = []

results.append({
    "model": "hgb_b",
    "thre": threshold,

    "acc": accuracy,
    "prc": precision,
    "rec": recall,
    "f1": f1,
    "spc": specificity,
    "roc": roc_auc,
    "ap": ap,
    "y_true": y_true.round(2),
    "y_pred": y_pred.round(2),
    "brier": brier,
    "TP": tp,
    "FP": fp,
    "TN": tn,
    "FN": fn
})

results_df = pd.DataFrame(results)

print(results_df.round(3))

results_df.to_csv("test_results.csv", index=False)

    
    

