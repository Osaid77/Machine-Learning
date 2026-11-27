# Predict in-hospital mortality using clinical information collected within first 24 hours of hospitalization


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss

from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from sklearn.inspection import permutation_importance

import joblib




patients = pd.read_csv("Scikit-Learn/training_v2.csv")

print(patients)                 # Dataset inspection
print(patients.shape)
print(patients.columns)
print(patients.dtypes)
print(patients.nunique())
print(patients["hospital_death"].value_counts())         # made sure the target is binary

print()
print(patients.isna().mean() * 100)
print()
print("_____________")
# print(patients["apache_3j_diagnosis"].value_counts())       # I made sure of the value counts of each feature in the dataset 186 feature especially the 113 that we'll use


# print(patients["encounter_id"].is_unique)      # Both of them are unique so we won't drop_duplicates here, I'm thinking that each patient is a unique patient. No patient with multiple visits
# print(patients["patient_id"].is_unique)

# Replace -1 values for probabilities with np.nan  -> np.nan will be replaced with median just like other numerical features within numerical_pipeline
patients.loc[patients["apache_4a_hospital_death_prob"] == -1, "apache_4a_hospital_death_prob"] = np.nan      
patients.loc[patients["apache_4a_icu_death_prob"] == -1, "apache_4a_icu_death_prob"] = np.nan

# Examining after -1 probability removal:
# print(patients[["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"]].skew())     # before filling nan with median in numerical_preprocessing is: 2.5 - 3.4
# print(patients[["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"]].describe())
# print(patients[["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"]].isna().mean() * 100)    both missing 11% and they will be filled with the median









# Missing a lot but I added them due to their clinical relevance:
# bilirubin_apache 63% missing, d1_bilirubin_max missing 58% but I put it
# h1_glucose_max, h1_glucose_min: both missing 57% but still include them
# h1_inr_max, h1_inr_min: both missing 63% but I still include them because maybe they're not indicated to investigate in 63% of patients 
# d1_arterial_pco2_max, d1_arterial_pco2_min, d1_arterial_ph_max, d1_arterial_ph_min, d1_arterial_po2_max, d1_arterial_po2_min: each is missing 64% but they're relevant, I will put them
# h1_arterial_pco2_max, h1_arterial_pco2_min, h1_arterial_ph_max, h1_arterial_ph_min, h1_arterial_po2_max, h1_arterial_po2_min: Each is missing 82, I'm confused whether to add them or not but I'm leaning way more towards not adding them due to missing a lot. NOT included
# paco2_apache 77% missing, pao2_apache 77% missing, 
# d1_inr_min both missing in 63% of records but I still put them because maybe they're not indicated to investigate in 63% of patients 
# d1_lactate_max 74% missing


# Missing a lot even though they're clinically relevant but I cannot put them due to missing a lot and having better substitutes included: 
# d1_diasbp_invasive_max 74% missing, d1_diasbp_invasive_min 74% missing, 
# h1_diasbp_invasive_max 81% missing, h1_diasbp_invasive_min 81% missing, 
# ph_apache 77% missing,
# d1_mbp_invasive_max 73% missing, d1_mbp_invasive_min 73% missing, 
# h1_mbp_invasive_max 81% missing, h1_mbp_invasive_min 81% missing, 
# d1_sysbp_invasive_max 74% missing, d1_sysbp_invasive_min 74% missing,
# h1_sysbp_invasive_max 81% missing, h1_sysbp_invasive_min 81% missing,
# h1_lactate_max missing 91%, 
# h1_albumin_min missing 91%, 
# h1_calcium_max 82% missing, h1_calcium_min 82% missing,
# h1_bilirubin_max 92% missing, 
# h1_bun_max 81%, 
# h1_creatinine_max missing 81%, 
# h1_hco3_max 82% missing, h1_hco3_min missing 82%, 
# h1_hemaglobin_min missing 79%, h1_hematocrit_min missing 80%,
# h1_platelets_min 82% missing, 
# h1_wbc_min 82% is missing, 
# h1_potassium_max, h1_potassium_min, 
# h1_sodium_max, h1_sodium_min: each of these 4 is missing 78%



# Missing a lot with no substitute:
# fio2_apache 77% missing, 




# 109 total features based on most important clinical relevance and as low missing values as possible
categorical_features = ["gender", "hospital_admit_source", "icu_admit_source", "apache_3j_bodysystem", "apache_2_bodysystem"]

numerical_features = ["age", "bmi", "pre_icu_los_days", "bun_apache", "creatinine_apache", "glucose_apache", "heart_rate_apache", "hematocrit_apache", 
    "map_apache", "resprate_apache", "sodium_apache", "temp_apache", "urineoutput_apache", "wbc_apache", "d1_diasbp_max", "d1_diasbp_min", 
    "d1_diasbp_noninvasive_min", "d1_heartrate_max", "d1_heartrate_min", "d1_mbp_max", "d1_mbp_min", "d1_mbp_noninvasive_min", "d1_resprate_max", "d1_resprate_min", "d1_spo2_min", "d1_sysbp_max", "d1_sysbp_min", 
    "d1_sysbp_noninvasive_max", "d1_sysbp_noninvasive_min", "d1_temp_max", "d1_temp_min", "h1_diasbp_max", "h1_diasbp_min", "h1_diasbp_noninvasive_max", 
    "h1_diasbp_noninvasive_min", "h1_heartrate_max", "h1_heartrate_min", "h1_mbp_max", "h1_mbp_min", "h1_mbp_noninvasive_max", "h1_mbp_noninvasive_min", 
    "h1_resprate_max", "h1_resprate_min", "h1_spo2_max", "h1_spo2_min", "h1_sysbp_max", "h1_sysbp_min", "h1_sysbp_noninvasive_max", "h1_sysbp_noninvasive_min", 
    "h1_temp_max", "h1_temp_min", "d1_albumin_min", "d1_bun_max", "d1_calcium_max", "d1_calcium_min", "d1_creatinine_max", "d1_creatinine_min", 
    "d1_glucose_max", "d1_glucose_min", "d1_hco3_max", "d1_hco3_min", "d1_hemaglobin_min", "d1_hematocrit_min", "d1_platelets_min", "d1_potassium_max", 
    "d1_potassium_min", "d1_sodium_max", "d1_sodium_min", "d1_wbc_min", "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob", "bilirubin_apache", "d1_bilirubin_max",
    "h1_glucose_max", "h1_glucose_min", "d1_arterial_pco2_max", "d1_arterial_pco2_min", "d1_arterial_ph_max", "d1_arterial_ph_min", "d1_arterial_po2_max", 
    "d1_arterial_po2_min", "paco2_apache", "pao2_apache", "d1_lactate_max", "d1_inr_max", "d1_inr_min"]   

binary_features = ["elective_surgery", "ventilated_apache", "arf_apache", "apache_post_operative", "gcs_unable_apache", "aids", "cirrhosis", 
                   "diabetes_mellitus", "hepatic_failure", "immunosuppression", "leukemia", "lymphoma", "solid_tumor_with_metastasis", "intubated_apache"]
 
high_cardinality_features = ["apache_3j_diagnosis"]

ordinal_features = ["gcs_eyes_apache", "gcs_motor_apache", "gcs_verbal_apache"]







X = patients[categorical_features + numerical_features + binary_features + high_cardinality_features + ordinal_features]
y = patients["hospital_death"] == 1    # Both solutions work
y = np.where(patients["hospital_death"] == 1, 1, 0)


X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.15/0.85, random_state=42, stratify=y_dev)

print(y_train.mean())          # positive class represents 8.63%, that's class imbalance and we need to balance using class_weight="balanced" 
print(y_val.mean())


numerical_summary = pd.DataFrame({
    "mean": X_train[numerical_features].mean(),
    "median": X_train[numerical_features].median(),
    "skewness": X_train[numerical_features].skew(),
    "missing_%": X_train[numerical_features].isna().mean() * 100
}).sort_values("skewness", ascending=False)
# print(numerical_summary.round(3))         # I investigated correlation between features I suspect they overlap/correlate as seen below and removed some features that showed high correlation like >= 0.90

# print(X_train["d1_inr_max"].equals(X_train["h1_inr_max"]))        # True     So I removed h1 variants in both
# print(X_train["d1_inr_min"].equals(X_train["h1_inr_min"]))        # True

# print(X_train["d1_inr_max"].corr(X_train["h1_inr_max"]))     1.0            Examined correlation
# print(X_train["d1_inr_min"].corr(X_train["h1_inr_min"]))     0.99   

# print(X_train["h1_glucose_min"].corr(X_train["h1_glucose_max"]))      0.96           Kept both features even though with high correlation but each represents diff value max/min

# print(X_train["d1_diasbp_max"].corr(X_train["d1_diasbp_noninvasive_max"]))    # 0.99 corr score        They measure bp in diff ways, invasive and non-invasive. However, these aren't two exclusively different measurement methods.
# print(X_train["d1_mbp_max"].corr(X_train["d1_mbp_noninvasive_max"]))       # 0.98 corr score           I kept the general cuz they contain invasive and non-invasive


# print(X_train["ventilated_apache"].corr(X_train["intubated_apache"]))    # 0.61 corr score       Kept them both cuz weak redundancy between them and different meaningful meaning for each one


# pre_icu_los_days is missing 0% so no diff in imputation

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])
high_cardinality_pipeline = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", min_frequency=100))     # rare-category grouping. All categories occurring < 100 times during fitting are combined into one encoded category.
])
binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True))
])
ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),            # preserves ordering and handles missingness
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("high_cardinality", high_cardinality_pipeline, high_cardinality_features),
    ("binary", binary_pipeline, binary_features),
    ("ordinal", ordinal_pipeline, ordinal_features)
])

models = {
    "dummy": DummyClassifier(strategy="most_frequent", random_state=42),

    "logr": LogisticRegression(max_iter=1000, random_state=42), 
    "logr_b": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"), 

    "dt": DecisionTreeClassifier(random_state=42), 
    "dt_b": DecisionTreeClassifier(random_state=42, class_weight="balanced"), 

    "rf": RandomForestClassifier(n_jobs=-1, random_state=42), 
    "rf_b": RandomForestClassifier(n_jobs=-1, random_state=42, class_weight="balanced"), 

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
        "model__n_estimators": [100, 150, 200],
        "model__max_depth": [5, 10, 15],
        "model__min_samples_leaf": [10, 15, 20, 30]
    },
    "rf_b": {
        "model__n_estimators": [100, 150, 200],
        "model__max_depth": [5, 10, 15],
        "model__min_samples_leaf": [10, 15, 20, 30]
    },
    "hgb": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.5],
        "model__max_iter": [100, 150, 200],
        "model__max_leaf_nodes": [10, 15, 20, 30]
    },
    "hgb_b": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.5],
        "model__max_iter": [100, 150, 200],
        "model__max_leaf_nodes": [10, 15, 20, 30]
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

fitted_pipes = {}
calibrated_pipes = {}

for name, model in models.items():
    print(f"Training {name}...", flush=True)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    if name == "dummy":
        pipe.fit(X_train, y_train)
        best_score = None
        best_params = {}

    
    else: 
        if name in ["logr", "logr_b", "dt", "dt_b"]:
            search = GridSearchCV(pipe, param_grid=param_grids[name], scoring=scoring, n_jobs=-1, refit="ap", cv=cv, verbose=2)

        elif name in ["rf", "rf_b", "hgb", "hgb_b"]:
            search = RandomizedSearchCV(pipe, param_distributions=param_grids[name], n_iter=10, scoring=scoring, cv=cv, refit="ap", n_jobs=-1, random_state=42, verbose=2)

        search.fit(X_train, y_train)

        best_score = search.best_score_
        best_params = search.best_params_
        pipe = search.best_estimator_

    pipe_proba = pipe.predict_proba(X_val)[:, 1]


    if name in ["logr", "logr_b", "dt", "dt_b", "rf", "rf_b", "hgb", "hgb_b"]:
        fitted_pipes[name] = pipe




    pipe_c = CalibratedClassifierCV(pipe, method="sigmoid", n_jobs=-1, cv=3)

    pipe_c.fit(X_train, y_train)
    pipe_c_proba = pipe_c.predict_proba(X_val)[:, 1]

    calibrated_pipes[name] = pipe_c


    thresholds = [0.10, 0.20, 0.30, 0.50]      # I added 0.20 threshold, I wish it's good idea to test more thresholds

    for threshold in thresholds:

        # original
        pipe_predict = np.where(pipe_proba >= threshold, 1, 0)

        cm = confusion_matrix(y_val, pipe_predict, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_val, pipe_predict)

        precision = precision_score(y_val, pipe_predict, zero_division=0)
        recall = recall_score(y_val, pipe_predict, zero_division=0)
        f1 = f1_score(y_val, pipe_predict, zero_division=0)

        specificity = tn / (tn + fp)

        roc_auc = roc_auc_score(y_val, pipe_proba)
        ap = average_precision_score(y_val, pipe_proba)

        y_true, y_pred = calibration_curve(y_val, pipe_proba, n_bins=5, strategy="quantile")
        brier = brier_score_loss(y_val, pipe_proba)


        # Calibrated
        pipe_c_predict = np.where(pipe_c_proba >= threshold, 1, 0)

        cm_c = confusion_matrix(y_val, pipe_c_predict, labels=[0, 1])
        tn_c, fp_c, fn_c, tp_c = cm_c.ravel()

        accuracy_c = accuracy_score(y_val, pipe_c_predict)

        precision_c = precision_score(y_val, pipe_c_predict, zero_division=0)
        recall_c = recall_score(y_val, pipe_c_predict, zero_division=0)
        f1_c = f1_score(y_val, pipe_c_predict, zero_division=0)

        specificity_c = tn_c / (tn_c + fp_c)

        roc_auc_c = roc_auc_score(y_val, pipe_c_proba)
        ap_c = average_precision_score(y_val, pipe_c_proba)

        y_true_c, y_pred_c = calibration_curve(y_val, pipe_c_proba, n_bins=5, strategy="quantile")
        brier_c = brier_score_loss(y_val, pipe_c_proba)


        results.append({
            "model": name,
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
            "FN": fn,


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
            "best_params": best_params
        })

results_df = pd.DataFrame(results)

joblib.dump(fitted_pipes, "fitted_pipes.joblib")
joblib.dump(calibrated_pipes, "calibrated_pipes.joblib")

results_df.to_csv("model_results.csv", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)


print(results_df.round(3))


###########

# logr feature coefficients

logr_pipe = fitted_pipes["logr"]
logr_b_pipe = fitted_pipes["logr_b"]

features = logr_pipe.named_steps["preprocessor"].get_feature_names_out()

logr_coeff = logr_pipe.named_steps["model"].coef_[0]
logr_b_coeff = logr_b_pipe.named_steps["model"].coef_[0]

feature_coef = pd.DataFrame({
    "features": features,
    "logr_coeff": logr_coeff,
    "logr_b_coeff": logr_b_coeff
}).sort_values("logr_coeff", ascending=False)

print(feature_coef.round(3))


###########

# trees feature importance

dt_pipe = fitted_pipes["dt"]
dt_b_pipe = fitted_pipes["dt_b"]
rf_pipe = fitted_pipes["rf"]
rf_b_pipe = fitted_pipes["rf_b"]

features = dt_pipe.named_steps["preprocessor"].get_feature_names_out()

dt_imp = dt_pipe.named_steps["model"].feature_importances_
dt_b_imp = dt_b_pipe.named_steps["model"].feature_importances_

rf_imp = rf_pipe.named_steps["model"].feature_importances_
rf_b_imp = rf_b_pipe.named_steps["model"].feature_importances_

tree_imp_df = pd.DataFrame({
    "features": features,

    "dt_imp": dt_imp,
    "dt_b_imp": dt_b_imp,
    "rf_imp": rf_imp,
    "rf_b_imp": rf_b_imp

}).sort_values("dt_imp", ascending=False)

print(tree_imp_df.round(3))


###########

# permutation importance

logr_pipe = fitted_pipes["logr"]
logr_b_pipe = fitted_pipes["logr_b"]
dt_pipe = fitted_pipes["dt"]
dt_b_pipe = fitted_pipes["dt_b"]
rf_pipe = fitted_pipes["rf"]
rf_b_pipe = fitted_pipes["rf_b"]
hgb_pipe = fitted_pipes["hgb"]
hgb_b_pipe = fitted_pipes["hgb_b"]

features = X_val.columns

logr_imp = permutation_importance(logr_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42)
logr_b_imp = permutation_importance(logr_b_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42)

dt_imp = permutation_importance(dt_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42)
dt_b_imp = permutation_importance(dt_b_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42)

rf_imp = permutation_importance(rf_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42)
rf_b_imp = permutation_importance(rf_b_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42)

hgb_imp = permutation_importance(hgb_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42)
hgb_b_imp = permutation_importance(hgb_b_pipe, X_val, y_val, scoring="roc_auc", n_repeats=5, n_jobs=-1, random_state=42)

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















