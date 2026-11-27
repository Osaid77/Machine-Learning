# Project overview

A machine learning model to Predict in-hospital mortality using clinical information collected within first 24 hours of hospitalization. I use multiple classifiers to predict our target including the dummy classifier.
Data is split into 70% training, 15% validation, and 15% for testing and data is already unique i.e. each record represents a different patient.
Class imbalance is spotted as the positive class labels represents only 8.63% of the dataset.


# Clinical problem & objective

In-hospital mortality is widely common after being hospitalized for certain amount of time. I built a model to predict whether a patient will experience mortality within and after 24 hours of admission using a variety of features including categorical, numerical, binary, ordinal, and high-cardinality features.



# Dataset description

- Dataset: WiDS Datathon 2020: ICU Mortality Prediction. 
- Instances: 91713
- No duplicates exist
- Features: 186
- Examined each of the 186 features based on which ones are appropriate from clinical sense perspective, value counts, missingness percentage, skewness, equality, and correlation. Only kept the most relevant ones for predictions.
- Features used in this project are 109 in total:
- categorical_features = ["gender", "hospital_admit_source", "icu_admit_source", "apache_3j_bodysystem", "apache_2_bodysystem"] # 5
- numerical_features = ["age", "bmi", "pre_icu_los_days", "bun_apache", "creatinine_apache", "glucose_apache", "heart_rate_apache", "hematocrit_apache", 
    "map_apache", "resprate_apache", "sodium_apache", "temp_apache", "urineoutput_apache", "wbc_apache", "d1_diasbp_max", "d1_diasbp_min", 
    "d1_diasbp_noninvasive_min", "d1_heartrate_max", "d1_heartrate_min", "d1_mbp_max", "d1_mbp_min", "d1_mbp_noninvasive_min", "d1_resprate_max", "d1_resprate_min", "d1_spo2_min", "d1_sysbp_max", "d1_sysbp_min", 
    "d1_sysbp_noninvasive_max", "d1_sysbp_noninvasive_min", "d1_temp_max", "d1_temp_min", "h1_diasbp_max", "h1_diasbp_min", "h1_diasbp_noninvasive_max", 
    "h1_diasbp_noninvasive_min", "h1_heartrate_max", "h1_heartrate_min", "h1_mbp_max", "h1_mbp_min", "h1_mbp_noninvasive_max", "h1_mbp_noninvasive_min", 
    "h1_resprate_max", "h1_resprate_min", "h1_spo2_max", "h1_spo2_min", "h1_sysbp_max", "h1_sysbp_min", "h1_sysbp_noninvasive_max", "h1_sysbp_noninvasive_min", 
    "h1_temp_max", "h1_temp_min", "d1_albumin_min", "d1_bun_max", "d1_calcium_max", "d1_calcium_min", "d1_creatinine_max", "d1_creatinine_min", 
    "d1_glucose_max", "d1_glucose_min", "d1_hco3_max", "d1_hco3_min", "d1_hemaglobin_min", "d1_hematocrit_min", "d1_platelets_min", "d1_potassium_max", 
    "d1_potassium_min", "d1_sodium_max", "d1_sodium_min", "d1_wbc_min", "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob", "bilirubin_apache", "d1_bilirubin_max",
    "h1_glucose_max", "h1_glucose_min", "d1_arterial_pco2_max", "d1_arterial_pco2_min", "d1_arterial_ph_max", "d1_arterial_ph_min", "d1_arterial_po2_max", 
    "d1_arterial_po2_min", "paco2_apache", "pao2_apache", "d1_lactate_max", "d1_inr_max", "d1_inr_min"]    # 86 
- binary_features = ["elective_surgery", "ventilated_apache", "arf_apache", "apache_post_operative", "gcs_unable_apache", "aids", "cirrhosis", 
                   "diabetes_mellitus", "hepatic_failure", "immunosuppression", "leukemia", "lymphoma", "solid_tumor_with_metastasis", "intubated_apache"]  # 14
- high_cardinality_features = ["apache_3j_diagnosis"] # 1
- ordinal_features = ["gcs_eyes_apache", "gcs_motor_apache", "gcs_verbal_apache"] # 3
- Limitation: only 109 features were chosen out of 186 and there is very minor redundancy within features 
- Target: Mortality within hospitalization
- Target percentage: 8.63%
- Features like "apache_4a_icu_death_prob" and "apache_4a_hospital_death_prob" included -1 probability for approx. 2000 records for each. I replaced -1 with the median probability as there was some skewness spotted and mean was inappropriate to be used
- Citation:
Lee, Meredith, et al. "WiDS (Women in Data Science) Datathon 2020: ICU Mortality Prediction" (version 1.0.0). PhysioNet (2020). RRID:SCR_007345. https://doi.org/10.13026/vc0e-th79

# Data preparation & leakage prevention

- No duplicates exist in the data
- Splitting before preprocessing as I do stratification which preserves approximately the same class proportions among training, validation and testing set. So, the positive class becomes 8.63% among all sets.
- fitting transformation on training data only
- Separated the preprocessing of each of the feature categories: numerical, categorical, binary, high-cardinality, and ordinal features. Each in its own preprocessing pipeline.


# Technologies

- Python
- Numpy
- Pandas
- Scikit-Learn


# Installation & usage
| Step | Instruction |
| :--- | ---: |
| 1 | Download the dataset | 
| 2 | Place training_v2.csv inside the project folder | 
| 3 | Install dependencies using pip install -r requirements.txt | 
| 4 | Execute the script using Python mortality_model.py



# Reproducibility
rnadom_state=42 was implemented across training, validation, and testing splitting. It's also implemented in all models training, hyperparameter tuning and stratification, and permutation importance.



# Machine Learning methodology

- Preprocessing techniques including imputing, scaling and one-hot encoding
- Models: DummyClassifier, Logistic regression, Decision tree, Random forest, and Gradient boosting
- Pipelines and Columntransformers
- Metrics including precision, recall, f1 score, ROC-AUC, and Average precision
- Confusion Matrix
- Dealing with an imbalanced dataset
- Calibration
- Cross validation
- Hyperparamter tuning
- Permutation importance
- Logistic regression coefficients
- Decision tree and Random forests feature importances



# Results Interpretation

- I prioritzed recall due to the sensitivity of missing TP alerts as this is a mortality prediction so we should prioritize not missing TP cases and minimize FN even in the expense of having excessive alerts.
- The best models that combine highest recall, precision, specificity, f1, ROC-AUC, AP, and Brier are:


| Model | Thresh | Prc | Rec | F1 | Spec | ROC-AUC | AvgPrc | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| logr | 0.10 | 0.31 | 0.74 | 0.44 | 0.84 | 0.88 | 0.52 | 0.056 |
| logr_b | 0.30 | 0.20 | 0.89 | 0.33 | 0.67 | 0.88 | 0.50 | 0.12 |
| dt_b | 0.30 | 0.188 | 0.86 | 0.31 | 0.65 | 0.85 | 0.41 | 0.14 |
| rf | 0.10 | 0.286 | 0.78 | 0.419 | 0.81 | 0.88 | 0.527 | 0.056 |
| rf_c | 0.10 | 0.389 | 0.65 | 0.488 | 0.90 | 0.88 | 0.527 | 0.057 |
| rf_b | 0.30 | 0.218 | 0.89 | 0.35 | 0.699 | 0.886 | 0.51 | 0.10 |
| hgb | 0.10 | 0.32 | 0.777 | 0.457 | 0.847 | 0.894 | 0.56 | 0.054 |
| hgb_b | 0.20 | 0.185 | 0.93 | 0.309 | 0.61 | 0.89 | 0.55 | 0.11 |
| hgb_b_c | 0.20 | 0.424 | 0.636 | 0.509 | 0.918 | 0.89 | 0.548 | 0.54 |
| hgb_b | 0.50 | 0.305 | 0.78 | 0.439 | 0.83 | 0.89 | 0.55 | 0.11 |

### - The choice here is difficult because there are many good models with tempting high recall values but generally the original hgb at threshold 0.10 is the most promising, at 0.77 recall, 0.32 precision, 0.45 F1, 0.84 specificity, 0.89 ROC-AUC, 0.56 Average precision, and 0.054 Brier score. 
### It displays the highest F1, ROC-AUC, Average prec, and specificity, and lowest Brier score among all competitor models. It's the model of choice as it balances all competitor models perfectly. Its calibrated version just below it competes strongly with it but we cannot choose it because it provides lower recall which is our #1 metric.

### - Other competitor models include the highest recall models, outperforming the recall of hgb are: 
- hgb_b at thre 0.20 showing recall 0.93. Its calibrated version is so tempting showing the highest precision in the list at 0.42
- rf_b at thre 0.30 showing recall 0.89
- logr_b at thre 0.30 showing recall 0.89
- dt_b at thre 0.30 showing recall 0.86
### Unfortunately, all of them show a weak precision at range of 0.18-0.20 which hgb outperforms. Also, hgb outperforms them in F1, specificity, ROC-AUC, Average precision, and Brier score. 

### Other competitor models but are outperformed by hgb in every metric are: 
- logr thre 0.10 
- rf thre 0.10
- hgb_b thre 0.50

### All models however display good discrimination at ROC-AUC 0.88-0.89 for most models. 

### Dummy classifier was outperformed by all our models. Even though it displayed accuracy of 0.914 which is the highest accuracy among all models, it showed 0 recall and precision with 0.50 ROC-AUC which is also the lowest. That's why accuracy alone is misleading when interpreted alone. 


## Threshold Interpretation
- Thresholds 0.10, and 0.30 show the best performance among all thresholds across all models especially our selected shortlisted models.
- 0.10 threshold in particular shows the highest recall among other thresholds within same model e.g. hgb threshold 0.10 has higher recall than hgb thresholds 0.20, 0.30, and 0.50.
- However, as the threshold increases, the precision increases, specificity increases, but recall decreases. This applies to all models. 
- Generally, all balanced models achieved best results with thresholds 0.30 and 0.50. Meanwhile, original models achieved best results at thresholds 0.10 and 0.20. 



## Class balancing Interpretation
- Even though class balance gives more attention & weight contribution to minority positive class which only represents 8.63% in our dataset, balanced models tend to perform pretty similar to original models in our project. Balanced models tend to produce higher recall but in the expense of lower precision.  


## Calibration interpretation
- Calibration generally produces lower Brier scores for calibrated models meaning it improves probability quality which is a good sign of making better calibration but on the expense of significantly worsening most metrics especially for our selected models.
- Calibration produces lower recall and higher precision and specificity for most models. Even in this case, our selected models of choice above are produce better balance without calibration. Recall is the most important metric in our case and calibration lowers it for most models.
- It doesn't add any benefit to our model of choice: hgb at thre 0.10.



# Clinical deployment
Choosing the best model depends on clinical engineering judgement and requirements like:
- highest accepted Positive alerts number that combines TP and FP, we need to understand what makes an alert fatigue from excessive FP alerts.
- Consequences of missing TP cases. So, we need to know total number of missed positive cases that can be accommodated especially in such a sensitive case like mortality.
- External validation using patients from other hospitals.
- Clinical deployment also relies on data quality and representativeness, security, hospital requirements and more validation procedures.



# Hyperparamter Tuning
- The parameters of our top models that provided best scores are:
- hgb threshold 0.10: 'model__max_leaf_nodes': 30, 'model__max_iter': 150, 'model__learning_rate': 0.05. indicating that the middle balance is the best balance in gradient boosting for our 10 iterations in RandomizedSearchCV.
- All other hgb original & calibrated threshold 0.20 and hgb_b at thre 0.50: 'model__max_leaf_nodes': 20, 'model__max_iter': 100, 'model__learning_rate': 0.1. indicating the best balance at threshold 0.20 is also the middle balance.
- rf models all achieved best results at: 'model__n_estimators': 200, 'model__min_samples_leaf': 10, 'model__max_depth': 15. All these represent the highest values except min_samples_leaf which is the loest value we tested. This indicates the increase of trees and the more depth in rf with low minimum samples per leaf, the more accurate results will be produced.
- dt_b threshold 0.30: 'model__max_depth': 5, 'model__min_samples_leaf': 20. This is the lowest depth and highest samples per leaf in all decision tree models for both metrics indicating that as we go with with high depth, overfitting could happen. Also, preferring mixing low depth with highest samples per leaf to learn more about the data.
- logr_b at threshold 0.30 and logr models: 'model__C': 0.1. This is the highest regularization implemented and it indicates that all logistic regression models produce best results when regularized more.



# hna wslna: here next:


# logistic regression coef interpretation

## Top and bottom features sorted by logr:

| Feature | logr_coeff | logr_b_coeff |
| :--- | :---: | ---: |
| high_cardinality__apache_3j_diagnosis_401.01 | 0.725 | 0.756 |
| high_cardinality__apache_3j_diagnosis_202.01 | 0.679 | 0.798 | 
| high_cardinality__apache_3j_diagnosis_102.01 | 0.664 | 0.858 | 
| binary__ventilated_apache | 0.594 | 0.583 | 
| binary__solid_tumor_with_metastasis | 0.549 | 0.595 | 
| high_cardinality__apache_3j_diagnosis_306.01 | -0.511 | -0.688 | 
| binary__elective_surgery | -0.529 | -0.546 | 
| high_cardinality__apache_3j_diagnosis_410.01 | -0.564 | -0.595 |
| high_cardinality__apache_3j_diagnosis_1206.03 | -0.599 | -0.719 |
| high_cardinality__apache_3j_diagnosis_407.01 | -0.728 | -0.689 |

## Top and bottom features sorted by logr_b:

| Feature | logr_coeff | logr_b_coeff |
| :--- | :---: | ---: |
| high_cardinality__apache_3j_diagnosis_102.01 | 0.664 | 0.858 |
| high_cardinality__apache_3j_diagnosis_202.01 | 0.679 | 0.798 | 
| high_cardinality__apache_3j_diagnosis_401.01 | 0.725 | 0.756 | 
| high_cardinality__apache_3j_diagnosis_nan | 0.326 | 0.641 | 
| high_cardinality__apache_3j_diagnosis_1204.02 | 0.370 | 0.614 | 
| high_cardinality__apache_3j_diagnosis_306.01 | -0.511 | -0.688 | 
| high_cardinality__apache_3j_diagnosis_407.01 | -0.728 | -0.689 | 
| high_cardinality__apache_3j_diagnosis_1206.03 | -0.599 | -0.719 |
| high_cardinality__apache_3j_diagnosis_1207.01 | -0.508 | -0.733 |
| high_cardinality__apache_3j_diagnosis_209.01 | -0.414 | -0.805 |

- high_cardinality__apache_3j_diagnosis_401.01: intracranial haemorrhage or haematoma
- high_cardinality__apache_3j_diagnosis_202.01: Respiratory arrest
- high_cardinality__apache_3j_diagnosis_102.01: Cardiac arrest
- high_cardinality__apache_3j_diagnosis_306.01: Bleeding, lower GI
- high_cardinality__apache_3j_diagnosis_410.01: Coma/change in level of consciousness
- high_cardinality__apache_3j_diagnosis_1206.03: Coronary Artery Bypass Graft (CABG) with single valve repair or replacement
- high_cardinality__apache_3j_diagnosis_407.01: Seizures (primary - no structural brain disease)
- high_cardinality__apache_3j_diagnosis_nan: Missing/unrecorded record
- high_cardinality__apache_3j_diagnosis_1204.02: Aneurysm, thoracic
- high_cardinality__apache_3j_diagnosis_1207.01: Coronary artery bypass graft (CABG) with valve repair or replacement
- high_cardinality__apache_3j_diagnosis_209.01: Asthma

- Diagnoses share the highest percentage in logistic regression ranking and there is a high correlation between the original and the balanced model.
- logr top positive diagnoses are .... meaning ...
- logr top negative diagnoses are .... meaning ...
- logr_b top positive diagnoses are .... meaning ...
- logr_b top negative diagnoses are .... meaning ...
  
Both models agree on top 5 positive features and bottom 5 negative features 




number_inpatient is the highest feature for both logr and logr_b indicating the number of previous hospitalizations is associated with our target class the most, as it increases, so is the chance of readmission does. The more previous hospitalizations, the higher the change of a readmission within 30 days.

Other features that also contribute to our positive target class are:

time_in_hospital: The more time spent in hospital, the higher the chance of readmission within 30 days.

diabetesMed_Yes: If a patient was taking diabetes medications, there will be a higher chance of readmission within 30 days

num_lab_procedures: indicating the more number of lab procedures, the higher the chance of readmission with 30 days.



There are some conflicting features between the original and the balanced versions of logistic regression

Both gender_Male and gender_Female appear slightly positive in original model indicating positive impact on readmission probability but slightly negative in balanced model indicating negative impact. 



diabetesMed_No on the other hand is the most negative feature associated with our target of readmission within 30 days i.e. it is associated with lower readmission probability. Accompanied by other features such as:

A1Cresult\_>7

A1Cresult\_>8

A1Cresult_Norm

All prove that normal HgbA1C is associated with lower probability of readmission within 30 days.



# Decision tree and random forests feature importances

Original and balanced models differ in their choices of feature importance for purity and homogenity. 

num_lab_procedures feature is the top feature that help tree purity/homogeniety in both original DT and RF. Closely Followed by num_medications.

number_inpatient features is the top feature that help tree purity/homogeniety in both balanced DT and RF. Closely followed by time_in_hospital.

Reflecting how balanced models rely more on features with minority class.



# Feature importances

- All models logistic regression, decision trees, random forests, and gradient boosting originals and balanced rely on number_inpatient i.e. previous number of hospitalisations for ranking positive from negative class. Followed by time_in_hospital and diabetesMed.
However, there's a disagreement between trees feature importance and permutation importance

| Model | Feature | Tree importance | Permutation importance |
| :--- | :---: | :---: | ---: |
| dt | num_lab_procedures | 0.35 | -0.015 |

This means this feature contributes a lot in purity of the original decision tree and actually provided better shuffled ROC-AUC than the original ROC-AUC. 


# Test set evaluation
| Metric | Validation | Test |
| :--- | :---: | ---: |
| Acc | 0.57 | 0.56 |
| Prc | 0.11 | 0.11 | 
| Rec | 0.58 | 0.57 | 
| F1 | 0.19 | 0.18 | 
| Spc | 0.57 | 0.56 | 
| Roc | 0.60 | 0.60 | 
| AP | 0.13 | 0.13 | 
| Brier | 0.24 | 0.24 |

- Based on the results shown, we can see that testing and validation are pretty similar or the same confirming consistency of results and good generalization with unseen patients within the same dataset. Discrimination however remains weak despite good consistency.
- External validation is required.   


# Project limitations
- Imbalanced dataset with the positive class representing only 8.79% of the entire dataset.
- A historical dataset: Diabetes 130-US Hospitals for Years 1999-2008
- Weak discrimination at a range of ROC-AUC at only 0.60
- Lack of external validation


 # Future improvements
 - Additional clinical predictors
 - More thresholds to be selected like [0.08, 0.15, 0.20, 0.25]
 - More feature engineering and external validation
