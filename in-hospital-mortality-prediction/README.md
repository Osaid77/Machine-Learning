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
- Total Features in the dataset: 186
- Examined each of the 186 features based on which ones are appropriate from clinical sense perspective, value counts, missingness percentage, skewness, equality, and correlation. Only kept the most relevant ones for predictions.
- Features used in this project are 109 in total including categorical, numerical, binary, high-cardinality, and ordinal features. They span demographic, lab, vitals, probabilities, and diagnoses.

| Feature group | Count | Examples |
| :--- | :---: | ---: |
| Categorical | 5 | gender, admission source, body system |
| Numerical | 86 | age, creatinine, lactate, blood pressure, glucose, sodium | 
| Binary | 14 | Ventilation, intubation, comorbidities | 
| Ordinal | 3 | GCS eye, verbal and motor scores | 
| High-cardinality | 1 | APACHE III diagnosis | 
| Total | 109 |  | 

- Limitation: only 109 features were chosen out of 186 and there is a minor redundancy within those selected features 
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
| hgb_b_c | 0.20 | 0.424 | 0.636 | 0.509 | 0.918 | 0.89 | 0.548 | 0.054 |
| hgb_b | 0.50 | 0.305 | 0.78 | 0.439 | 0.83 | 0.89 | 0.55 | 0.11 |

### The choice here is difficult because there are many good models with tempting high recall values but generally the original hgb at threshold 0.10 with its calibrated version is the most promising, at 0.77 recall, 0.32 precision, 0.45 F1, 0.84 specificity, 0.89 ROC-AUC, 0.56 Average precision, and 0.054 Brier score. 
### It displays the highest F1, ROC-AUC, Average prec, and specificity, and lowest Brier score among all competitor models shown. It's the model of choice as it balances all metrics perfectly. Its calibrated version just below it competes strongly with it as it provides slightly higher ROC-AUC and AP and slightly lower Brier score.

### Other competitor models include the highest recall models, outperforming the recall of hgb are: 
- hgb_b at thre 0.20 showing recall 0.93. Its calibrated version is so tempting to choose showing the highest precision in the list at 0.42
- rf_b at thre 0.30 showing recall 0.89
- logr_b at thre 0.30 showing recall 0.89
- dt_b at thre 0.30 showing recall 0.86
### Unfortunately, all of them show a weak precision at range of 0.18-0.20 which hgb outperforms. Also, hgb outperforms them in F1, specificity, ROC-AUC, Average precision, and Brier score. 

### Other competitor models but are outperformed by hgb in most metrics are: 
- logr thre 0.10 
- rf thre 0.10
- hgb_b thre 0.50

### All models however display good discrimination at ROC-AUC 0.88-0.89 for most models. 

### Dummy classifier was outperformed by all our models. Even though it displayed accuracy of 0.914 which is one of the highest accuracies among all models, it showed 0 recall and precision with 0.50 ROC-AUC which is also the lowest. That's why accuracy alone is misleading when interpreted alone. 


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
- high_cardinality__apache_3j_diagnosis_nan: Missing/unrecorded diagnosis
- high_cardinality__apache_3j_diagnosis_1204.02: Aneurysm, thoracic
- high_cardinality__apache_3j_diagnosis_1207.01: Coronary artery bypass graft (CABG) with valve repair or replacement
- high_cardinality__apache_3j_diagnosis_209.01: Asthma

### Among all 109 features including labs, vitals, demographics and others, the high-cardinality feature "apache_3j_diagnosis" share the highest percentage in logistic regression ranking. Generally, there is a high correlation between the original and the balanced logr.

### logr top positive diagnoses in order are:

- intracranial haemorrhage
- respiratory arrest
- cardiac arrest
- ventilated apache/receiving ventilation
- tumor with metastasis
- It means they're associated with our target class of in-hospital mortality the most. As these features exist, so is the chance of mortality. The higher the number is, the more association with mortality. The lower the number is, the lower association.

  
### logr top negative diagnoses in order are:

- seizure without structural brain disease
- Coronary Artery Bypass Graft (CABG) with single valve repair or replacement
- Coma/change in level of consciousness
- elective/non-urgent surgery
- bleeding/lower GI
- Meaning they're the least associated with our target class of in-hospital mortality. The more negative the number is, the less the association with mortality. The less negative the number is, the less is the association. 

#### logr_b top positive diagnoses are cardiac arrest, respiratory arrest, intracranial hemorrhage, Missing/unrecorded diagnosis, and Aneurysm, thoracic.
#### logr_b top negative diagnoses are Asthma, Coronary artery bypass graft (CABG) with valve repair or replacement, CABG with single valve repair or replacement, Seizures without structural brain disease, Bleeding lower GI.

  
## Both models share an extreme correlation in positive and negative features. However, they disagree on some features such as:

| Feature | logr_coeff | logr_b_coeff |
| :--- | :---: | ---: |
| numerical__missingindicator_apache_4a_hospital_death_prob | 0.078 | -0.044 |
| numerical__hematocrit_apache | 0.063 | -0.010 | 
| numerical__missingindicator_d1_spo2_min | 0.017 | -0.003 | 
| numerical__h1_glucose_max | 0.015 | -0.027 | 
| numerical__bilirubin_apache | 0.015 | -0.009 | 
| numerical__d1_potassium_min | 0.011 | -0.009 | 
| numerical__missingindicator_d1_wbc_min | 0.008 | -0.060 | 
| numerical__missingindicator_d1_resprate_max | 0.007 | -0.001 | 
| numerical__missingindicator_d1_resprate_min | 0.007 | -0.001 | 
| categorical__gender_F | -0.321 | 0.041 | 
| categorical__gender_M | -0.289 | 0.093 | 
| high_cardinality__apache_3j_diagnosis_211.09 | -0.232 | 0.037 | 
| categorical__apache_3j_bodysystem_Sepsis | -0.115 | 0.003 | 
| categorical__hospital_admit_source_Emergency Department | -0.107 | 0.013 | 
| numerical__missingindicator_apache_4a_icu_death_prob | -0.094 | 0.012 | 
| numerical__apache_4a_icu_death_prob | -0.077 | 0.014 | 
| high_cardinality__apache_3j_diagnosis_204.01 | -0.061 | 0.092 | 

- high_cardinality__apache_3j_diagnosis_204.01: ARDS — Acute Respiratory Distress Syndrome / non-cardiogenic pulmonary edema
- high_cardinality__apache_3j_diagnosis_211.09: Respiratory disorder, other

### Some of the most prominent positive features for logr and negative for logr_b are:
- hematocrit_apache
- bilirubin_apache
- d1_spo2_min: lowest oxygen saturation measured in first 24 hours
- h1_glucose_max: highest glucose measured in first hour
- d1_potassium_min: lowest potassium measured in first 24 hours
- Those features are more likely associated with mortality for original logr but NOT associated with mortality in balanced logr.


### Some of the most prominent negative features for logr and positive for logr_b are:
- ARDS — Acute Respiratory Distress Syndrome / non-cardiogenic pulmonary edema
- hospital_admit_source_Emergency Department
- apache_3j_bodysystem_Sepsis: diagnosed as sepsis case
- Respiratory disorder, other.
- Those features are more likely associated with mortality for balanced logr but NOT associated with mortality in original logr.


### Gender difference is as follows:
- Original logr: negative for both male and female, meaning neither genders is associated with increased mortality i.e. gender doesn't play a role in mortality prediction. However, male gender is more likely associated with mortality because it demonstrates less negative coefficient. 
- Balanced logr: positive for both male and female, meaning both genders are associated with increased mortality i.e. gender plays a role in mortality prediction. However, male gender is more like associated with mortality because it demonstrates higher positive value.  


### Intracranial haemorrhage, respiratory arrest, cardiac arrest are the top 3 highest positive features for both logr and logr_b indicating that these conditions are associated with our target class the most, as they exist, so is the chance of mortality does.

### Other features agreed on the most by the balanced and original models that also contribute to our positive target class are: 
- ventilated_apache: whether the patient received ventilation or not
- solid_tumor_with_metastasis: binary
- Subdural/Epidural haematoma
- Acute hepatic failure
- Subarachnoid haemorrhage / intracranial aneurysm
- Head (CNS) only trauma
- arf_apache: Acute renal failure in first 24 hours of hospital admission
- immunosuppression: binary feature
- Head trauma +/- multi trauma
- Acid-Base / electrolyte disturbance

#### Notice that all those features are mostly derived from the high cardinality "apache_3j_diagnosis" feature. The presence of any of these reflects the most association with mortality.


### On the other hand, the most negative features agreed on by both original and balanced logr are:
- Seizures (primary-no structural brain disease)
- Coma/change in level of consciousness
- elective_surgery: Binary whether the patient was admitted for elective surgery
- Bleeding, lower GI: like hemorrhoids, anal fistula/fissure
- CABG alone, coronary artery bypass grafting
- icu_admit_source_Operating Room / Recovery
- Bleeding, upper GI: like Peptic ulcer disease
- Asthma
- Carbon monoxide, arsenic, or cyanide poisoning
- Diabetic ketoacidosis (DKA)

#### These features are the least associated with in-hospital mortality and they all belong to the high cardinality "apache_3j_diagnosis" feature. 




# Decision trees and random forests feature importances

#### Original and balanced decision tree models are correlated in their choices of feature importance for purity and homogenity. They both use these features the most:
- apache_4a_icu_death_prob: probabilistic prediction of in ICU mortality for the patient based on Apache 4a. Original decision tree gives it 0.445 feature importance for homogenity.
- apache_4a_hospital_death_prob: probabilistic prediction of in-hospital mortality based on Apache 4a. Balanced decision tree gives it 0.793 feature importance for homogenity.
- d1_lactate_max: Highest lactate concentration in first 24 hours.
- d1_heartrate_min: Lowest heart rate in first 24 hours.
- d1_spo2_min: Lowest o2 saturation in first 24 hours.
- d1_temp_max: Highest temperature recorded in first 24 hours.
- d1_sysbp_min: Lowest systolic bp recorded in first 24 hours.


#### Original and balanced random forest models use the same features above for purity & homogenity and even other features including:
- numerical__d1_sysbp_noninvasive_min: Lowest systolic blood pressure in first 24 hours, non-invasively measured
- numerical__d1_sysbp_min: Lowest systolic blood pressure during in first 24 hours, either non-invasively or invasively measured
- d1_arterial_ph_min: Lowest arterial pH in first 24 hours
- gcs_motor_apache: motor component of the Glasgow Coma Scale



# Permutation importances

### ROC-AUC is used to measure performance changes between original validation ROC-AUC and mean shuffled ROC-AUC. All models logistic regression, decision trees, random forests, and gradient boosting originals and balanced rely on similar set of features for ranking positive from negative classes including:
- logr + logr_b: age - apache_4a_hospital_death_prob - h1_mbp_noninvasive_min - d1_bun_max - d1_hemaglobin_min - apache_3j_diagnosis - d1_mbp_min 
- dt + dt_b: apache_4a_hospital_death_prob - apache_4a_icu_death_prob - d1_spo2_min - d1_lactate_max 
- rf + rf_b: apache_4a_hospital_death_prob - apache_4a_icu_death_prob - age - ventilated_apache 
- hgb + hgb_b: apache_4a_hospital_death_prob - apache_4a_icu_death_prob - age - ventilated_apache - d1_spo2_min



However, there is an agreement between trees feature importance and permutation importance. So, original & balanced dt tree importances and permutation importances are both positive. 

| Model | Feature | Tree importance | Permutation importance |
| :--- | :---: | :---: | ---: |
| dt | apache_4a_hospital_death_prob | 0.143 | 0.11 |
| dt | apache_4a_icu_death_prob | 0.445 | 0.38 |
| dt | d1_spo2_min | 0.031 | 0.006 |
| dt | d1_lactate_max | 0.066 | 0.006 |
| dt | age | 0.005 | 0.000 |
| dt | ventilated_apache | 0.001 | 0.000 |
| rf | apache_4a_hospital_death_prob | 0.092 | 0.013 |
| rf | apache_4a_icu_death_prob | 0.083 | 0.010 |
| rf | age | 0.010 | 0.003 |
| rf | ventilated_apache | 0.017 | 0.003 |


This means these features contribute a lot in purity of the decision trees and random forests and actually provided lower shuffled ROC-AUC than the original validation ROC-AUC. Which means shuffling the feature hurts performance because shuffling harmed ROC-AUC and made it lower. 










# I will write the test set evaluation section after you give me feedback on all what I wrote first. I plan to go with hgb thre 0.10 for testing set 

# Test set evaluation

### Out model of choice is: gradient boosting at threshold 0.10.

| Metric | Validation | Test |
| :--- | :---: | ---: |
| Acc | 0.84 |  |
| Prc | 0.32 |  | 
| Rec | 0.777 |  | 
| F1 | 0.457 |  | 
| Spc | 0.847 |  | 
| Roc | 0.894 |  | 
| AP | 0.56 |  | 
| Brier | 0.054 |  |


## ignore these 2 bullet lines:
- Based on the results shown, we can see that testing and validation are pretty similar or the same confirming consistency of results and good generalization with unseen patients within the same dataset. Discrimination however remains weak despite good consistency.
- External validation is required.   


# Project limitations
- Imbalanced dataset with the positive class representing only 8.63% of the entire dataset.
- A historical dataset: WiDS Datathon 2020: ICU Mortality Prediction. 
- Lack of external validation


 # Future improvements
 - Additional clinical predictors
 - More thresholds to be selected like [0.07, 0.15, 0.25]
 - More feature engineering and external validation
