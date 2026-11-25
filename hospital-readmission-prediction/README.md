# Introduction

A machine learning model to predict readmission of within 30 days after discharge. We use multiple classifiers to predict.
Data is split into 70% training, 15% validation, and 15% for testing. Only 1 record per patient is allowed.
Class imbalance is spotted as the positive class labels represents only 8.79% of the dataset.

# Dataset description

Datset: Diabetes 130-US Hospitals for Years 1999-2008. 
Instances before deduplication: 101766
Instances after deduplication: 71518
Features: 47
Features used in this project:
numerical_features = ["time_in_hospital", "num_lab_procedures", "num_medications", "number_inpatient"]
categorical_features = ["gender", "A1Cresult", "diabetesMed"]
Target: Readmission with 30 days after discharge
Target percentage: 8.79%
Link: https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us-hospitals%2Bfor%2Byears%2B1999-2008?utm_source 
Citation:
Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.



# Technologies

- Python
- Numpy
- Pandas
- Scikit-Learn



# Topics covered

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

- The best models that combine highest recall, precision, specificity, f1, ROC-AUC, and AP are:


| Model | Thresh | Prc | Rec | F1 | Spec | ROC-AUC | AvgPrc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| hgb_b | 0.50 | 0.11 | 0.58 | 0.19 | 0.57 | 0.60 | 0.13 |
| dt_b | 0.50 | 0.11 | 0.58 | 0.18 | 0.56 | 0.59 | 0.12 |
| logr_b | 0.50 | 0.12 | 0.46 | 0.19 | 0.67 | 0.59 | 0.13 |
| rf | 0.1 | 0.12 | 0.39 | 0.18 | 0.72 | 0.59 | 0.13 |

### Balanced gradient boosting and decision tree at threshold 0.50 both show good recall at 0.58 but low precision at only 0.11 i.e. they spot 58% of positive cases but 11 out of 100 positive alerts are actually true positives. 

### Balanced logistic regression at threshold 0.50 shows slightly lower recall than the balanced gradient boosting and decision tree but it shows higher specificity.

### Random forest at threshold 0.1 has the highest specificity of 0.72 among all models with an acceptable precision, recall. It's the model of choice if we care about TN the most.

### All models however display weak discrimination at only 0.59 ROC-AUC. 


# Threshold Interpretation
- Threshold 0.50 shows the best performance among all thresholds across all models especially our selected shortlisted models.  



# Class Imbalance Interpretation
- The balanced models generally produce better outcomes due to their attention on the minority positive class that represents only 8.79% and giving it more weight contribution.



# Calibration interpretation
- Calibration generally produces slightly lower Brier scores for calibrated models which is a good sign of making better calibration but on the expense of significantly worsening most metrics especially for our selected models. Although it makes ROC-AUC and average precision scores become insignificantly higher by 0.02. Our top models such as hgb_b and dt_b at thresholds 0.50 each benefits from calibration with original brier being 0.24 and after calibration they become 0.07 which is a better result but it lowers all other metrics. So generally, calibration doesn't benefit any of our top selected models neither other models for metrics.



# Clinical deployment
Choosing the best model depends on requirements like:
- highest accepted Positive alerts number that combines TP and FP, we need to understand what makes an alert fatigue from excessive FP.
- Consequences of missing TP cases. So, we need to know total number of missed positive cases that can be accommodated.
- External validation using patients from other hospitals
- Clinical deployment also relies on data quality and representativeness, security, hospital requirements and more validation procedures.



# Hyperparamter Tuning
- The parameters of our top models that provided best scores are:
- hgb_b threshold 0.50: 'model__max_leaf_nodes': 10, 'model__max_iter': 150, 'model__learning_rate': 0.05. indicating that the middle balance is the best balance in gradient boosting for our 10 iterations.
- dt_b threshold 0.50: 'model__max_depth': 5, 'model__min_samples_leaf': 10. This is the lowest measurement for both metrics indicating that as we go with higher values, overfitting could happen.
- logr_b at threshold 0.50: 'model__C': 0.1. This is the highest regualarization implemented and it indicate logistic regression produces best results when regularized more. 
- rf at threshold 0.1: 'model__n_estimators': 200, 'model__min_samples_leaf': 30, 'model__max_depth': 15. All these represent the highest values indicating the increase of trees and minimum samples per leaf and the more depth in rf, the more accurate results will be produced.



# logistic regression coef interpretation

number_inpatient is the highest feature for both logr and logr_b indicating the number of previous hospitalizations is associated with our target class the most, as it increases, so is the chance of readmission does. The more previous hospitalizations, the higher the change of a readmission within 30 days.

Other features that also contribute to our positive target class are:

time_in_hospital: The more time spent in hospital, the higher the chance of readmission within 30 days.

diabetesMed_Yes: If a patient was taking diabetes medications, there will be a higher chance of readmission within 30 days

num_lab_procedures: indicating the more number of lab procedures, the higher the chance of readmission with 30 days.



There are some conflicting features between the original and the balanced versions of logistic regression

Both gender_Male and gender_Female appear slightly positive in original model indicating positive impact on readmission probability but slightly negative in balanced model indicating negative impact. 



diabetesMed_No on the other hand is the most negative feature associated with our target of readmission within 30 days i.e. the lower and more negative it is, the lower the readmission probability. Accompanied by other features such as:

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



# Summary 
- best performing models
- Thresholds performance
- Effect of class_weight="balanced" i.e. balancing the datasets
- Calibration effect
- Hyperparameter tuning with best score
- logistic regression coef interpretation
- DT and RF feature_importances interpretation
- permutation importance
- conclusion about clinical deployment




