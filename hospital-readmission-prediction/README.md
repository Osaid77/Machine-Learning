A machine learning model to predict readmission of within 30 days after discharge. We use multiple classifiers to predict

Data is split into 70% training, 15% validation, and 15% for testing. Only 1 record per patient is allowed.&#x20;

Class imbalance is spotted as the positive class labels represents only 8.79% of the dataset.&#x20;

Features used:&#x20;

numerical_features = ["time_in_hospital", "num_lab_procedures", "num_medications", "number_inpatient"]\
categorical_features = ["gender", "A1Cresult", "diabetesMed"]\


Technologies used:

- Python
- Numpy
- Pandas
- Scikit-Learn



Topics covered:

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



Results Interpretation:

- The best models that combine highest recall, precision, f1, ROC-AUC, and AP are:&#x20;

\#1 hgb_b threshold 0.50: prc 0.11, rec 0.58, f1 0.19, spc 0.57, ROC-AUC 0.60, AP 0.13.&#x20;

\#2 dt_b threshold 0.50: prc 0.11, rec 0.58, f1 0.18, spc 0.56, ROC-AUC 0.59, AP 0.12.

\#3 logr_b at threshold 0.50: prc 0.12, rec 0.46, f1 0.19, spc 0.67, ROC-AUC 0.59, AP 0.13. It shows slightly lower recall than the above two but it shows higher specificity.

\#4 rf at threshold 0.1: prc 0.12, rec 0.39, f1 0.18, spc 0.72, ROC-AUC 0.59, AP 0.13. This is the highest specificity among all model with an acceptable precision, recall. Model of chose if we care about TN the most.&#x20;



- Threshold 0.50 shows the best performance among the thresholds reflecting that lowering the threshold doesn't reveal higher TP and TN nor lowering FP and FN. Instead, it mostly provides lower performances across the metrics.&#x20;



- The balanced models generally produce better outcomes due to their attention on the minority positive class that represents only 8.79%.&#x20;



- Calibration generally produces slightly lower Brier scores for calibrated models which is agood sign of making better calibration but on the expense of significantly worsening most metrics especially for our selected models. Although it makes ROC-AUC and average precision scores become insignificantly higher by 0.02. Our top models such as hgb_b and dt_b at thresholds 0.50 each benefits from calibration with original brier being 0.24 and after calibration it becomes 0.07 which is a better result but it lowers all other metrics. So generally, calibration doesn't benefit any of our top selected models neither other models for metrics.&#x20;



- Choosing the best model depends on requirements like highest accepted Positive alerts number that combines TP and FP. Also, we need to know total number of missed positive cases that can be accommodated. Clinical deployment also relies on data quality and representativeness, security, hospital requirements and more validation procedures.



- The parameters of our top models that provided best scores are:
- hgb_b threshold 0.50: 'model__max_leaf_nodes': 10, 'model__max_iter': 150, 'model__learning_rate': 0.05. indicating that the middle balance is the best balance in gradient boosting.
- dt_b threshold 0.50: 'model__max_depth': 5, 'model__min_samples_leaf': 10 &#x20;
- logr_b at threshold 0.50: 'model__C': 0.1
- rf at threshold 0.1: 'model__n_estimators': 200, 'model__min_samples_leaf': 30, 'model__max_depth': 15. Indicating the increase of trees in rf produces most accurate results.&#x20;



- logistic regression coef interpretation:&#x20;

number_inpatient is the highest feature for both logr and logr_b indicating the number of previous hospitalizations determines our target class the most, as it increases, so is the chance of readmission does. The more previous hospitalizations, the higher the change of a readmission within 30 days.&#x20;

Other features that also contribute to our positive target are:

time_in_hospital: The more time spent in hospital, the higher the chance of readmission within 30 days.

diabetesMed_Yes: the more diabetes medications being taken, the higher the chance of readmission within 30 days

num_lab_procedures: indicating the more number of lab procedures, the higher the chance of readmission with 30 days.&#x20;



There are some conflicting features between the original and the balanced versions of logistic regression:&#x20;

gender_Male: appears slightly positive in original model indicating positive impact on readmission probability

gender_Female: appears slightly negative in the balanced model indicating negative impact on readmission probability.

For these two features, we might rely more on the balanced version as it gives more attention to the minority class and this could be more accurate in most scenarios.&#x20;



diabetesMed_No on the other hand is the feature that's the most protective from our target i.e. the lower and more negative it is, the lower the readmission probability. Accompanied by other features such as:

A1Cresult\_>7

A1Cresult\_>8

A1Cresult_Norm

All prove that normal HgbA1C can be protective from readmission within 30 days probability after readmission.&#x20;



- Decision tree and random forests feature importances:

Original and balanced models differ in their choices of feature importance for purity and homogenity. &#x20;

num_lab_procedures features is the top feature that help tree purity/homogeniety in both original DT and RF. Closely Followed by num_medications.

number_inpatient features is the top feature that help tree purity/homogeniety in both balanced DT and RF. Closely followed by time_in_hospital.&#x20;

Reflecting how balanced models rely more on features with minority class.&#x20;



- Feature importances:

All models logistic regression, decision trees, random forests, and gradient boosting originals and balanced rely on number_inpatient i.e. previous number of hospitalisations for ranking positive from negative class. Followed by time_in_hospital and diabetesMed.&#x20;



Now, I'm building my interpretation skills of ML results, so I can write my own human way in the README file in the GitHub portfolio. Above I talked about:&#x20;

- best performing models
- Thresholds performance
- Effect of class_weight="balanced" i.e. balancing the datasets
- Calibration effect&#x20;
- Hyperparameter tuning with best score
- logistic regression coef interpretation
- DT and RF feature_importances interpretation
- permutation importance
- conclusion about clinical deployment





Can I just write these in the README as they are? Anything wrong in my interpretation?

Also, Is there anything else I should write about besides these in the README? What is it?

You cannot exactly give me the text and I copy and paste, that's illegal and I will be caught using AI, but you can tell me what are the things to talk about.
