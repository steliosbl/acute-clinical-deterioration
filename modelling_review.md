# Prior Work on Modelling Health Outcomes with ML

## Review: Explainable ML for Hospital Mortality - Stenwig
Provides guidelines/framework for developemnt process and what to include in explainable ML report.

 1. Motivations
    - Papers generally only focus on the discriminative capabilities of models. However, ML are generally black boxes and models that are not trusted will not be used. 
    - Investigating how individual features impact predictions from different ML models to learn how they compare to common medical theory.

 2. Data Preprocessing
    - Uses same features as baseline APACHE score
    - Tries scaling, removing patients with over $n$ missing values, filling missing values with APACHE reference values.
    - Tries different ratios between classes (deceased and alive patients).
 3. Model Training
    - Random Forest, LR, Adaptive Boost, Naive Bayes
    - Computes SHAP values, but only for a subset of the data due to high computational cost.
    - Intentionally does not calibrate models before presenting results.
 4. Results
    - Presents ROC and calibration curves (showing model distribution vs data distribution).
    - Presents SHAP summary plots:
        - Bar chart where lengths show mean SHAP value per feature.
        - Density plot aggregating individual rows' SHAP per feature.
        - SHAP force plots of average effect (y) of continuous features (x).
        - SHAP force plots of cumulative effect of features on label:

![SHAP force plot](/images/Stenwig22_Fig8.png)

## Readmission Risk with COPD using ML - Min
### Background:
 - Prior studies on EHR data have demonstrated the potential of ML for hospital readmission prediction compared to LACE and HOSPITAL
 - This paper predicting readmission risk for COPD patients using ML on longitudinal claims records. Output is probability of readmission within 30 days. Prediction is to be made *on discharge*.
 - Temporal dimension in patient records is important as it may indicate the disease progression process [pp. 4]

### Data Preprocessing

 - Kept only patients with COPD, over 40y/o, Male, with at least one inpatient admission, and at least 60 days total. 
 - Total 112k admissions involving 27k patients (92k first admissions and 20k readmissions)

**Features**:
- Worked out HOSPITAL (HOS only) and LACE. 
    - (a) counted number of procedures during the spell, 
    - (b) counted number of admissions in the previous year
    - (c) counted number of stays >= 5 days long
    - (d) determined acute admission or not
    - (e) counted length of stay
    - (f) counted number of ED visits within 6 months
- (g) age
- (h) total length of all stays in the previous year
- (i) number of all kinds of admissions (including outpatient)
- (j) type of admission (transfer, readmission)
- Diagnostic Codes:
    - Grouped based on 3-code
    - Grouped based on CCS
    - Grouped based on HCC
- Procedure Codes:
    - Grouped based on CCS
    - Grouped based on BETOS
    - Grouped by revenue code (?)
- Pharmacy/medications prescribed

**Preprocessing for ML**: Aggregate admissions into an observation window (1 year prior to discharge) to form the patient vector. Ignore temporality otherwise.
 - Bag-Of-Words: Count the frequency of each feature within the time period.
 - Boolean Bag-Of-Words: Only consider whether each feature appears or not within the time period.
 - TFIDF: Normalisation of BoW by inverse feature popularity, to suppress highly prevalent features (which could be non-informative).

**Preprocessing for DL**:
 - Encoding the admission records per patient:
	- Baseline: One-hot encoding
	- Contextual embedding with Skip-grams:
		- Using a time window (instead of a context window) i.e. the time gap between successive medical codes.
		- Weighing the event pairs accoridng to the time gap between them (closer is weighed higher).
	- Med2Vec (published embedding system)
 - Encoding the full patient record:
	- Sequence Representation: Embed each event using one of the above techniques, then weigh it by the timedelta between it and the discharge date. 
	- Matrix Representation with Regular Time Intervals: Each column is a medical event and each row represents a regular time interval. Cells are 0 unless that event occurred that many time intervals after the start. Very Sparse
	- Matrix Representation with Irregular Time Intervals: Keep only the nonzero rows of the above, and record the timestamps alongside it. 

### Model Training
 - ML: Logistic Regression, LR with L1 penalty (to promote sparsity and pick out the best/most informative variables), LR with L2 penalty (to improve numerical stability in parameter estimation), Random Forest, SVM, Gradient Boosting Decision Tree, Multi Layer Perceptron
 - DL: Composite architecture involving the embedding layers, followed by CNN, LSTM, or GRU, then a dense layer.
 - No mention of stratification for cross-validation or correcting class imbalance
 
### Results
 - Presented:
 	- Summary statistics of the patients (mean, std, min, max of each variable)
 	- A table coefficients of notable (knowledge-driven) features under LR and LR_L1
 	- ROCAUC bar charts for different feature encodings per ML and DL model
 	- Comparison table of ROCAUC score per ML model for different time window lengths
 - Conclusions:
 	- ML:
 		- Coefficient table indicates old age, long LOS, and many admissions increase risk of inadmission (all obvious conclusions)
 		- Grouped ICD-10 improve ML performance (potentially due to lower dimensionality)
 		- Pharmacy feature is found to not be very informative
		- GBDT has the best performance generally
	- DL: 
		- Coarse time grandularity improves performance (I assume it reduces overfitting)
		- Irregular matrix with attention is not necessarily better than matrix with regular time and no attention.
		- All embedding strategies perform similarly
	- Overall:
		- Asserts the interpretability, generalisability, and performance of HOSPITAL and LACE but combining with data-driven features gives the best performance.
		- History beyond one year doesn't make a great impact
		- Best DL was on par with best ML
		- Incorporating domain knowledge into the model building is vital
		- A complicated model may not necessarily improve anything

## Review: ML for Predicting Readmissions - Huang
 - Recent reviews have demonstrated that ML can be applied to predict various outcomes including disease diagnosis, prognosis, therapeutic outcomes, mortality, and readmission. 
 - About 25% of identified studies use single-hospital EMR datasets
 - All studies considered demographic characteristics and primary diagnosis and/or comorbidity index. More than half considered SDH and illness severity. Some considered mental health and a few considered overall health/functional status. 
 - Only 16% reported addressing imbalanced data
 - 65% had AUC over 0.7

## Comparing ML Model with Existing EWS - Levin
2. Data Preprocessing
 	- Features: Demographics, Arrival mode (walk-in or ambulatory), vital signs, primary chief complaint, and active medical history. 
 	- Outcomes: Predicts critical care, emergency procedure, and inpatient admission in parallel
 		- "Critical care" outcome is compositely defined as ICU admission or hospital mortality
 		- Emergency procedures are surgeries that occur within 12 hours of arrival
 3. Model Training:
 	- Ensemble: Three distinct DTs are trained to predict critical care, procedure, and hospitalisation outcome. 
    - They are then applied in parallel to generate outcome probabilities mapped to a single tirage level. 
 4. Model Evaluation:
 	- To compare with ESI, calibrated the final model to proportionally distribute patients similarly to the ESI group (per hospital).
        - However, can re-calibrate model to meet targets or objectives. For example, reducing number of patients traiged to the "ambiguous" peak of bellcurve by shifting more to lower risk scores. 
 	- Low-scoring patients (scoring 4 and 5) were grouped into group 4&5. 
 	- Reported differences of model and ESI according to:
 		- Agreement: model and ESI produced equivalent triage decision
 		- Up-triage: Model determines higher risk than ESI
 		- Down-triage: Model determines lower risk than ESI
 5. Future:
 	- Calibration according to ESI is good for comparisons, but can re-calibrate

## References
[Huang21] [Application of machine learning in predicting hospital readmissions: a scoping review of the literature](https://doi.org/10.1186/s12874-021-01284-z)

[Levin21] [Machine-Learning-Based Electronic Triage More Accurately Differentiates Patients With Respect to Clinical Outcomes Compared With the Emergency Severity Index](https://doi.org/10.1016/j.annemergmed.2017.08.005)

[Min19] [Predictive Modeling of the Hospital Readmission Risk from Patients’ Claims Data Using Machine Learning: A Case Study on COPD](https://doi.org/10.1038/s41598-019-39071-y)

[Stenwig22] [Comparative analysis of explainable machine learning prediction models for hospital mortality](https://doi.org/10.1186/s12874-022-01540-w)