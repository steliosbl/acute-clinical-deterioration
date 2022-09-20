# Background on Modelling Health Outcomes
## TL;DR
 - Survey works have conducted systematic comparisons of predictive models for unplanned readmission or mortality in hospitals [Mahmoudi20, Liu21].
    - Liu et al. qualitatively compare the clinical value and provide a "checklist" to follow.
 - Established, widely used Early Warning Scores are NEWS, LACE, and HOSPITAL [RCP17, Walraven10, Donze13].
 - Literature on *risk adjustment* provides guidance and justification on feature selection and handling [Lezzoni12]. 
 - Levin at al. focus on triage and systematically compare with existing model used in USA hospitals (ESI). 

## Motivation
 - **Outcomes**: Outcomes considered in Risk Adjustment are [Lezzoni12, pp. 16]:
   - Clinical: Death, complications, physical functional status, mental health
   - Resource Use: Length of stay
   
   These can be composited. For example, [Levin21] considers "Critical Care" outcome as ICU admission OR hospital mortality.
 - **Stratification**: The diversity of the overall population means that its many dimensions, alone or in combination, may help delinate subpopulations that have different risks for various health-related outcomes [Lezzoni12, pp. 19].
 - **Clinical Adoption**: Interpretable & clinically relevant models would encourage the identification of patients who need efficient allocation of limited available resources [Mahmoudi20, pp. 7].
 - **Triage**: Augmenting triage with ML can support improved patient differentiation compared with existing procedures.  Expedited ED care for patients destined for the ICU or OR are consistently associated with improved patient outcomes.

### Existing Approaches
The NHS uses the National Early Warning Score (NEWS). It is a weighted sum of:
 - Respiration Rate, Oxygen Saturation, Systolic BP, Pulse, Level of Consciousness, Temperature

The rationale came from a 2007 NICE report which recommended the routine measurement of these paramters in hospital, to assess illness severity [RCP17, pp. 15]. Some exluded factors are:
 - Age: NEWS Development Group determined it was unnecessary to apply *arbitrary* weighting to the score based on patient's age [RCP17, pp. 19].
 - Comorbidities: NEWS is generic, while many illnesses have specific scoring systems. It is meant to reflect the physiological perturbations they cause regardless.

Useful facts:
 - The NEWS was designed for paper charts and calculation by hand. The latest revision only considers gradual replacement of paper-based observation charts with electronic data systems in some hospitals [RCP17, pp. 33-35].

Other widely used data-driven models include:
 - HOSPITAL Score: Prediction model for potentially avoidable 30-day hospital readmissions [Donze13].
    - Hemoglobin level, Oncology, Sodium level, Procedure during spell, Index Type (elective or not), Admission count during the previous year, and Length of stay.
 - LACE Index: Points-based model of risk of mortality or 30-day readmission.
    - Length of stay, Acute admission (or elective), Charlson comorbidity index score, number of visits to Emergency department. 

## Features 
Per [Mahmoudi20] the top predictors overall are:
 - Previous inpatient visits within 3-6 months.
 - Clinical data (e.g. via constructed severity scores)

### Physiological Data
 - In risk adjustment, acute clinical stability reflects patients physiological condition as indicated by basic homeostatic measures (vital signs, hematologic findings, arterial O2, level of consciousness) to assess whether patients face an imminent risk of death [Lezzoni12, pp. 47]. 
 - The value of acute physiological parameters in assessing patient risk for imminent clinical outcomes is undisputed [Lezzoni12, pp. 50].
 - The original NEWS study asserts that single extreme variation in one parameter is very rare and, while not reflected in the Score, will be noticed by the clinician. It is designated "less serious" than an overall high score. 

### Diagnosis 
 Diagnosis is sometimes impossible to make [Lezzoni12, pp. 50].
 - The assigned "diagnosis" is a statement of symptoms reflecting a sign of underlying disease rather than the disease itself. 
 - In serious cases, the demands of acutely managing a cataclysmic event may divert attention from precise diagnosis. 
 - Diverse diseases may lead to similar problems, and at the same time the same disease may present in all manner of severities.

In a model, having a definitive diagnosis that meets rigorous standards may not be essential. Vagaries reflect the realities of contemporary clinical practice [Lezzoni12, pp. 51].

If administrative data is used (such as ICD10), the primary clinical insight comes from diagnoses coded with questionable accuracy, completeness, clinical scope, and meaningfulness [Lezzoni12, pp. 96]. This data also does not contain information about patients' preferences such as DNRs, or the prupose of hospitalisation (e.g. palliative care).

### Patient Characteristics
 - Age: Symptoms of disease may differ between older and younger persons. Age may have an effect on patient risk independent of other factors. For example, in ICUs, it independently predicts imminent death regardless of the extend of the disease [Lezzoni12, pp. 37]. 
    - The PRISM model (risk model for pediatrics) employs different ranges for systolic GP, heart rate, and respiratory rate depending on the age of the child.

 - Sex: Males and females face divergent risks for certain diseases and death by age strata. 
    - Beyond the physiological and anatomical differences, socioeconomic circumstances that influence women's lives differently from men's can differentiate the incidence and reaction to diseases and treatments, as well as the treatment individauls face within the healthcare system [Lezzoni12, pp. 40]. 
    - Distinction becomes even more complex when considering gender identity rather than chromosomal sex.

### Social Determinants of Health (SDH)
 - E.g. community context, economic stability, education, healthcare access have been found to be strong predictors of readmission [Mahmoudi20, pp. 4]. 
 - Despite significant links between social factors and risk, health systems still do not systematically collect this data [Mahmoudi20, pp. 6-7].
 - Several studies use Census block- or zip-level aggregate data as a routinely available proxy, but these are often too coarse to be useful.
 - NLP on clinician's notes has shown promise.

## High-Level Considerations 
Liu et al. asserts the following key *failing points* of many published models:
 - **Validation**: Inadequate validation means poor generalisation/discriminative value. 
 - **Features**: Over-realiance on biomedical features from administrative data.
 - **Timeframe**: The authors express skepticism about the standard 30-day timefrime being optimal for accurate prediction
 - **Data Access**: Dependence on inaccessible, low-quality, outdated, or manually enterred data.
    - Interoperability: Linking to primary/secondary care DBs for end-use.
    - Insufficient Data: Nobody knows what the optimal "neccesary" data is.
 - **Resources**: Insufficient statistical expertise or financial resources.
    - Prevents access to the EHR back-end to pull the best possible features or employing data collection staff.
 - **Vision**: Lack of purpose or policy priorities.
 - **Clinical Relevance**: Unclear clinical utility, actionability & relevance of outputs.
 - **Workflow Integration**: Poor integration into the clinical workflow.
    - Consider who will receive results, when, and how. Tie the integration into physical patient-risk-reducing interventions.
 - **Maintenance**: Inadequate maintenance/continuous improvement. 

Christodoulou et al. give the following guidelines on paper-writing:
 1. Fully report on all modelling steps: 
    - Sample size, number of outcome events in dataset and splits, overviw of all predictors considered in data-driven modelling, hyperparameter tuning, continuous variable handling, resampling.
 2. If resampling used for internal validation, report the model(s) performance on the entire dataset
 3. Report training and validation performance:
    - Usually the former is omitted because it is optimistic, however the difference is informative of how much overfitting is going on.
 4. Report on calibration of the risk predictions:
    - Calibration here means evaluating the reliability of probabilistic risk estimates.
    - Discrimination performance is insufficient, calibration informs on over- or underestimation of predicted risk.

I also consider:
 - **Training**: The RCP states that the NEWS only works if the staff taking the measurements are trained in its use, and adequate response systems/teams are in place to deliver the recommended interventions [RCP17, pp. 32].

### More On Clinical Relevance
The medical meaningfulness of a model is dictated by the risk factors it takes into account. The scope of risk factors determines whether the outputs are credible and valid [Lezzoni12. pp. 29].
 - Given that the goal is to affect clinicians' workflow and gain their cooperation, medical meaningfulness is essential.
 - Collection of information on all potential risk factors is logistically and practically infeasible
 - Important to consider the conceptually important but necesasrily excluded risk factors when interpreting the model outputs

Mahmoudi et al. also point out transparency as an essential quality. 
 - ML methods often use far too many features, leaving open important clinical questions about the implementation and interpretability of the results.
 - Variation in interpretability of ML methods impedes clinical buy-in. 

Smith et al anticipate that methods that use many parameters are more vulnerable to operational error over solutions that use fewer parameters. Operational pressures in the day-to-day workflow may hinder the reliability or timeliness of data.

Levin et al. state that some measured outcomes (e.g. ICU) are inherently variable by hospital (criterio for ICU admission may differ by hospital) so a single-hospital dataset inehrently adapts to the particular hospital. 
 - However, a model can be re-calibrated to instead target particular *objectives*, e.g. shifting the bellcurve away from ambiguous "midpoint" scores. 

Of course, outputs of model are simple indicators for high-severity medical need that span a broad range of conditions encountered by clinicians [Levin21]. 

## Model Evaluation
Large datasets permit us to develop and test models empirically. However, in general, the most statistically and conceptually robust models result from interaction between statistical modelling and clinicians [Lezzoni12, pp. 27].

To compare with existing score (ESI), [Levin21] calibrates model according to how ESI distributes triage patients. 

## References
[Christodoulou19] [A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models](https://doi.org/10.1016/j.jclinepi.2019.02.004)

[Donze13] [Potentially avoidable 30-day hospital readmissions in medical patients: derivation and validation of a prediction model](https://doi.org/10.1001/jamainternmed.2013.3023)

[Levin21] [Machine-Learning-Based Electronic Triage More Accurately Differentiates Patients With Respect to Clinical Outcomes Compared With the Emergency Severity Index](https://doi.org/10.1016/j.annemergmed.2017.08.005)

[Lezzoni12] [Risk adjustment for measuring health care outcomes](http://discover.durham.ac.uk/permalink/f/1tj5oqu/44DUR_LMS_DS.b35685797)

[Liu21] [Published models that predict hospital readmission: a critical appraisal](https://doi.org/10.1136/bmjopen-2020-044964)

[Mahmoudi20] [Use of electronic medical records in development and validation
of risk prediction models of hospital readmission: systematic
review](https://doi.org/10.1136/bmj.m958)

[Walraven10] [Derivation and validation of an index to predict early death
or unplanned readmission after discharge from hospital to
the community](https://doi.org/10.1503/cmaj.091117)
