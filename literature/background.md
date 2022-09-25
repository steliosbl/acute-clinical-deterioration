# Background

## ML in Healthcare
 - The first commercial AI system using EHR data to receive clearance for widespread clinical use by the FDA was WAVE [Parikh19]. 
 - Standard outcomes EWSs are used to track are mortality and ICU admission [Gerry20, pp. 1], while prior literature has used used ML to stratify patients according to their risk of mortality, admission to hospital or to the ICU, acute morbidity and infectious diseases, expected resource use, etc [Fernandes20]. 
 - Historically paper-based points-based EWS make the assumption that each vital sign considered has the same predictive value [per Churpek12, from Gerry20, pp. 10].
 
## Electronic records and EWS
 - EWSs have historically been designed for paper charts, and were often expected to be calculated by hand. This includes the points-based NEWS [RCP17]. However, they are increasingly being integrated into EHR systems [Gerry20, pp. 1]. 
 - The NEWS was designed for paper charts and calculation by hand. The latest revision only considers gradual replacement of paper-based observation charts with electronic data systems in some hospitals [RCP17, pp. 33-35].

## NEWS
 - Rationale comes from a 2007 NICE report that recommended the routine measurement of six physiological parameters to assess illness severity [RCP17]. 
 - Use of EWS in the NHS is mandated by NICE guidelines [CG50]. 
 - Determining the risk of mortality, unanticipated ICU admission, cardiac arrest, or a composite of the previous is common in EWS literature [Gerry20, pp.2]. The NEWS study [Smith13] concluded that cardiac arrest may have been unnecessary as they found it to be indistinguishable from death in most cases, the only difference being whether a cardiac arrest team had been called in or not [Smith13, pp. 4]. 

## Clinical Relevance:
 - Recent reviews have found little evidence of clinical effectiveness of EWSs [Gerry20, pp. 9]. But this may be because many are developed using inadequate methods, with false reassurances about their predictive ability and generalisability. 
 - Variation in interpretability of ML methods impedes clinical buy-in [Mahmoudi].

## Balancing Sensitivity and Specificity 
 - In ML, classification algorithms typically produce a numerical score for each input, (e.g., a probability 0.0-1.0) indicating the algorithm's confidence that the input belongs in the positive class. The final class label is then applied based on whether the score is over a certain threshold (e.g., 0.5) and, while this has no bearing on the operating characteristics of the model, the choice of threshold directly influences model's the sensitivity and specificity. Tuning the decision threshold is therefore recommended, especially in domains with imbalanced classes [Zou16]. 
 - Many developers of clinical risk models simply select a threshold that maximises the sum of TPR and TNR. However, this is only valid if we weigh sensitivity and specificity equally [Perkins06]. 
 - In the clinical context, the choice of decision threshold must include both clinical and operational considerations. It is vital to direct care where it is needed in a timely manner, and this outweighs the cost of false-positives, which entail monitoring certain patients more than necessay [Calster18]. At the same time, however, excessive false alarms are detrimental to the model's utility due to the effects of alert fatigue, including poor alert response and a general lack of confidence in the system [Bedoya19, Ancker17]. 

### Alert Fatigue
 - Alert fatigue, the desensitisation to safety alerts due to workload and alert frequency, has been studied in relation to operationalising EWS in clinical settings.
 - Research has linked poor alert response and alert fatigue with the frequency of repeated alerts a system produces, the prevalence of false alarms, as well as increased cognitive workload and complexity of work [Ancker17, pp.6]. 
 - One prior study considered to hospitals where an EWS (specifically, the NEWS) was integrated into EHR and generated automated alerts (Best Practice Advisories, BPAs) in real time to alert clinical staff that a patient reached the predetermined high-risk threshold [Bedoya19]. The poor operating characteristics of the system at those care centers manifested as a high frequency of false positives. These persistent false alarms that required no clinical intervention created *alert fatigue* in the frontline nursing staff and caused "a general mistrust of the NEWS workflow". 
 - Another retrospective analysis of a NEWS implementation found that 85% of alerts were ignored by clinical staff [Obrien20, pp. 8], 
 - A study utilising a NEWS derivative found frontline staff did not notify clinicians about elevated patient scores beneath the critical threshold as the high prevalence made it disruptive to workflow [Petersen16].

## Model Selection
 - Logistic regression is highly prevalent in EWS development studies [Gerry20, pp. 6]. 
 - Rather than limit the input data to features we considered important, we use modelling algorithms with the capability to choose the features they find informative (as in [Lundberg18]). This carries the possibility the model will assign weight to unexpected features. 
    - Interpretability functionality allows us to investigate these importances
    - For some features it may be helpful, as in [Lundberg18], to tag them with brief indicators of their relevance to the patient's risk, as some connections may be non-obvious.

## Missing Data Handling:
 - Imputing missing values is preferable to ignoring them, as data are usually not missing at random, and thus excluding a record due to incompleteness can result in serious bias [Janssen10].

## Interpretable AI
 - While the primary goal in predictive modelling tasks is to maximise predictive power, interpretability of machine learning models, which are often treated as black-boxes, is a growing concern.
 - One concern whenever a clinical risk model is proposed is whether it aligns with domain knowledge about the relationships contained in data, as learning spurious or biased relationships is an ever present risk [Fu20, pp.13]. 
 - Lack of understanding among clinical staff and patients about how AI systems make predictions has stood as an obstacle to wide clinical adoption of machine learning technologies [Watson19].
 - However, interpretable modelling techniques can, alongside predictions, explain why a certain prediction was made for a given patient by isolating the patient characteristics that contributed to it [Lundberg18, pp.3].
 - Interpretability has been defined in the ML literature as the ability to explain a system's reasoning in understandable terms to a human [Velez17]. Explainable AI encompasses multiple lines of work, initiatives, and efforts to respond to growing AI transparency and trust concerns [Adadi18]. 
 - Global model interpretability reveals the general patterns behind a model's decision making i.e. it offers us an understanding of the interactions between all the variables and their relative importance in generating the final predictions. Conversely, local interpretability justifies the reasoning behind a specific output based on the corresponding input values [Velez17, pp.7]. 
    - In other words, we may pursue the goals of undersanding all the patterns the model has learned from its training data, or just those that are relevant to the individual patient it produced a prediction for. 
    - Applications of the former include validating the quality of the model's fit and whether it aligns with domain knowledge, as well as shedding new light on complicated problems, in tasks where AI has outperformed human experts. [Velez17].
    - The latter can contribute to building confidence in the system or, in cases where mistakes carry severe consequences, help expert users to validate the predictions. In the clinical setting, the factors that contributed to a certain prediction might be valuable for the patient's prognosis and further assessment.
    - Further, ethical considerations demand we be able to investigate biases that weren't identified a priori but emerge over time, and that algorithms that produce decisions based on individual user-level predictors be transparent to the individuals who are affected by them (though the degree of transparency that is mandatory has been the subject of debate in legal circles [Wachter17]).  
 - Certain modelling algorithms are designed to be intrinsically interpretable [citation needed], while other approaches achieve interpretability by placing constraints on the complexity of the machine learning model [Caruana15]. However, academic interest has surged in developing post-hoc interpretability techniques that are model-agnostic, and can operate on any black-box model [Adadi18]. 
    - One approach is to train a highly accurate but opaque conventional ML model that will be relied on for predictions, and later fit a white-box surrogate model that provides interpretations of these predictions [Burkart21]. A notable implementation of this approach is SHAP. 

### SHAP
 - A recently popularised technique for obtaining local explanations is SHAP (SHapely Additive exPlanations) [Lundberg18]. It belongs to a class of methods that deduce the influence of each feature on a modal by varying its value and measuring how much the change affects the model's output. 
    - In its general, model-agnostic form, SHAP operates on any trained model and takes a dataset record of interest as input. It frames the task of generating local explanations similarly to the problem of distributing payout proportionally among players in coalitional game theory. SHAP assumes that the model's output for a given input is a sum of the individual feature contributions added to the model's baseline response, and represents these contributions using Shapely values [Strumbelj14]. The feature contributions are determined by marginalising over each feature and obseriving the model's behaviour in its absence [Burkart21, pp.35].
 - The general SHAP algorithm estimates the Shapeley values of the features using a weighted local linear regression technique. In addition to this model-agnostic form, more efficient estimation techniques have been developed specifically for linear and tree-based models, including the tree ensemble models we investigate [Lundberg20].
 - In their notable clinical application of SHAP [Lundberg18], the authors temper their claims, stating that the produced feature importances do not imply a causal relationship and thus do not represent a complete diagnosis [Lundberg18, pp.7]. However, by presenting risk predictions as a cumulative effect of patient features, the model's justification becomes more clear. 
 - At their core, the computed feature impact values represent the change in the model's output when a certain patient feature is observed, versus when it is not observed. 

### Implementation Considerations
Task-specific factors dictate the interpretability technique used, and the type of explanations offered to users.
 - Time Constraints: "How long can the user afford to spend to understand the explanation?". For example, decisions that need to be made during admission to a busy ED must be understood quickly [Velez17, pp.8]. 
    - Models that decide which features they consider important may learn non-obvious relationships. As such, for some features it may be helpful, as in [Lundberg18], to tag them with brief indicators of their relevance to the patient's risk, as some connections may be non-obvious and we have the time to investigate them while clinical staff on the floor does not. 
 - User Expertise: "How experienced is the user in the task?". Users experience determines their background knowledge and communication style, as well as the level of detail and sophistication they expect in explanations [Velez17, pp.8]. 
 - An explanation should/can be composed of basic units (cognitive chunks) which can be raw features or derived features, and should have carry semantic meaning to the domain expert user [Velez17, pp.8]. These may involve a degree of compositionality, where we provide explanations in terms of a derived unit is a function of several raw ones. We do this to limit how much a user needs to process at one time.  
 - It is recommended to strike a balance between predictive power and interpretability when designing new EWSs [Fu18, pp.13]. This is crucial when the chosen approach forces a tradeoff between accuracy and interpretability.

### Why make models interpretable?
 - Well-reasoned scenarios for when ML interpretability should be a requirement have been put forward in the literature. Examples include any complex application where the system cannot be completely tested end-to-end and as such we cannot exhaustively consider every scenario where it may fail. 
    - In a clinical setting, there is a need for a model to produce explanations to make sure its decision wasn't made erroneously, particularly when the decision is unexpected [Adadi18, pp.5].
    - When unacceptable results carry significant consequences, interpretability is critical [Velez17, pp.3]. 
 - Further, ethical considerations demand we be able to investigate biases that weren't identified a priori but emerge over time, and that algorithms that produce decisions based on individual user-level predictors be transparent to the individuals who are affected by them (though the degree of transparency that is mandatory has been the subject of debate in legal circles [Wachter17]).  

## Clinical Relevance
 - Medical meaningfulness of any automated system or model is essential if the goal is to affect clinicians' behaviour or gain their trust and cooperation [Lezzoni13, pp. 29].
 - To what degree a model is medically meaningful depends on the risk factors it takes into account. Recording all potential risk factors is, of course, infeasible. 
 - "Better prediction can improve quality of clinical care when tied to an intervention." [Parikh19]. 

# Useful points I've come up with (introduction or discussion)
 - The ED is responsible for the initial intake and management of patients of varied acuity, including ones in critical condition ([per Mohr20, from Kao21]).

## Limitations
 - Neither the NEWS nor our model is intended or able to replace clinical expertise, experience, and the prognostic systems and workflows already in use. In clinical practice, a model such as ours could complement the existing patient flow and help to quickly bring relevant information to the attention of the receiving team.
    - There are risk factors that are decisive in determining the patient's likely progression but are necessarily excluded from this model, and must be taken into account by the users [Lezzoni13, pp.29]. 

 - "The large dataset permits us to develop and test the model empirically, but the most statistically and conceptually robust models generally result from interaction between statistical modelling and clinicians [Lezzoni13, pp.27]."
    - Observational analyses and validation can give us evidence of the value of predictions, but the gold standard to assess any algorithmic solution is experimental data from randomised controlled trials [Parikh19]. 

 - Modelling and empirical assessment are vulnerable to misleading or spurious patterns present in the data. One frequently cited example involves an early application of machine learning to model risk of death which incorrectly learned that patients presenting with pneumonia that had a history of asthma had low risk. The source of the pattern was that the critical care provided to these patients as soon as possible lowered their risk of dying fromn the pneumonia compared to the general population [Caruana15]. 

 - Use of coded administrative data: We derive some clinical insight from diagnoses coded with indeterminate accuracy and completeness [Lezzoni13, pp.96 - further citation needed as that one is too old to consider ICD-10]. Further, in the current clinical setting the practice is to apply them retrospectively (?) and not in real time. As mentioned, however, having diagnoses on-record that meet rigorous standards may not be essential for modelling purposes.  

 - Training: The RCP states that the NEWS only works if the staff taking the measurements are trained in its use, and adequate response systems/teams are in place to deliver the recommended interventions [RCP17, pp. 32].

 - Breakdown into mortality and critical care: Death constitutes a competing risk when predicting ICU admission only [Wolbers09].

## Justification for a more dynamic EWS than the NEWS
 - The simplicity of the NEWS permits (and is intended to act as) a one-size-fits-all approach. However, "The population is remarkably diverse, and its many dimensions, alone or in combination, may help delinate subpopulations that have different risks for various health-related outcomes" [Lezzoni13, pp.19]. 

## Features and Justifications
 - Vitals
    - The most frequently included predictors in a systematic review of 34 EWS development studies [Gerry20] were heart rate, O2 saturation, temperature, systolic blood pressure, and consciousness. 
    - Prior work has shown (repeatedly) that clinical instability and deterioration leading to arrest, ICU admission, or death is, in most cases, preceded by abnormal vital signs [Kause04]. 
    - As such, it has become standard practice in acute secondary care hospitals to gauge patients' acute clinical stability and physiological condition using basic homeostatic measures such as cardiorespiratory parameters, consciousness level, fever, and pain scales. 
 
 - Age: 
    - Older patients have been shown to have different risks of deterioration and mortality than younger patients, as well as different disease presentation and progressions [per Loeches19, from Kao21]. For example, older pneumonia patients are noted to be less likely to report non-respiratory symptoms [Metlay97]. 
    - Further, in critical care settings, advanced age is independently predictive of the patient's death regardless of the severity of their acute condition, so separating patients into younger and older strata may offer insight [Lezzoni13, pp. 37].
    - The NEWSDIG elected not to include age in the NEWS as they considered it unnecessary to "apply an arbitrary weighting to the NEWS aggregate score on the basis of age, based on current evidence" [RCP17, pp.42]. 
evidence. 
 - Sex:
    - Males and females differ physiologically and anatomically, and their level of risk differs depending on condition and age. In past modelling attempts, sex has been only modestly predictive of short-term clinical risk, and many models, including the NEWS, exclude it entirely. 
        - At the same time, differing socioeconomic circumstances between men and women can influence the incidence of diseases and their reaction to treatments, as well as the treatment individauls face within the healthcare system [Geensway01].
        - This distinction becomes even more complex when considering gender identity rather than chromosomal sex.  
 - Diagnoses:
    - In some cases, finding and recording an accurate diagnosis may be infeasible or even impossible in the ED context, which carries time constraints and operational pressures. Acutely managing patients who may be in critical condition takes priority over providing precise diagnoses. 
        - In such cases, the "diagnosis" on-record is often a statement of symptoms indicating underlying disease, rather than definitively identifying the disease itself. 
        - When developing a predictive model for clinical use, it may not be necessary to have definitive diagnoses for each patient as, even if helpful for the model's validity, it is not reflective of the realities of clinical practice [Lezzoni13, pp. 50].
        - Prior studies have used routinely collected administrative data as predictors of mortality risk with some success [Aylin07]. 
    - Patients' comorbidities were not included in the NEWS as it was intended to be a generic EWS, and replace specialised, disease-specific risk scores. For rapidly assessing patients, it was anticipated to reflect the physiological perturbations they cause regardless [RCP17, pp.19]. 
 - Social Determinants of Health (SDH):
    - E.g. community context, economic stability, education, healthcare access have been found to be strong predictors of readmission [Mahmoudi20, pp. 4]. 
    - Despite significant links between social factors and risk, health systems still do not systematically collect this data [Mahmoudi20, pp. 6-7].
    - Several studies use Census block- or zip-level aggregate data as a routinely available proxy, but these are often too coarse to be useful.
    - NLP on clinician's notes has shown promise.
