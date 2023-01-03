# Predicting Acute Clinical Deterioration with Interpretable Machine Learning

This is the implementation for the paper titled: *Predicting Acute Clinical Deterioration with Interpretable Machine Learning to support Emergency Care Decision Making*.

[Paper Link (Manuscript Under Review)](https://doi.org/10.21203/rs.3.rs-2361002/v1)

## Table of Contents
------

1. [Introduction](#introduction)
2. [Installation instructions](#installation-instructions)
3. [Repo contents](#repo-contents)
4. [References](#references)

## Introduction
This repository contains the source code for the linked paper, including all data pre-processing, model training and tuning, systematic comparisons of model variations, and evaluation. The Jupyter notebooks at the root directory document the experiments we ran to demonstrate the models' performance and accuracy on the proprietary dataset.

We apply state-of-the-art machine learning methods to predict patient deterioration early, based on their first recorded vital signs, observations, laboratory results, and other predictors documented in Electronic Patient Records. We build on prior work by incorporating interpretable machine learning and fairness-aware modelling, and achieve improved classification performance over the current standard, the National Early Warning Score 2 [1], measured by average precision and daily alert rate. 

We use a cross-sectional Electronic Patient Record dataset comprising 121,058 unplanned admissions at Salford Royal Hospital, UK, to systematically compare model variations for predicting mortality and critical care utilisation within 24 hours of admission. We use Shapely Additive exPlanations [2] to justify the models' outputs, verify that the captured data associations align with domain knowledge, and pair predictions with the causal context of each patient’s most influential characteristics. 

## Installation instructions
------
*Note: Executing this code requires access to the proprietary Salford Royal Hospital dataset.*
1. (Optionally) create a virtual environment
```
python3 -m venv acpenv
source acpenv/bin/activate
```
2. Clone into directory
```
git clone https://github.com/stelioslogothetis/acute-care-pathways.git
cd acute-care-pathways
```
3. Install requirements via pip
```
pip install -r requirements.txt
```

### Requirements:

 - scikit-Learn >= 1.1.2
 - [SHAP](https://github.com/slundberg/shap) >= 0.41.0
 - [Optuna](https://github.com/optuna/optuna) >= 3.0.2
 - [Shapely](https://github.com/shapely/shapely) >= 1.8.4
 - [imbalanced-Learn](https://github.com/scikit-learn-contrib/imbalanced-learn) >= 0.70.0
 - Openpyxl

## Repo contents
------
The notebook `comparison_results.ipynb` documents the results of the systematic model comparison experiments. 

The notebooks `model_selection.ipynb`, `model_evaluation.ipynb`, and `clinical_models.ipynb` demonstrate the model testing for the project. `baseline_model.ipynb` documents measuring the performance of the reference model, NEWS2, on various outcomes. 

`data_profiling.ipynb` and `sci_preprocessing.ipynb` document the data pre-processing stage.

Supporting code:
 - `dataset.py`: Classes for interacting with the proprietary dataset used for testing.
 - `hyperparameter_tuning.py`: The source code for hyperparameter tuning of different types of classifiers. 
 - `initial_preprocessing.py`: Script to execute all the manual data cleaning steps. 
 - `models.py`: Classes for interacting with classifier types.
 - `systematic_comparison.py`: The source code for the systematic model comparison experiments.

## References
------
[1] Royal College of Physicians. National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS (RCP, 2017). https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2

[2] Lundberg, S. M. et al. Explainable machine-learning predictions for the prevention of hypoxaemia during surgery. Nat.
Biomed. Eng. 2, 749 (2018). https://doi.org/10.1038/s41551-018-0304-0
