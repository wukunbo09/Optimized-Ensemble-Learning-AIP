# Optimized-Ensemble-Learning-AIP
Optimized Ensemble Learning with Multi-Feature Fusion for Enhanced Anti-Inflammatory Peptide Prediction
## Introduction
* Anti-inflammatory peptides (AIPs) represent promising therapeutic agents, however, their discovery via traditional experimental methods is constrained by low throughput and high costs. Current computational prediction methods face persistent challenges, including dataset imbalance, inadequate feature integration, and limited predictive accuracy. To address these limitations, we assembled a high-quality benchmark dataset by integrating sequences from PreAIP, AIPpred, and IF-AIP. After evaluating feature selection methods including LASSO, PCA, Autoencoder and RF, we finally selected the 377-dimensional features ranked by RF. For classification algorithms, we implemented three deep learning architectures (LSTM, CNN, DNN) and five traditional machine learning classifiers (XGBoost, RF, AdaBoost, GBDT, LightGBM). Comparative analysis demonstrated that a soft voting strategy integrating predictions from the five ensemble classifiers achieved superior performance, surpassing state-of-the-art predictors. Further sequence composition analysis revealed significant enrichment of positively charged residues (e.g., Lysine, Arginine) in AIPs, whereas non-AIP sequences were predominantly characterized by hydrophobic residues. This finding provides a molecular basis for the rational design of novel anti-inflammatory peptides.
## Environment
* python 3.9.7  
*  biopython 1.85  
*  numpy 1.23.0  
*  pandas 1.5.3  
*  scikit-learn 1.2.0  
*  scipy 1.9.3  
*  torch 2.6.0   
## Usage  
* We provide model.py under the prediction folder. Just modify the file address in the code to the input data you are interested in.
