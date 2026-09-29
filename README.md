#Bias-Correction for Privacy-Protected Spatial Autoregressive Models with  Application to Restaurant Network Analysis

This repository provides the Python code for the Corrected Likelihood Estimator (CLE) and the Corrected Least Squares Estimator (CLS) for spatial autoregressive (SAR) models with noise-added privacy protection, accompanying the paper.  

>Huang, D., Kong, Z., Wu, S., and Wang, H. (2024). Privacy-Protected Spatial Autoregressive Model. arXiv preprint arXiv:2403.16773 

## Overview
The classical SAR model relies on faithfully observed network data to estimate network dependence. However, to meet modern data privacy requirements, data providers inject artificially generated random noise into the response and covariate variables. This repository implements two estimation methods designed to correct the biases introduced by this noise:  Corrected Likelihood Estimator (CLE) and  Corrected Least Squares Estimator (CLS).

## Repository Structure

|File name| Description |
|-------------|---------------|
|**`CLE.py`**| Contains the functions for the iterative estimation and statistical inference of the CLE method. |
|**`CLS.py`**| Contains the functions for the iterative estimation and statistical inference of the CLS method. |
|**`generation.py`**| Handles the simulation data generation. |
|**`main.py`**| The main script to run Monte Carlo simulations. It executes the estimators under outputs performance metrics including Bias, Coverage Probability (CP), Mean Squared Error (MSE), and Standard Errors (SE). |
|**`time.py`**| A comparative script to measure the running time of the CLE and CLS methods across different sample sizes. |