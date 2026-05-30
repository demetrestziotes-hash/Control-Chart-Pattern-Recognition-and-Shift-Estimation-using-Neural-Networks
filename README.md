### Control Chart Pattern Recognition and Shift Estimation using Neural Networks

**Description**
This R script implements a dual-module Artificial Neural Network (ANN) system for Statistical Process Control (SPC). It automatically detects unnatural control chart patterns (Trend, Shift, Cyclic) and subsequently estimates the exact magnitude of detected process shifts.

This workflow utilizes the `neuralnet` package to build a robust pattern recognition and parameter estimation pipeline for time-series process monitoring. The script simulates a sliding window of data streams containing normal variance and three unnatural process disturbances: 
* Upward Trends
* Downward Shifts
* Cyclic Patterns

**System Architecture:**
* **Module 1 (BPPR):** Acts as a Back-Propagation Pattern Recognizer that classifies the incoming data stream. It is evaluated using a Monte Carlo simulation across various hidden layer configurations, shift sizes, and probability cutoffs, calculating performance metrics like the Rate of Target (ROT) and Average Target Pattern Run Length (ATPRL). 
* **Module 2 (Estimator):** If Module 1 detects a process shift, Module 2 is triggered to analyze the window and predict the continuous magnitude of the shift. 

The script evaluates the estimator's accuracy by calculating the Root Mean Square Error (RMSE), Standard Error (SE), and a 95% confidence interval for the predicted shifts.
