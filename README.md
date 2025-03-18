# EvolutionaryComputing
-Ass1
Neural Network for Salary Prediction

Overview

This project implements a single-layer perceptron in C# to predict salary based on multiple input features. The model is trained using gradient descent and evaluated using Mean Squared Error (MSE) and Sum Squared Error (SSE).

Features

Reads and normalizes data from a CSV file (SalData.csv).

Uses an 80/20 split for training and validation.

Implements gradient descent to train the perceptron.

Evaluates model performance using MSE and SSE.

Supports multiple learning rates for optimization.

Saves training performance metrics to results.csv.

Outputs predicted vs actual salary values.

Code Structure

1. Program.cs

Loads and normalizes the dataset.

Splits data into training and validation sets.

Trains the perceptron with different learning rates.

Saves evaluation results to results.csv.

Displays predicted vs actual salaries.

2. SalaryList.cs

Handles data loading, normalization, and denormalization.

Reads salary data from SalData.csv.

Applies min-max normalization for feature scaling.

3. Individual.cs

Represents a single data record with features:

Salary

Education

NrSupervise

NrPositionHeld

PercResponsibility

NumChildren

Age

YearsExperience

4. Perceptron.cs

Implements a single-layer perceptron for salary prediction.

Uses gradient descent to update weights and bias.

Prediction formula:

y = \sum_{i=1}^{n} (w_i \times x_i) + b

Model evaluation:

EvaluateMSE(): Computes Mean Squared Error.

EvaluateSSE(): Computes Sum Squared Error.


-Ass 2
The neural network has three main layers. The input layer receives the 27 features from
the dataset. The hidden Layer consists of 128 neurons. This layer processes the inputs
from the previous layer, applying a non-linear transformation(ReLU) to capture complex
patterns in the data. The Output Layer, contains 7 neurons, corresponding to the 7 fault
categories in the dataset. This layer uses the softmax activation to produce probabilities
for each class. The network uses a fully connected architecture, where each neuron in one
layer is connected to every neuron in the subsequent layer. W1 and W2 represent the
weight matrices connecting the input layer to the hidden layer and the hidden layer to the
output layer, respectively.

-Ass 3
Genetic Algorithm for Salary Prediction
This project uses a Genetic Algorithm (GA) to predict salaries based on input features. The model is trained on a dataset (SalData.csv), where salary is the target variable, and various features are used as predictors.

Features
Data Preprocessing: Reads CSV data, normalizes features using Z-score normalization, and splits into an 80% training and 20% test set.
Genetic Algorithm Implementation:
Three models with different parameter sizes (7, 8, and 15)
Uses Tournament Selection, Crossover, and Mutation
Two Crossover Rates tested: 0.85 and 0.95
L2 Regularization for fitness evaluation
Model Training: The GA evolves weights over 500 generations to minimize Sum of Squared Errors (SSE).
Performance Evaluation:
SSE vs. Generations plots for all models
Test set evaluation to determine the best crossover rate
Comparison of predicted vs. actual salary values
Results
The script identifies the best crossover rate for each model and selects the optimal parameters based on the lowest SSE.


