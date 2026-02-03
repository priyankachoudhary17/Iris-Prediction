# Iris-Prediction

Project Overview

This project implements an Artificial Neural Network (ANN) to classify iris flowers into three species based on physical measurements of their petals and sepals.

The three flower species are:

Setosa

Versicolor

Virginica

Using supervised learning, the model learns patterns from labeled data and predicts the correct species for new samples.

This project demonstrates the full machine learning workflow:

Data loading

Exploration and preprocessing

Feature scaling

Neural network construction

Training and validation

Model evaluation

Saving the trained model

📂 Dataset Description

The Iris dataset is a classic benchmark dataset in machine learning.

Each record contains:

Sepal length

Sepal width

Petal length

Petal width

The target column represents the flower species.

This is a small and clean dataset, which makes it perfect for learning and experimentation.

⚙️ Technologies Used

Python

NumPy, Pandas

Scikit-learn

TensorFlow / Keras

Matplotlib, Seaborn

Jupyter Notebook

🪜 Step-by-Step Methodology
✅ Step 1: Importing Libraries

All necessary Python libraries are imported for:

Numerical computation

Data handling

Visualization

Data preprocessing

Model training

Performance evaluation

This keeps the workflow organized and reproducible.

✅ Step 2: Loading the Dataset

The dataset is loaded into a Pandas DataFrame.

Purpose:

Converts raw data into a structured table

Makes inspection and manipulation easy

✅ Step 3: Exploratory Data Analysis (EDA)

Basic exploration includes:

Dataset shape

Column names and types

Summary statistics

Checking for missing values

Why EDA matters:

Helps understand feature distributions

Confirms data quality

Identifies possible anomalies

Builds intuition before modeling

✅ Step 4: Feature and Target Separation

The dataset is split into:

X: Input features

y: Target labels

Why:

Neural networks require a clear distinction between predictors and outputs.

✅ Step 5: Train–Test Split

The data is divided into training and testing sets.

A typical split is:

80 percent for training

20 percent for testing

Reason:

Ensures unbiased evaluation

Tests how well the model generalizes to unseen data

✅ Step 6: Feature Scaling

All features are standardized using StandardScaler.

This rescales values so they have:

Mean equal to 0

Standard deviation equal to 1

Why scaling is crucial for ANN:

Gradient-based optimization converges faster

Prevents large-scale features from dominating smaller ones

Leads to more stable training

The scaler is fitted only on training data and applied to the test set to avoid data leakage.

✅ Step 7: Neural Network Architecture

The ANN model is built using a Sequential structure with:

An input layer connected to hidden layers

One or more hidden layers with ReLU activation

An output layer with Softmax activation for multi-class classification

Architecture reasoning:

ReLU introduces non-linearity

Multiple hidden layers allow learning complex patterns

Softmax outputs class probabilities that sum to one

✅ Step 8: Model Compilation

The model is compiled with:

Optimizer: Adam

Loss Function: Categorical or Sparse Categorical Crossentropy

Metric: Accuracy

Why these choices:

Adam adapts learning rates automatically

Crossentropy loss suits multi-class classification

Accuracy provides a clear performance measure

✅ Step 9: Model Training

The ANN is trained for several epochs using mini-batch gradient descent.

Validation data is used during training to monitor performance on unseen samples.

Why:

Training data teaches the model

Validation helps detect overfitting

Learning curves show convergence behavior

✅ Step 10: Visualizing Training History

Plots of training and validation accuracy and loss are created.

Purpose:

Track learning progress

Identify underfitting or overfitting

Decide whether more tuning is required

✅ Step 11: Model Evaluation

The trained model is evaluated on the test dataset.

Metrics analyzed:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Why:

Gives a detailed view of classification performance

Highlights types of mistakes made by the model

✅ Step 12: Saving the Model

The final trained ANN model is saved to disk.

Reason:

Allows reuse without retraining

Supports deployment

Makes the project reproducible
