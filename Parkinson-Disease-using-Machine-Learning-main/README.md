This project provides a Machine Learning solution for the early detection of Parkinson's Disease using voice data. The repository contains both the ML code and the dataset required to train and test the models.

Files Included
Parkinson_disease_using_machine_learning.ipynb — The main Jupyter Notebook containing the complete ML pipeline.

parkinson.data — The dataset containing vocal features of individuals with and without Parkinson’s Disease.

How to Run
Open Google Colab.

Upload both parkinsons_detection.ipynb and parkinsons_dataset.csv to your Colab environment.

Open the notebook and run all cells sequentially.

The models will train and you will see the performance metrics for each algorithm.

Project Overview
Various supervised learning algorithms are implemented:

Logistic Regression

Decision Tree

Random Forest (Gini and Entropy)

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Gaussian and Bernoulli Naive Bayes

Voting Classifier (Ensemble)

Performance metrics like Accuracy, Precision, Recall, and F1-score are used for evaluation.

Random Forest (Entropy) and Voting Classifier demonstrated the best results.

Requirements
Python 3.x

Libraries: pandas, numpy, sklearn, matplotlib (all pre-installed in Colab)

Results
By following the above steps, you will obtain model evaluation metrics and see which algorithm performs best for detecting Parkinson’s Disease using vocal data.

