# Project 2: Iris Classification Using KNN (DecodeLabs Internship)

## What is this?
This is my second project at DecodeLabs. It uses the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

## Dataset
- Name: Iris Dataset (built-in in sklearn)
- Samples: 150 flowers
- Features: Sepal Length, Sepal Width, Petal Length, Petal Width
- Classes: Setosa, Versicolor, Virginica

## How to Run
1. Install required libraries:
   pip install numpy pandas matplotlib seaborn scikit-learn
2. Run the file: IRRIS-KNN-classification-p2.py
3. The program will show accuracy, F1 score, and graphs

## Results
- Accuracy: ~96-100% (depends on random split)
- F1 Score: ~0.96-1.00
- Optimal K Value: 5 or 7 (auto-selected by elbow method)

## Output:


=======================================================
       DecodeLabs — Project 2: Iris Classification
=======================================================

Dataset Shape  : (150, 5)
 Classes        : [np.str_('setosa'), np.str_('versicolor'), np.str_('virginica')]
Features       : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

First 5 Rows 
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
2                4.7               3.2                1.3               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
4                5.0               3.6                1.4               0.2  setosa

Class Distribution 
species
setosa        50
versicolor    50
virginica     50
Name: count, dtype: int64

Basic Statistics
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000         150.000000        150.000000
mean            5.843333          3.057333           3.758000          1.199333
std             0.828066          0.435866           1.765298          0.762238
min             4.300000          2.000000           1.000000          0.100000
25%             5.100000          2.800000           1.600000          0.300000
50%             5.800000          3.000000           4.350000          1.300000
75%             6.400000          3.300000           5.100000          1.800000
max             7.900000          4.400000           6.900000          2.500000

Feature Scaling applied (StandardScaler: Mean=0, Variance=1)

Training samples : 120
Testing samples  : 30

Optimal K found : 2

=======================================================
            MODEL RESULTS
=======================================================
  Accuracy  : 100.00%
  F1 Score  : 1.0000

 Classification Report 
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Confusion Matrix 
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

 Visualization saved as 'iris_results.png'

=======================================================
         Custom Prediction (Sample Test)
=======================================================
  Input   : Sepal L=5.1, Sepal W=3.5, Petal L=1.4, Petal W=0.2
  Predicted Species : SETOSA
  Confidence Setosa: 100.0% | Versicolor: 0.0% | Virginica: 0.0%

 Project 2 Complete! Badge Earned 

## Libraries Used
| Library | Purpose |
|---------|---------|
| numpy | calculations |
| pandas | data handling |
| matplotlib | graphs |
| seaborn | heatmap |
| scikit-learn | KNN algorithm |

## Author
Alina Zubair - AI Intern at DecodeLabs

## Date
DecodeLabs Internship Program 2026
